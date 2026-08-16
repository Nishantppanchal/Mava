# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from typing import Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn
from flax.linen.initializers import orthogonal

from mava.networks.distributions import MaskedEpsGreedyDistribution
from mava.networks.torsos import MLPTorso
from mava.types import (
    GraphObservation,
    MavaObservation,
    Observation,
    ObservationGlobalState,
    RNNGlobalObservation,
    RNNObservation,
)
from mava.utils.graph.gnn_utils import is_graph_observation, validate_graph_components


class FeedForwardActor(nn.Module):
    """Feed Forward Actor Network."""

    torso: nn.Module
    action_head: nn.Module

    @nn.compact
    def __call__(
        self, observation: Union[Observation, GraphObservation[Observation]]
    ) -> tfd.Distribution:
        """Forward pass."""

        if is_graph_observation(observation):
            validate_graph_components(self.torso, observation)
            obs_embedding = self.torso(observation)
            action_mask = observation.observation.action_mask
        else:
            obs_embedding = self.torso(observation.agents_view)
            action_mask = observation.action_mask
        return self.action_head(obs_embedding, action_mask)


class FeedForwardValueNet(nn.Module):
    """Feedforward Value Network. Returns the value of an observation."""

    torso: nn.Module
    centralised_critic: bool = False

    @nn.compact
    def __call__(
        self,
        observation: Union[Observation, ObservationGlobalState, GraphObservation[MavaObservation]],
    ) -> chex.Array:
        """Forward pass."""

        if is_graph_observation(observation):
            validate_graph_components(self.torso, observation)
            critic_output = self.torso(observation)
        else:
            if self.centralised_critic:
                if not isinstance(observation, ObservationGlobalState):
                    raise ValueError("Global state must be provided to the centralised critic.")
                # Get global state in the case of a centralised critic.
                observation = observation.global_state
            else:
                # Get single agent view in the case of a decentralised critic.
                observation = observation.agents_view
            critic_output = self.torso(observation)
        critic_output = nn.Dense(1, kernel_init=orthogonal(1.0))(critic_output)

        return jnp.squeeze(critic_output, axis=-1)


class FeedForwardQNet(nn.Module):
    """Feedforward Q Network. Returns the value of an observation-action pair."""

    torso: nn.Module
    centralised_critic: bool = False

    def setup(self) -> None:
        self.critic = nn.Dense(1, kernel_init=orthogonal(1.0))

    def __call__(
        self,
        observation: Union[Observation, ObservationGlobalState],
        action: chex.Array,
    ) -> chex.Array:
        if self.centralised_critic:
            if not isinstance(observation, ObservationGlobalState):
                raise ValueError("Global state must be provided to the centralised critic.")
            # Get global state in the case of a centralised critic.
            observation = observation.global_state
        else:
            # Get single agent view in the case of a decentralised critic.
            observation = observation.agents_view

        x = jnp.concatenate([observation, action], axis=-1)
        x = self.torso(x)
        y = self.critic(x)

        return jnp.squeeze(y, axis=-1)


class ScannedRNN(nn.Module):
    hidden_state_dim: int = 128

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: chex.Array, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, :, jnp.newaxis],
            self.initialize_carry((ins.shape[0], ins.shape[1]), self.hidden_state_dim),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[-1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size: Sequence[int], hidden_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (*batch_size, hidden_size))


class RecurrentActor(nn.Module):
    """Recurrent Actor Network.

    Fork §51: ``temporal_core`` selects the memory module — "gru" (stock
    ScannedRNN, default; zero behavior change) or "window_attention"
    (ScannedWindowAttention; ``hidden_state_dim`` must then equal
    attn_window * (attn_token_dim + 1) so zero-carry init sites work
    unchanged).

    Fork §53: ``aux_predict`` adds an auxiliary evader-position head off the
    temporal core (Dense(2) on the recurrent embedding), exposed ONLY via
    flax's "intermediates" sow — the (hstate, pi) return contract is
    untouched, so every existing consumer (evaluator, render, BC, eval
    scripts) is unaffected; the PPO loss opts in with
    ``mutable=["intermediates"]`` and trains it supervised on the true
    future evader position (dense hindsight signal shaping the shared
    representation toward route prediction).
    """

    pre_torso: nn.Module
    post_torso: nn.Module
    action_head: nn.Module
    hidden_state_dim: int = 128
    temporal_core: str = "gru"
    attn_token_dim: int = 128
    attn_window: int = 64
    attn_heads: int = 4
    aux_predict: bool = False

    @nn.compact
    def __call__(
        self,
        policy_hidden_state: chex.Array,
        observation_done: RNNObservation,
    ) -> Tuple[chex.Array, tfd.Distribution]:
        """Forward pass."""
        observation, done = observation_done

        if is_graph_observation(observation):
            validate_graph_components(self.pre_torso, observation)
            policy_embedding = self.pre_torso(observation)
            action_mask = observation.observation.action_mask
        else:
            policy_embedding = self.pre_torso(observation.agents_view)
            action_mask = observation.action_mask

        policy_rnn_input = (policy_embedding, done)
        if self.temporal_core == "window_attention":
            policy_hidden_state, policy_embedding = ScannedWindowAttention(
                hidden_state_dim=self.hidden_state_dim,
                token_dim=self.attn_token_dim,
                window=self.attn_window,
                n_heads=self.attn_heads,
            )(policy_hidden_state, policy_rnn_input)
        elif self.temporal_core == "ssm":
            # Fork §53: diagonal gated selective SSM (Mamba-lineage).
            policy_hidden_state, policy_embedding = ScannedSSM(
                hidden_state_dim=self.hidden_state_dim
            )(policy_hidden_state, policy_rnn_input)
        elif self.temporal_core == "dual_gru":
            # Fork §80: phase-gated dual-timescale GRU (see ScannedDualGRU).
            policy_hidden_state, policy_embedding = ScannedDualGRU(
                hidden_state_dim=self.hidden_state_dim
            )(policy_hidden_state, policy_rnn_input)
        else:
            policy_hidden_state, policy_embedding = ScannedRNN(self.hidden_state_dim)(
                policy_hidden_state, policy_rnn_input
            )
        if self.aux_predict:
            # §53 aux head reads the CORE output (pre post-torso) so the
            # gradient shapes the recurrent representation itself.
            aux = nn.Dense(2, name="aux_evader_head")(policy_embedding)
            self.sow("intermediates", "aux_evader_pred", aux)
        policy_embedding = self.post_torso(policy_embedding)
        pi = self.action_head(policy_embedding, action_mask)

        return policy_hidden_state, pi


class RecurrentValueNet(nn.Module):
    """Recurrent Critic Network."""

    pre_torso: nn.Module
    post_torso: nn.Module
    centralised_critic: bool = False
    hidden_state_dim: int = 128

    @nn.compact
    def __call__(
        self,
        value_net_hidden_state: Tuple[chex.Array, chex.Array],
        observation_done: Union[RNNObservation, RNNGlobalObservation],
    ) -> Tuple[chex.Array, chex.Array]:
        """Forward pass."""
        observation, done = observation_done

        if is_graph_observation(observation):
            validate_graph_components(self.pre_torso, observation)
            value_embedding = self.pre_torso(observation)
        else:
            if self.centralised_critic:
                if not isinstance(observation, ObservationGlobalState):
                    raise ValueError("Global state must be provided to the centralised critic.")
                # Get global state in the case of a centralised critic.
                observation = observation.global_state
            else:
                # Get single agent view in the case of a decentralised critic.
                observation = observation.agents_view

            value_embedding = self.pre_torso(observation)

        value_rnn_input = (value_embedding, done)
        value_net_hidden_state, value_embedding = ScannedRNN(self.hidden_state_dim)(
            value_net_hidden_state, value_rnn_input
        )
        value = self.post_torso(value_embedding)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(value)

        return value_net_hidden_state, jnp.squeeze(value, axis=-1)


class RecQNetwork(nn.Module):
    """Recurrent Q-Network."""

    pre_torso: nn.Module
    post_torso: nn.Module
    num_actions: int
    hidden_state_dim: int = 128

    @nn.compact
    def get_q_values(
        self,
        hidden_state: chex.Array,
        observations_resets: RNNObservation,
    ) -> chex.Array:
        """Forward pass to obtain q values."""
        obs, resets = observations_resets

        assert not is_graph_observation(obs), "GraphObservation is not supported for RecQNetwork"

        embedding = self.pre_torso(obs.agents_view)

        rnn_input = (embedding, resets)
        hidden_state, embedding = ScannedRNN(self.hidden_state_dim)(hidden_state, rnn_input)

        embedding = self.post_torso(embedding)

        q_values = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01))(embedding)

        return hidden_state, q_values

    def __call__(
        self,
        hidden_state: chex.Array,
        observations_resets: RNNObservation,
        eps: float = 0,
    ) -> chex.Array:
        """Forward pass with additional construction of epsilon-greedy distribution.
        When epsilon is not specified, we assume a greedy approach.
        """
        obs, _ = observations_resets
        assert not is_graph_observation(obs), "GraphObservation is not supported for RecQNetwork"
        hidden_state, q_values = self.get_q_values(hidden_state, observations_resets)
        eps_greedy_dist = MaskedEpsGreedyDistribution(q_values, eps, obs.action_mask)

        return hidden_state, eps_greedy_dist


class QMixingNetwork(nn.Module):
    num_actions: int
    num_agents: int
    hyper_hidden_dim: int = 64
    embed_dim: int = 32
    norm_env_states: bool = True

    def setup(self) -> None:
        self.hyper_w1: MLPTorso = MLPTorso(
            (self.hyper_hidden_dim, self.embed_dim * self.num_agents),
            activate_final=False,
        )

        self.hyper_b1: MLPTorso = MLPTorso(
            (self.embed_dim,),
            activate_final=False,
        )

        self.hyper_w2: MLPTorso = MLPTorso(
            (self.hyper_hidden_dim, self.embed_dim),
            activate_final=False,
        )

        self.hyper_b2: MLPTorso = MLPTorso(
            (self.embed_dim, 1),
            activate_final=False,
        )

        self.layer_norm: nn.Module = nn.LayerNorm()

    @nn.compact
    def __call__(
        self,
        agent_qs: chex.Array,
        env_global_state: chex.Array,
    ) -> chex.Array:
        B, T = agent_qs.shape[:2]  # batch size

        agent_qs = jnp.reshape(agent_qs, (B, T, 1, self.num_agents))

        if self.norm_env_states:
            states = self.layer_norm(env_global_state)
        else:
            states = env_global_state

        # First layer
        w1 = jnp.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = jnp.reshape(w1, (B, T, self.num_agents, self.embed_dim))
        b1 = jnp.reshape(b1, (B, T, 1, self.embed_dim))

        # Matrix multiplication
        hidden = nn.elu(jnp.matmul(agent_qs, w1) + b1)

        # Second layer
        w2 = jnp.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = jnp.reshape(w2, (B, T, self.embed_dim, 1))
        b2 = jnp.reshape(b2, (B, T, 1, 1))

        # Compute final output
        y = jnp.matmul(hidden, w2) + b2

        # Reshape
        q_tot = jnp.reshape(y, (B, T, 1))

        return q_tot


class ScannedWindowAttention(nn.Module):
    """Sliding-window temporal attention cell — a GRU-replacement (fork §51).

    Same outer interface as ``ScannedRNN`` (scan over (embedding, done);
    carry created by zeros of ``hidden_state_dim``): the carry is a FLAT
    ``(B, N, window * (token_dim + 1))`` array packing a KV cache of the
    last ``window`` input tokens plus a per-slot validity mask, so
    ``ScannedRNN.initialize_carry`` (zeros == empty memory) works unchanged
    at every call site. Each step: project the input to ``token_dim``,
    attend (with learned slot-position embeddings, newest slot last) over
    the valid cached tokens + the current token, residual + LayerNorm +
    FFN, emit; then roll the cache. ``done`` zeroes the cache (episode
    reset), matching GRU reset semantics. Causality is inherent to the
    scan. ``hidden_state_dim`` MUST equal ``window * (token_dim + 1)`` —
    asserted at call time.
    """

    hidden_state_dim: int = 8256
    token_dim: int = 128
    window: int = 64
    n_heads: int = 4

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    # Fork §53: scan-over-remat. Without it, BPTT stores every step's
    # attention internals — a single ~10GB residual buffer at 128 envs
    # (observed OOM). Remat recomputes the cell in the backward pass from
    # (carry, input) instead; memory drops ~T-fold for ~30% extra compute.
    @nn.remat
    @nn.compact
    def __call__(self, carry: chex.Array, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        ins, resets = x
        M, d = self.window, self.token_dim
        assert self.hidden_state_dim == M * (d + 1), (
            f"hidden_state_dim must be window*(token_dim+1) = {M * (d + 1)}, "
            f"got {self.hidden_state_dim}"
        )
        carry = jnp.where(resets[:, :, jnp.newaxis], jnp.zeros_like(carry), carry)
        lead = carry.shape[:-1]  # (B, N)
        buffer = carry[..., : M * d].reshape(*lead, M, d)
        valid = carry[..., M * d :]  # (B, N, M)

        tok = nn.Dense(d)(ins)  # (B, N, d)
        pos = self.param(
            "pos_emb", nn.initializers.normal(0.02), (M + 1, d)
        )
        kv = jnp.concatenate([buffer, tok[..., None, :]], axis=-2) + pos
        kv_valid = jnp.concatenate(
            [valid, jnp.ones((*lead, 1), valid.dtype)], axis=-1
        )  # (B, N, M+1)
        attn_mask = kv_valid[..., None, None, :] > 0.5  # (B, N, 1, 1, M+1)
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=d
        )(tok[..., None, :], kv, mask=attn_mask)[..., 0, :]
        y = nn.LayerNorm()(tok + attended)
        y = nn.LayerNorm()(y + nn.Dense(d)(nn.relu(nn.Dense(2 * d)(y))))

        new_buffer = jnp.concatenate([buffer[..., 1:, :], tok[..., None, :]], axis=-2)
        new_valid = jnp.concatenate(
            [valid[..., 1:], jnp.ones((*lead, 1), valid.dtype)], axis=-1
        )
        new_carry = jnp.concatenate(
            [new_buffer.reshape(*lead, M * d), new_valid], axis=-1
        )
        return new_carry, y


class ScannedSSM(nn.Module):
    """Diagonal gated selective SSM cell — a GRU replacement (fork §53).

    Mamba-lineage recurrence in ``ScannedRNN``'s exact interface: the carry
    is the flat ``(B, N, hidden_state_dim)`` diagonal state (zeros == empty
    memory, so every ``initialize_carry`` site works unchanged) and ``done``
    zeroes it. Per step, input-dependent (selective) step sizes modulate a
    learned stable per-channel decay:

        dt = softplus(W_dt x)                    input-dependent step size
        a  = exp(-softplus(A_log) * dt)          decay in (0, 1)
        h' = a * h + (1 - a) * (W_in x)          ZOH-style state update
        y  = LayerNorm(h' * silu(W_gate x))      gated readout

    Output width == ``hidden_state_dim`` (the GRU convention), so post-torsos
    and all hstate plumbing are untouched. Per-step internals are O(S) — no
    remat needed (nothing like the attention cell's window buffers).
    """

    hidden_state_dim: int = 128

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: chex.Array, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        ins, resets = x
        carry = jnp.where(resets[:, :, jnp.newaxis], jnp.zeros_like(carry), carry)
        s = self.hidden_state_dim
        assert carry.shape[-1] == s, (
            f"ScannedSSM carry width {carry.shape[-1]} != hidden_state_dim {s}"
        )
        dt = jax.nn.softplus(nn.Dense(s, name="dt_proj")(ins))
        a_log = self.param("A_log", nn.initializers.normal(0.5), (s,))
        a = jnp.exp(-jax.nn.softplus(a_log) * dt)
        h = a * carry + (1.0 - a) * nn.Dense(s, name="in_proj")(ins)
        y = nn.LayerNorm()(h * jax.nn.silu(nn.Dense(s, name="gate_proj")(ins)))
        return h, y


class ScannedDualGRU(nn.Module):
    """Phase-gated dual-timescale GRU — a GRU replacement (fork §80).

    Two half-width GRU cells in ``ScannedRNN``'s exact interface. The FAST
    core updates every step (reactive chase geometry). The SLOW core's update
    is per-unit gated by a learned, input-conditioned rate g in (0, 1) — an
    adaptive-timescale (leaky) recurrence that can hold search-phase memory
    across hundreds of steps while the fast core churns. The gate reads the
    observation embedding, which carries the contact-phase signal
    (fused-sighting validity), so the network can learn to run the slow core
    open during search and nearly frozen in-chase — the two-phase task
    anatomy (§71/§74) expressed as architecture. Gate bias -2 initialises
    slow-core rates near 0.12 (~8-step timescale), learnable per unit.

    Carry = concat(h_fast, h_slow), width == ``hidden_state_dim``, zeros ==
    empty memory — every ``initialize_carry`` site and done-reset works
    unchanged; output width matches the GRU convention so post-torsos and
    hstate plumbing are untouched.
    """

    hidden_state_dim: int = 128

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: chex.Array, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        ins, resets = x
        carry = jnp.where(resets[:, :, jnp.newaxis], jnp.zeros_like(carry), carry)
        s = self.hidden_state_dim
        assert s % 2 == 0, f"dual_gru needs an even hidden_state_dim, got {s}"
        assert carry.shape[-1] == s, (
            f"ScannedDualGRU carry width {carry.shape[-1]} != hidden_state_dim {s}"
        )
        half = s // 2
        h_fast, h_slow = carry[..., :half], carry[..., half:]
        new_fast, _ = nn.GRUCell(features=half, name="fast")(h_fast, ins)
        cand_slow, _ = nn.GRUCell(features=half, name="slow")(h_slow, ins)
        g = jax.nn.sigmoid(nn.Dense(half, name="rate_gate")(ins) - 2.0)
        new_slow = (1.0 - g) * h_slow + g * cand_slow
        new_carry = jnp.concatenate([new_fast, new_slow], axis=-1)
        return new_carry, new_carry
