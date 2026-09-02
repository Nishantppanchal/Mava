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

from mava.networks.distributions import IdentityTransformation, MaskedEpsGreedyDistribution
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

    Fork §51/§97: ``temporal_core`` selects the memory module — "gru"
    (stock ScannedRNN, default; zero behavior change) or "dual_gru"
    (ScannedDualGRU, the §80 phase-gated dual-timescale core). The §51/§53
    window-attention and SSM cores were removed in §97 (both closed
    negative, DESIGN §80/§82b) — git history has them.

    Fork §53: ``aux_predict`` adds an auxiliary evader-position head off the
    temporal core (Dense(2) on the recurrent embedding), exposed ONLY via
    flax's "intermediates" sow — the (hstate, pi) return contract is
    untouched, so every existing consumer (evaluator, render, BC, eval
    scripts) is unaffected; the PPO loss opts in with
    ``mutable=["intermediates"]`` and trains it supervised on the true
    future evader position (dense hindsight signal shaping the shared
    representation toward route prediction).

    Fork §138/WP3: ``return_core`` makes ``__call__`` return a THIRD element,
    the post-GRU core embedding (the same tensor the §53 aux head reads, before
    ``post_torso``). It exists for ``GatedResidualActor``, whose residual has
    to condition on what the champion's recurrent state already knows — a sow
    cannot be read back inside the same forward pass, and re-running the
    champion would double the cost. Params are untouched by the flag, so a
    checkpoint grafts in either way.
    """

    pre_torso: nn.Module
    post_torso: nn.Module
    action_head: nn.Module
    hidden_state_dim: int = 128
    temporal_core: str = "gru"
    aux_predict: bool = False
    return_core: bool = False

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
        if self.temporal_core == "dual_gru":
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
        core_embedding = policy_embedding
        policy_embedding = self.post_torso(policy_embedding)
        pi = self.action_head(policy_embedding, action_mask)

        if self.return_core:
            return policy_hidden_state, pi, core_embedding
        return policy_hidden_state, pi


class GatedResidualActor(nn.Module):
    """Fork §134/§138: a frozen champion plus a gated residual.

        merged = champion(head) + g * clip(delta, -delta_clip, +delta_clip)

    ``gate="hard"`` is §134f, unchanged and still the default: ``g`` is the
    environment's hard 0/1 alarm column, so on a team with no liars it is
    exactly 0, the product is exactly 0, and IEEE addition of 0.0 leaves the
    champion's logits bit-identical. That was a GUARANTEE, and §138 records
    why it was too strong to keep as the only option: an alarm-gated residual
    can only act after the symbolic filter has already convicted someone, so
    its ceiling is the damage that survives detection -- measured at +0.0020
    [-0.0042, +0.0079], i.e. nothing.

    ``gate="soft"`` replaces the guarantee with a BUDGET. ``g`` is a learned
    sigmoid, biased at ``gate_bias0`` (default -4.0, ~2 % open) so the policy
    starts almost closed, and ``delta`` comes from a zero-initialised head, so
    at init ``merged`` equals the champion EXACTLY regardless of ``g`` and the
    first PPO ratio is 1. What keeps the honest cost small afterwards is not
    the architecture but ``system.honest_kl_coef``, a KL penalty against the
    champion on honest-episode steps, plus ``delta_clip`` on the logit
    displacement. "Cannot regress" becomes "regresses by at most epsilon,
    measured" -- §138's decision, taken with the user.

    ``full_view`` (§138/WP3) decides what the residual READS. The §134f
    residual saw only the tail window, which is enough to notice a liar and
    nothing like enough to do anything about one: a policy that cannot see its
    own pose, the local map or the per-source reports cannot choose a
    verification detour or a search pattern. With ``full_view`` it reads the
    whole ``agents_view`` (head AND tail) concatenated with the champion's
    post-GRU core embedding (stop-gradiented, so the frozen half stays frozen
    and the residual inherits the champion's memory for free).

    The champion always sees ``agents_view`` with the trust tail cut out, which
    is bit-identical to the view it was trained on, so a pre-trust checkpoint
    grafts in unchanged. The recurrent carry is the champion's, so every
    evaluator, renderer and probe works untouched.

    Sows ``gate`` and ``champion_logits`` into "intermediates" for the loss
    (the honest-episode KL needs the reference logits) and for logging.
    """

    champion: nn.Module  # a RecurrentActor, frozen
    residual_torso: nn.Module
    action_dim: int
    tail_start: int
    tail_width: int
    hidden_state_dim: int = 128
    freeze_champion: bool = True
    gate: str = "hard"
    gate_bias0: float = -4.0
    delta_clip: float = 5.0
    full_view: bool = False

    @nn.compact
    def __call__(
        self,
        policy_hidden_state: chex.Array,
        observation_done: RNNObservation,
    ) -> Tuple[chex.Array, tfd.Distribution]:
        if self.gate not in ("hard", "soft"):
            raise ValueError(
                f"residual.gate must be 'hard' or 'soft', got {self.gate!r}"
            )
        observation, done = observation_done
        view = observation.agents_view
        lo, hi = self.tail_start, self.tail_start + self.tail_width
        head = jnp.concatenate([view[..., :lo], view[..., hi:]], axis=-1)
        tail = view[..., lo:hi]

        champion_out = self.champion(
            policy_hidden_state, (observation._replace(agents_view=head), done)
        )
        if len(champion_out) == 3:
            policy_hidden_state, pi, core = champion_out
        else:
            policy_hidden_state, pi = champion_out
            core = None
        logits = pi.distribution.logits
        if self.freeze_champion:
            logits = jax.lax.stop_gradient(logits)

        if self.full_view:
            if core is None:
                raise ValueError(
                    "residual.full_view needs the champion's core embedding: "
                    "build the champion with RecurrentActor(return_core=True)"
                )
            # stop_gradient on BOTH halves of the frozen network's contribution.
            # Without it the residual's loss would reach back through the GRU
            # into the champion, and "frozen" would be true of the logits only.
            residual_in = jnp.concatenate([view, jax.lax.stop_gradient(core)], axis=-1)
        else:
            residual_in = tail
        residual_out = self.residual_torso(residual_in)

        # Zero-init: the residual starts as an exact no-op, so the first PPO
        # ratio is 1 even on the steps where the gate IS open. A randomly
        # initialised head would move the policy before it had learned
        # anything, which is the usual way a "safe" fine-tune destroys a
        # champion in its first update.
        r = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="residual_head",
        )(residual_out)
        alarm = tail[..., -1:]  # the env writes this LAST in the tail block
        if self.gate == "hard":
            g = alarm
            delta = r
        else:
            # Zero kernel + constant bias: the gate is a pure function of the
            # bias at init (~2 % open at -4.0) and cannot depend on the input
            # until it has learned to, so the init identity holds for a reason
            # independent of the head's zero-init.
            g = nn.sigmoid(
                nn.Dense(
                    1,
                    kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.constant(self.gate_bias0),
                    name="residual_gate",
                )(residual_out)
            )
            delta = jnp.clip(r, -self.delta_clip, self.delta_clip)
        masked_champion = jnp.where(
            observation.action_mask, logits, jnp.finfo(jnp.float32).min
        )
        merged = jnp.where(
            observation.action_mask,
            logits + g * delta,
            jnp.finfo(jnp.float32).min,
        )
        # Read by _actor_loss_fn (the honest-episode KL) and by the loggers.
        # Sowing is a no-op unless the caller passes mutable=["intermediates"],
        # so this costs the rollout nothing.
        self.sow("intermediates", "gate", g)
        self.sow("intermediates", "champion_logits", masked_champion)
        return policy_hidden_state, IdentityTransformation(
            distribution=tfd.Categorical(logits=merged)
        )


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
        value_embedding = self.post_torso(value_embedding)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(value_embedding)

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
