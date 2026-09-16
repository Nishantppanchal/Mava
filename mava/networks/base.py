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
import numpy as np
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

    Fork: ``temporal_core`` selects the memory module. Only "gru"
    (stock ScannedRNN, the default) remains — window-attention, SSM and
    phase-gated dual-timescale cores were all tried, all closed negative and
    all deleted; git history has them.

    Fork: ``aux_predict`` adds an auxiliary evader-position head off the
    temporal core (Dense(2) on the recurrent embedding), exposed ONLY via
    flax's "intermediates" sow — the (hstate, pi) return contract is
    untouched, so every existing consumer (evaluator, render, BC, eval
    scripts) is unaffected; the PPO loss opts in with
    ``mutable=["intermediates"]`` and trains it supervised on the true
    future evader position (dense hindsight signal shaping the shared
    representation toward route prediction).

    Fork: ``return_core`` makes ``__call__`` return a THIRD element,
    the post-GRU core embedding (the same tensor the aux head reads, before
    ``post_torso``) — the only way to read that tensor inside the forward pass
    that produced it, since a sow cannot be read back. Its original consumer,
    a residual adapter, is gone. Params are untouched by the
    flag, so a checkpoint grafts in either way.

    Fork: ``target_gate`` is the LEARNED, differentiable replacement for the
    environment's hard entropy gate
    (``env.kwargs.belief_target_entropy_max``). Before the pre-torso the actor
    computes, per agent and per step,

        g = sigmoid(k * (h0 - H) + w . c + b)

    with ``H`` the row marginal's normalised entropy and ``c`` the remaining
    confidence features (the mode's basin mass, the credibility of the reporter
    whose claim is nearest it, own-sensor agreement, the fused sighting's
    freshness) — both read from ``conf_slices``, which the ENV derives from its
    own layout table (``PursuitEvasionEnvMava.actor_conf_slice``; the entropy
    range is LAST by contract). Every column in ``target_slices`` is then
    multiplied by ``g``; the confidence columns themselves are left UNGATED, so
    the gate can always see what it is deciding on, and no other column moves.

    ``k``, ``h0``, ``w`` and ``b`` are learnable and initialised at
    ``gate_init_slope``, ``gate_init_threshold``, 0 and 0, so at init the
    module reproduces the hard gate: at ``k = 200`` and ``h0 = 0.98`` a row at
    H = 0.90 passes at g > 0.999 and one at H = 1.0 is 98 % closed
    (sigmoid(-4) = 0.018). What it BUYS over the hard gate is a gradient: the
    threshold, the sharpness and a linear correction on the other confidence
    features are all trained by the policy loss, so "when is a belief target
    worth following" stops being a swept constant.

    The flag is OBS-NEUTRAL (no width moves) and, when False, creates NO
    params — the tree and every output are bit-identical to plain
    ``rnn_pursuit``, which is the ablation's control.
    """

    pre_torso: nn.Module
    post_torso: nn.Module
    action_head: nn.Module
    hidden_state_dim: int = 128
    temporal_core: str = "gru"
    aux_predict: bool = False
    return_core: bool = False
    # Fork. Slices are (start, width) pairs in the env's own layout,
    # handed down by learner_setup from that layout table.
    target_gate: bool = False
    target_slices: Tuple[Tuple[int, int], ...] = ()
    conf_slices: Tuple[Tuple[int, int], ...] = ()
    gate_init_threshold: float = 0.98
    gate_init_slope: float = 200.0

    def _gate_targets(self, view: chex.Array) -> chex.Array:
        """Fork: scale the target-derived columns of ``view`` by ``g``.

        Built as a multiplicative MASK over the full width rather than a
        scatter of slices: the confidence columns live INSIDE the belief-mode
        block (which is itself a target block), so "gate the targets, not the
        confidence" is one vector of 1s and gs and not an ordering puzzle.
        """
        if not self.target_slices or not self.conf_slices:
            raise ValueError(
                "network.target_gate.enabled needs the env's target and "
                "confidence slices: set env.kwargs.belief_modes_obs > 0 beside "
                "the routing family (fused_sighting, route_to_*)"
            )
        conf = jnp.concatenate(
            [view[..., s : s + w] for s, w in self.conf_slices], axis=-1
        )
        # The entropy range is LAST by the env-side contract, so H needs no
        # second copy of the belief-mode slot's offsets here.
        entropy, rest = conf[..., -1:], conf[..., :-1]
        k = self.param(
            "tgate_k", nn.initializers.constant(self.gate_init_slope), (), jnp.float32
        )
        h0 = self.param(
            "tgate_h0",
            nn.initializers.constant(self.gate_init_threshold),
            (),
            jnp.float32,
        )
        w = self.param(
            "tgate_w", nn.initializers.zeros, (rest.shape[-1],), jnp.float32
        )
        b = self.param("tgate_b", nn.initializers.zeros, (), jnp.float32)
        logit = k * (h0 - entropy) + jnp.sum(rest * w, axis=-1, keepdims=True) + b
        g = nn.sigmoid(logit)
        mask = np.zeros((view.shape[-1],), dtype=np.float32)
        for s, wd in self.target_slices:
            mask[s : s + wd] = 1.0
        for s, wd in self.conf_slices:
            mask[s : s + wd] = 0.0
        # Read by the learner's telemetry (target_gate_mean / _h0 / _k). Sowing
        # is a no-op unless the caller passes mutable=["intermediates"].
        self.sow("intermediates", "target_gate", g)
        self.sow("intermediates", "tgate_h0", h0)
        self.sow("intermediates", "tgate_k", k)
        return view * (1.0 + jnp.asarray(mask) * (g - 1.0))

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
            view = observation.agents_view
            if self.target_gate:
                view = self._gate_targets(view)
            policy_embedding = self.pre_torso(view)
            action_mask = observation.action_mask

        policy_rnn_input = (policy_embedding, done)
        policy_hidden_state, policy_embedding = ScannedRNN(self.hidden_state_dim)(
            policy_hidden_state, policy_rnn_input
        )
        if self.aux_predict:
            # The aux head reads the CORE output (pre post-torso) so the
            # gradient shapes the recurrent representation itself.
            aux = nn.Dense(2, name="aux_evader_head")(policy_embedding)
            self.sow("intermediates", "aux_evader_pred", aux)
        core_embedding = policy_embedding
        policy_embedding = self.post_torso(policy_embedding)
        pi = self.action_head(policy_embedding, action_mask)

        if self.return_core:
            return policy_hidden_state, pi, core_embedding
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
