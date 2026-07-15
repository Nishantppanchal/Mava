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

from typing import Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp

from mava.systems.ppo.types import PPOTransition, RNNPPOTransition


def calculate_gae(
    traj_batch: Union[PPOTransition, RNNPPOTransition],
    last_val: chex.Array,
    last_done: chex.Array,
    gamma: float,
    gae_lambda: float,
    unroll: int = 16,
    discount_traj: Optional[chex.Array] = None,
    next_val_traj: Optional[chex.Array] = None,
) -> Tuple[chex.Array, chex.Array]:
    """Computes truncated generalized advantage estimates.

    The advantages are computed in a backwards fashion according to the equation:
    Âₜ = δₜ + (γλ) * δₜ₊₁ + ... + ... + (γλ)ᵏ⁻ᵗ⁺¹ * δₖ₋₁
    where δₜ = rₜ₊₁ + γₜ₊₁ * v(sₜ₊₁) - v(sₜ).
    See Proximal Policy Optimization Algorithms, Schulman et al.:
    https://arxiv.org/abs/1707.06347

    Args:
        traj_batch (B, T, N, ...): a batch of trajectories.
        last_val  (B, N): value of the final timestep. Unused when
            ``next_val_traj`` is supplied (which already carries the bootstrap
            for every step, including the last); only its shape is read.
        last_done (B, N): whether the last timestep was a terminated or truncated.
        gamma (float): discount factor.
        gae_lambda (float): GAE mixing parameter.
        unroll (int): how much XLA should unroll the scan used to calculate GAE.
        discount_traj (T, B, N), optional: per-step discount AFTER each step
            (i.e. ``timestep.discount`` of the step's outcome). When provided,
            the bootstrap term is masked by this discount instead of by
            ``(1 - done)``, distinguishing true termination (discount=0) from
            time-limit truncation (discount=1) — the CleanRL PPO term-vs-trunc
            fix (https://github.com/vwxyzjn/cleanrl/pull/424). The
            episode-boundary mask on GAE accumulation still uses ``(1 - done)``
            so advantages don't leak across truncation boundaries. When
            ``None``, the legacy ``(1 - done)`` bootstrap mask is used and
            truncation is treated identically to termination.
        next_val_traj (T, B, N), optional: V(s_{t+1}) evaluated on the TRUE
            next observation of each step — i.e. the critic applied to
            ``timestep.extras["real_next_obs"]``. Requires ``discount_traj``.

            This is the other half of the CleanRL term-vs-trunc fix, and
            without it ``discount_traj`` alone is unsound. ``AutoResetWrapper``
            overwrites ``timestep.observation`` with the RESET observation on
            any episode end, stashing the true final observation in
            ``extras["real_next_obs"]``. The bootstrap V(s_{t+1}) is otherwise
            read from the next transition's ``value``, so on a truncation it is
            V(a brand-new episode) rather than V(the state actually reached).
            ``discount_traj`` unmasks that bootstrap (discount=1 on truncation),
            which turns a previously-inert corrupted value into a live one: a
            flat, state-independent bonus for reaching the time limit. Envs
            where the agent can choose between terminating and running out the
            clock will learn to run out the clock.

            On non-terminal steps ``real_next_obs`` IS the next observation, so
            this trajectory equals the next transition's value there and only
            changes the arithmetic at episode boundaries.

    Returns Tuple[(B, T, N), (B, T, N)]: advantages and target values.
    """
    if next_val_traj is not None and discount_traj is None:
        raise ValueError("next_val_traj requires discount_traj (it masks the bootstrap it feeds).")

    if discount_traj is None:
        # Legacy path — bootstrap mask is (1 - done); truncation is treated
        # as termination. Kept for backward-compat with callers that don't
        # carry a discount trajectory through the rollout.
        def _get_advantages(
            carry: Tuple[chex.Array, chex.Array, chex.Array],
            transition: RNNPPOTransition,
        ) -> Tuple[Tuple[chex.Array, chex.Array, chex.Array], chex.Array]:
            gae, next_value, next_done = carry
            done, value, reward = transition.done, transition.value, transition.reward

            delta = reward + gamma * next_value * (1 - next_done) - value
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae
            return (gae, value, done), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val, last_done),
            traj_batch,
            reverse=True,
            unroll=unroll,
        )
    elif next_val_traj is None:
        # Term-vs-trunc-aware path, bootstrapping the NEXT TRANSITION's value.
        # ``discount_traj[t]`` is the discount of the step that produced
        # s_{t+1}; for an env that uses Jumanji's ``termination()``
        # (discount=0), ``truncation()`` (discount=1) and ``transition()``
        # (discount=1), this is the right mask for the V(s_{t+1}) bootstrap.
        # The GAE-propagation mask stays ``(1 - done)`` so accumulating
        # advantages across an episode boundary (term OR trunc) is still cut.
        #
        # WARNING: only sound when the caller's next observation survives the
        # auto-reset — see the ``next_val_traj`` docstring. Under
        # ``AutoResetWrapper`` this bootstraps V(reset obs) on truncation.
        # Prefer passing ``next_val_traj``.
        def _get_advantages_discount(
            carry: Tuple[chex.Array, chex.Array, chex.Array],
            inputs: Tuple[RNNPPOTransition, chex.Array],
        ) -> Tuple[Tuple[chex.Array, chex.Array, chex.Array], chex.Array]:
            transition, step_discount = inputs
            gae, next_value, next_done = carry
            done, value, reward = transition.done, transition.value, transition.reward

            delta = reward + gamma * next_value * step_discount - value
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae
            return (gae, value, done), gae

        _, advantages = jax.lax.scan(
            _get_advantages_discount,
            (jnp.zeros_like(last_val), last_val, last_done),
            (traj_batch, discount_traj),
            reverse=True,
            unroll=unroll,
        )
    else:
        # Term-vs-trunc-aware path, bootstrapping V(real_next_obs). The full
        # CleanRL fix: the discount mask distinguishes term from trunc AND the
        # bootstrapped value is the true next state's, so an auto-reset can't
        # smuggle a fresh episode's value into the truncation bootstrap.
        # ``next_val_traj[t]`` already holds V(s_{t+1}) for every t, so no value
        # needs carrying between steps and ``last_val`` is unused.
        def _get_advantages_real_next(
            carry: Tuple[chex.Array, chex.Array],
            inputs: Tuple[RNNPPOTransition, chex.Array, chex.Array],
        ) -> Tuple[Tuple[chex.Array, chex.Array], chex.Array]:
            transition, step_discount, next_value = inputs
            gae, next_done = carry
            done, value, reward = transition.done, transition.value, transition.reward

            delta = reward + gamma * next_value * step_discount - value
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae
            return (gae, done), gae

        _, advantages = jax.lax.scan(
            _get_advantages_real_next,
            (jnp.zeros_like(last_val), last_done),
            (traj_batch, discount_traj, next_val_traj),
            reverse=True,
            unroll=unroll,
        )

    return advantages, advantages + traj_batch.value
