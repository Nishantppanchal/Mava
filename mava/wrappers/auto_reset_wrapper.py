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

# Note this is only here until this is merged into jumanji
# PR: https://github.com/instadeepai/jumanji/pull/223

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jumanji.env import State
from jumanji.types import TimeStep
from jumanji.wrappers import Observation, Wrapper

from mava.types import MarlEnv


class AutoResetWrapper(Wrapper):
    """Automatically resets environments that are done. Once the terminal state is reached,
    the state, observation, and step_type are reset. The observation and step_type of the
    terminal TimeStep is reset to the reset observation and StepType.LAST, respectively.
    The reward, discount, and extras retrieved from the transition to the terminal state.
    NOTE: The observation from the terminal TimeStep is stored in timestep.extras["real_next_obs"].
    WARNING: do not `jax.vmap` the wrapped environment (e.g. do not use with the `VmapWrapper`),
    which would lead to inefficient computation due to both the `step` and `reset` functions
    being processed each time `step` is called. Please use the `VmapAutoResetWrapper` instead.
    """

    OBS_IN_EXTRAS_KEY = "real_next_obs"

    # This init isn't really needed as jumanji.Wrapper will forward the attributes,
    # but mypy doesn't realize this.
    def __init__(self, env: MarlEnv, defer_reset: bool = False):
        """Fork (pursuit DESIGN §148s / WP9): ``defer_reset`` splits ``step``.

        With ``defer_reset=False`` — the default, and the stock behaviour —
        ``step`` is exactly what it always was: env step, latch
        ``real_next_obs`` into extras, then the unconditional reset + per-leaf
        select.

        With ``defer_reset=True`` ``step`` stops after the latch and the reset
        tail is exposed as :meth:`finish_auto_reset`, so that
        :class:`BatchAutoResetWrapper` can run it OUTSIDE the per-env ``vmap``
        under a ``lax.cond`` on ``jnp.any(done)``. Deferring is exact rather
        than approximate: the tail reads only ``state.key`` and
        ``timestep.observation``, while the wrapper it is deferred past
        (``RecordEpisodeMetrics``) reads only ``timestep.reward`` and
        ``timestep.step_type`` and threads the env state through untouched — so
        the two orderings commute bit for bit.
        """
        super().__init__(env)
        self._env: MarlEnv

        self.defer_reset = defer_reset
        self.num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit
        self.action_dim = self._env.action_dim

    def _obs_in_extras(
        self, state: State, timestep: TimeStep[Observation]
    ) -> Tuple[State, TimeStep[Observation]]:
        """Place the observation in timestep.extras[real_next_obs]."""
        extras = timestep.extras
        extras[AutoResetWrapper.OBS_IN_EXTRAS_KEY] = timestep.observation
        return state, timestep.replace(extras=extras)

    def _auto_reset(
        self, state: State, timestep: TimeStep[Observation]
    ) -> Tuple[State, TimeStep[Observation]]:
        """Reset the state and overwrite `timestep.observation` with the reset observation
        if the episode has terminated.
        """
        if not hasattr(state, "key"):
            raise AttributeError(
                "This wrapper assumes that the state has attribute key which is used"
                " as the source of randomness for automatic reset"
            )

        # Make sure that the random key in the environment changes at each call to reset.
        # State is a type variable hence it does not have key type hinted, so we type ignore.
        key, _ = jax.random.split(state.key)  # type: ignore
        state, reset_timestep = self._env.reset(key)

        # Place original observation in extras.
        state, timestep = self._obs_in_extras(state, timestep)

        # Replace observation with reset observation.
        timestep = timestep.replace(observation=reset_timestep.observation)  # type: ignore

        return state, timestep

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        return self._obs_in_extras(*super().reset(key))

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """Step the environment, with automatic resetting if the episode terminates.

        NOTE: implemented as unconditional-reset + per-leaf ``jnp.where`` select
        rather than ``jax.lax.cond``. Under ``jax.vmap`` (Anakin), a cond with a
        batched predicate executes both branches anyway, so this is semantically
        identical — but JAX's cond BATCHING RULE also broadcasts the branches'
        closure constants into batched operands. With an env that closes over a
        large lookup table that broadcast is catastrophic: the pursuit env's
        compact BFS table (1.07 GiB at G=200) was tiled x num_envs into a
        136.56 GiB ``u16[128,571975056]`` buffer inside the rollout while-loop.
        The select form keeps constants unbatched. See the pursuit repo's
        DESIGN.md §30.
        """
        state, timestep = self._env.step(state, action)

        # Both paths of the old cond stored the pre-reset observation in extras.
        state, timestep = self._obs_in_extras(state, timestep)

        # Fork (§148s / WP9): with ``defer_reset`` the tail below is run later,
        # once per BATCH, by ``BatchAutoResetWrapper``.
        if self.defer_reset:
            return state, timestep

        return self.finish_auto_reset(state, timestep)

    def finish_auto_reset(
        self, state: State, timestep: TimeStep[Observation]
    ) -> Tuple[State, TimeStep[Observation]]:
        """The reset tail of :meth:`step`, verbatim, as a separate method.

        Unconditional reset (same key discipline as the old ``_auto_reset``),
        then select per leaf. Under ``vmap`` ``done`` is a scalar per
        environment, so the where broadcasts over every leaf shape.
        """
        key, _ = jax.random.split(state.key)  # type: ignore
        reset_state, reset_timestep = self._env.reset(key)

        done = timestep.last()

        def select(r: chex.Array, s: chex.Array) -> chex.Array:
            return jnp.where(done, r, s)

        state = jax.tree_util.tree_map(select, reset_state, state)
        timestep = timestep.replace(  # type: ignore
            observation=jax.tree_util.tree_map(
                select, reset_timestep.observation, timestep.observation
            )
        )

        return state, timestep


class BatchAutoResetWrapper(Wrapper):
    """Batch-level conditional auto-reset — the OUTERMOST wrapper of the train stack.

    Fork-only (pursuit DESIGN §148s / WP9). ``AutoResetWrapper`` pays a full
    ``env.reset`` on every env on every step so that a per-leaf ``where`` can
    select it; §148n measured that at 40 % of the belief arm's step cost while
    being needed on well under 1 % of env-steps (episodes ~800 steps). This
    wrapper takes BATCHED state/action, does the per-env work under one
    ``jax.vmap``, and then runs the deferred reset tail of the inner
    ``AutoResetWrapper`` under ``jax.lax.cond(jnp.any(done), ...)``. With 128
    envs and ~800-step episodes the reset branch runs on ~15 % of steps instead
    of 100 %. The false branch is the identity, which is exactly what the old
    per-leaf ``where`` computed when no env was done.

    BIT-IDENTICAL TO THE OLD STACK, measured leaf by leaf (pursuit
    ``tests/test_wp9_batch_reset``): state, reward, discount, step_type, extras
    and both halves of the observation, at G=25 and G=200, 64 envs x 300 steps.

    Getting there needed one thing from the env, and it is worth knowing about
    before wrapping a different one. JAX lifts a branch's closure constants into
    OPERANDS of the conditional, and XLA's algebraic simplifier folds ``x / c``
    into ``x * (1/c)`` only while ``c`` is a constant — so a constant divisor
    inside the branch stays a true division while the same expression outside it
    became a multiply. The pursuit env's critic vector had three such divisors
    and now writes the multiply itself (``_bfs_large_recip``). If a future env
    shows a last-ULP difference under this wrapper, that is the shape of it: fix
    the source expression, don't loosen the test.

    THE CONSTRAINT (DESIGN §30, and the note on ``AutoResetWrapper.step``): the
    predicate must be a TRUE SCALAR. Under ``jax.vmap`` a ``cond`` becomes a
    ``select`` — both branches run, nothing is saved — and, worse, JAX's cond
    batching rule broadcasts branch closure constants into batched operands,
    which tiled the 1.07 GiB BFS table x num_envs last time. So this wrapper
    must never be used inside a ``vmap``: ``rec_mappo`` detects
    ``batched_step`` and skips its own ``jax.vmap(env.step)``, and refuses to
    run with ``system.update_batch_size > 1`` (which would vmap the whole
    update step around it). ``pmap`` over devices is fine — each device sees a
    scalar predicate.

    ``reset`` is batched too (it takes a batch of keys), so the caller's stack
    is symmetric.
    """

    #: Read by ``rec_mappo`` to skip its own per-env ``jax.vmap``.
    batched_step = True

    def __init__(self, env: MarlEnv):
        super().__init__(env)
        self._env: MarlEnv

        if not getattr(env, "defer_reset", False):
            raise ValueError(
                "BatchAutoResetWrapper must wrap a stack containing an "
                "AutoResetWrapper built with defer_reset=True — otherwise the "
                "inner wrapper still resets every env on every step and this "
                "wrapper would reset them a second time."
            )

        self.num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit
        self.action_dim = self._env.action_dim

    def reset(self, keys: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Reset a BATCH of environments, one per key in ``keys``."""
        return jax.vmap(self._env.reset)(keys)

    def step(self, states: State, actions: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """Step a BATCH of environments, resetting only if some env is done."""
        states, timesteps = jax.vmap(self._env.step)(states, actions)

        # ``finish_auto_reset`` is forwarded through the inner wrappers by
        # ``jumanji.Wrapper.__getattr__`` (each one threads it through its own
        # state), so this reaches the deferred ``AutoResetWrapper``.
        finish = self._env.finish_auto_reset

        def reset_and_select(
            operand: Tuple[State, TimeStep[Observation]],
        ) -> Tuple[State, TimeStep[Observation]]:
            return jax.vmap(finish)(*operand)

        def identity(
            operand: Tuple[State, TimeStep[Observation]],
        ) -> Tuple[State, TimeStep[Observation]]:
            return operand

        return jax.lax.cond(
            jnp.any(timesteps.last()),
            reset_and_select,
            identity,
            (states, timesteps),
        )
