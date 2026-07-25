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
    def __init__(self, env: MarlEnv):
        super().__init__(env)
        self._env: MarlEnv

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

        # Unconditional reset (same key discipline as the old _auto_reset), then
        # select per leaf. Under vmap `done` is a scalar per environment, so the
        # where broadcasts over every leaf shape.
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
