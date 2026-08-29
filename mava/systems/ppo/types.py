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


import chex
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from optax._src.base import OptState
from typing import Optional

from typing_extensions import NamedTuple

from mava.types import Action, Done, HiddenState, Observation, State, Value


class Params(NamedTuple):
    """Parameters of an actor critic network."""

    actor_params: FrozenDict
    critic_params: FrozenDict


class OptStates(NamedTuple):
    """OptStates of actor critic learner."""

    actor_opt_state: OptState
    critic_opt_state: OptState


class HiddenStates(NamedTuple):
    """Hidden states for an actor critic learner."""

    policy_hidden_state: HiddenState
    critic_hidden_state: HiddenState


class LearnerState(NamedTuple):
    """State of the learner."""

    params: Params
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    dones: Done


class SebulbaLearnerState(NamedTuple):
    """State of the learner."""

    params: Params
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep


class RNNLearnerState(NamedTuple):
    """State of the `Learner` for recurrent architectures."""

    params: Params
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    dones: Done
    hstates: HiddenStates


class PPOTransition(NamedTuple):
    """Transition tuple for PPO."""

    done: Done
    action: Action
    value: Value
    reward: chex.Array
    log_prob: chex.Array
    obs: Observation


class RNNPPOTransition(NamedTuple):
    """Transition tuple for PPO."""

    done: Done
    action: Action
    value: Value
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    hstates: HiddenStates
    # Fork (trust-aware-drone-pursuit-evasion §131d): per-agent mask marking
    # whose transitions the POLICY loss may train on. The env supplies it as
    # ``extras["learn_mask"]`` when a compromised agent is present: that agent
    # runs the shared policy on a deliberately falsified input, so training on
    # its rows teaches the policy how to behave while believing a phantom.
    # ``None`` (the default, and every run that does not set the env flag)
    # means "train on everything" and is an exact no-op — None is an empty
    # pytree node, so the trajectory structure is unchanged.
    learn_mask: Optional[chex.Array] = None
