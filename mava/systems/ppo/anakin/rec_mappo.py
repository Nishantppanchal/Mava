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

import copy
import time
from typing import Any, Tuple

import chex
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
import flax
from flax.core.frozen_dict import FrozenDict
from jax import tree
from omegaconf import DictConfig, OmegaConf

from mava.evaluator import get_eval_fn, get_num_eval_envs, make_rec_eval_act_fn
from mava.networks import GatedResidualActor, RecurrentActor as Actor
from mava.networks import RecurrentValueNet as Critic
from mava.networks import ScannedRNN
from mava.systems.ppo.types import (
    HiddenStates,
    OptStates,
    Params,
    RNNLearnerState,
    RNNPPOTransition,
)
from mava.types import (
    ExperimentOutput,
    LearnerFn,
    MarlEnv,
    Metrics,
    RecActorApply,
    RecCriticApply,
)
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import replicate, unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.multistep import calculate_gae
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics


def _adaptive_ent_coef(base: float, entropy: chex.Array, target: Any, gain: float) -> chex.Array:
    """Fork §60/§64e: SIGNED target-entropy control for the entropy bonus.

    Fixed coefficients are bistable on wide policies and runaway-prone in
    flat-advantage regimes (§64d: long search phases let even a floored
    bonus drift entropy to ~1.4 and dissolve the policy). Below ``target``
    the bonus is boosted exponentially (the proven §60 half); ABOVE target
    the coefficient goes NEGATIVE — an active entropy penalty proportional
    to the excess — because zeroing the bonus alone has no authority when
    the surrogate itself is flat. Stateless — a pure function of the
    current batch — so checkpoints, warm-starts, and the learner-state
    pytree are unaffected. ``target=None`` reduces to the constant
    ``base``. (§60b's gen3w-ext ran the unsigned v1 semantics: floor 0.05
    instead of the negative branch.)
    """
    if target is None:
        return jnp.asarray(base)
    h = jax.lax.stop_gradient(entropy)
    # Continuous at the target (both branches equal ``base`` there): below,
    # the §60 exponential boost; above, a linear descent through zero into
    # the penalty region, crossing at h = target + 1/gain.
    boost = base * jnp.clip(jnp.exp(gain * (target - h)), 1.0, 20.0)
    penalty = base * jnp.clip(1.0 - gain * (h - target), -20.0, 1.0)
    return jnp.where(h < target, boost, penalty)


def _masked_normalise(x: chex.Array, w: Any = None) -> chex.Array:
    """Fork §131d: standardise ``x`` using only the rows ``w`` selects.

    PPO normalises the advantage at minibatch level, and with ``learn_mask``
    on that scaling was computed over EVERY row and only then were the masked
    rows dropped from the mean. A compromised agent's transitions therefore
    still set the location and scale that every honest agent's gradient was
    divided by — the same leak the mask exists to close, one step upstream of
    where it was closed. A liar acting on a falsified observation is exactly
    the agent whose advantages sit in the tail, so this is not a rounding
    difference: it is an attacker-controlled gain on the honest update.

    ``w=None`` takes the unmasked path and is BIT-IDENTICAL to the previous
    expression (same op order, same 1e-8), so every no-mask run — i.e. every
    run in the campaign record — is unaffected.

    The masked branch uses the weighted mean and the weighted second moment
    about it, guarded so an all-zero mask yields zeros rather than NaNs (a
    minibatch in which no agent is learnable contributes nothing anyway, and
    a NaN there would poison the whole update).

    The masked entries are ZEROED before the reductions rather than merely
    weighted by zero: ``NaN * 0`` is ``NaN``, so a single non-finite advantage
    on a row the mask drops would otherwise flow through the sum into the mean
    and the variance and back out onto every row the mask KEEPS — the masked
    rows leaking into the honest update again, by a different route. ``x``
    itself is still the numerator, so a non-finite value on a KEPT row is left
    visible instead of being silently zeroed.
    """
    if w is None:
        return (x - x.mean()) / (x.std() + 1e-8)
    w = w.astype(x.dtype)
    denom = jnp.sum(w) + 1e-8
    safe_x = jnp.where(w > 0, x, jnp.zeros_like(x))
    mean = jnp.sum(safe_x * w) / denom
    var = jnp.sum(w * jnp.square(safe_x - mean)) / denom
    return jnp.where(w > 0, (x - mean) / (jnp.sqrt(var) + 1e-8), 0.0)


def masked_policy_kl(
    new_logits: chex.Array,
    ref_logits: chex.Array,
    action_mask: chex.Array,
    weights: Any = None,
) -> chex.Array:
    """Fork §138/WP3: ``KL(pi_new || pi_ref)`` averaged over the rows ``weights``
    selects.

    This is the honest-cost BUDGET. §134f bought "an honest team pays exactly
    nothing" with an architecture that could only act after the symbolic filter
    had already convicted someone, which capped what it could ever be worth.
    §138 replaces the guarantee with a measured bound: the residual is free to
    move the policy, and this term prices every bit of movement on an episode
    with no liar in it. ``weights`` is the honest-episode mask, so attacked
    steps contribute nothing — the defence is *supposed* to deviate there.

    ``ref_logits`` come from the frozen champion and are already
    ``stop_gradient``-ed inside ``GatedResidualActor``, so the gradient of this
    term reaches the residual and the gate and nothing else.

    Illegal actions carry ``finfo.min`` in BOTH argument sets. After
    ``log_softmax`` their difference is ``-inf - -inf = NaN``, which would
    poison the whole update, so the per-action term is masked to zero rather
    than relying on the (vanishing) probability weight to do it.
    """
    log_p = jax.nn.log_softmax(new_logits, axis=-1)
    log_q = jax.nn.log_softmax(ref_logits, axis=-1)
    diff = jnp.where(action_mask, log_p - log_q, 0.0)
    p = jnp.where(action_mask, jnp.exp(log_p), 0.0)
    kl = jnp.sum(p * diff, axis=-1)
    if weights is None:
        return kl.mean()
    w = weights.astype(kl.dtype)
    return jnp.sum(jnp.where(w > 0, kl, 0.0) * w) / (jnp.sum(w) + 1e-8)


def _find_sown(d: Any, key: str) -> Any:
    """First ``key`` sown anywhere in an "intermediates" tree, or None.

    Fork §134: under ``GatedResidualActor`` the champion is a SUBMODULE, so
    flax nests its sows one level down and a top-level lookup raises KeyError.
    Find it wherever it is rather than hard-coding either shape.
    """
    if not isinstance(d, dict):
        return None
    if key in d:
        return d[key][0]
    for v in d.values():
        found = _find_sown(v, key)
        if found is not None:
            return found
    return None


def get_learner_fn(
    env: MarlEnv,
    apply_fns: Tuple[RecActorApply, RecCriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[RNNLearnerState]:
    """Get the learner function."""
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(learner_state: RNNLearnerState, _: Any) -> Tuple[RNNLearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
                - last_done (bool): Whether the last timestep was a terminal state.
                - hstates (HiddenStates): The hidden state of the policy and critic RNN.
            _ (Any): The current metrics info.

        """

        def _env_step(
            learner_state: RNNLearnerState, _: Any
        ) -> Tuple[RNNLearnerState, Tuple[RNNPPOTransition, Metrics]]:
            """Step the environment."""
            (
                params,
                opt_states,
                key,
                env_state,
                last_timestep,
                last_done,
                last_hstates,
            ) = learner_state

            key, policy_key = jax.random.split(key)

            # Add a batch dimension to the observation.
            batched_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
            ac_in = (batched_observation, last_done[jnp.newaxis, :])

            # Run the network.
            policy_hidden_state, actor_policy = actor_apply_fn(
                params.actor_params, last_hstates.policy_hidden_state, ac_in
            )
            critic_hidden_state, value = critic_apply_fn(
                params.critic_params, last_hstates.critic_hidden_state, ac_in
            )

            # Sample action from the policy and squeeze out the batch dimension.
            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)

            action, log_prob, value = action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

            # Step the environment.
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            done = timestep.last().repeat(env.num_agents).reshape(config.arch.num_envs, -1)
            # ``timestep.discount`` is the post-step discount from the env: 0 on
            # true termination (Jumanji ``termination()``) and 1 on truncation /
            # mid-episode steps. Carrying it through the rollout lets
            # ``calculate_gae`` apply the CleanRL term-vs-trunc fix — bootstrap
            # V(s_{t+1}) on truncation, zero it on termination. ``timestep.reward``
            # already has shape (num_envs, num_agents); discount matches.
            step_discount = timestep.discount

            # V(s_{t+1}) on the TRUE next observation — the other half of the
            # term-vs-trunc fix. ``AutoResetWrapper`` overwrites
            # ``timestep.observation`` with the reset observation on any episode
            # end, so reading the bootstrap off the next transition's value would
            # feed V(a fresh episode) into the truncation bootstrap that
            # ``step_discount`` just unmasked. ``extras["real_next_obs"]`` is the
            # observation actually reached; evaluate it under
            # ``critic_hidden_state`` (the state AFTER consuming obs_t) with no
            # done flag, since it continues the same episode. On non-terminal
            # steps this reproduces the next transition's value exactly.
            batched_real_next_obs = tree.map(
                lambda x: x[jnp.newaxis, :], timestep.extras["real_next_obs"]
            )
            _, next_val = critic_apply_fn(
                params.critic_params,
                critic_hidden_state,
                (batched_real_next_obs, jnp.zeros_like(last_done)[jnp.newaxis, :]),
            )
            next_val = next_val.squeeze(0)

            hstates = HiddenStates(policy_hidden_state, critic_hidden_state)
            transition = RNNPPOTransition(
                last_done,
                action,
                value,
                timestep.reward,
                log_prob,
                last_timestep.observation,
                last_hstates,
                # Fork §131d: absent unless the env opts in, in which case the
                # field stays None and everything below is an exact no-op.
                last_timestep.extras.get("learn_mask"),
                # Fork §138/WP3: the honest-episode budget's mask. Same source
                # (env.kwargs.learn_mask), same None-is-a-no-op contract.
                last_timestep.extras.get("honest_mask"),
            )
            learner_state = RNNLearnerState(
                params, opt_states, key, env_state, timestep, done, hstates
            )
            metrics = timestep.extras["episode_metrics"] | timestep.extras["env_metrics"]
            return learner_state, (transition, step_discount, next_val, metrics)

        # Step environment for rollout length
        learner_state, (traj_batch, discount_traj, next_val_traj, episode_metrics) = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # Calculate advantage
        params, opt_states, key, env_state, last_timestep, last_done, hstates = learner_state

        # Add a batch dimension to the observation.
        batched_last_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
        ac_in = (batched_last_observation, last_done[jnp.newaxis, :])

        # Run the network.
        _, last_val = critic_apply_fn(params.critic_params, hstates.critic_hidden_state, ac_in)

        # Squeeze out the batch dimension and mask out the value of terminal states.
        # NOTE: unused by the ``next_val_traj`` path of ``calculate_gae`` — which
        # carries the bootstrap for every step, the last one included — but still
        # computed to keep the rollout-boundary hidden state advancing as before.
        last_val = last_val.squeeze(0)

        advantages, targets = calculate_gae(
            traj_batch,
            last_val,
            last_done,
            config.system.gamma,
            config.system.gae_lambda,
            discount_traj=discount_traj,
            next_val_traj=next_val_traj,
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                params, opt_states, key = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: RNNPPOTransition,
                    gae: chex.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # Rerun network
                    obs_and_done = (traj_batch.obs, traj_batch.done)
                    aux_coef = config.system.get("aux_predict_coef", 0.0)
                    # Fork §138/WP3: the honest-episode KL budget needs the
                    # champion's logits, which GatedResidualActor sows, so it
                    # opts into the same mutable collection the §53 aux head
                    # uses. Both off = the stock single-return path, unchanged.
                    kl_coef = config.system.get("honest_kl_coef", 0.0)
                    aux_pred = champion_logits = gate = None
                    if aux_coef or kl_coef:
                        ((_, actor_policy), inters) = actor_apply_fn(
                            actor_params,
                            traj_batch.hstates.policy_hidden_state[0],
                            obs_and_done,
                            mutable=["intermediates"],
                        )
                        sown = inters["intermediates"]
                        aux_pred = _find_sown(sown, "aux_evader_pred")
                        champion_logits = _find_sown(sown, "champion_logits")
                        gate = _find_sown(sown, "gate")
                    else:
                        _, actor_policy = actor_apply_fn(
                            actor_params, traj_batch.hstates.policy_hidden_state[0], obs_and_done
                        )
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    # Calculate actor loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    # Nomalise advantage at minibatch level — over the rows the
                    # mask keeps, not over all of them (fork §131d; see
                    # ``_masked_normalise`` for why the difference matters).
                    lm = traj_batch.learn_mask
                    gae = _masked_normalise(gae, lm)
                    actor_loss1 = ratio * gae
                    actor_loss2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.system.clip_eps,
                            1.0 + config.system.clip_eps,
                        )
                        * gae
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2)
                    ent_per_agent = actor_policy.entropy(seed=key)
                    # Fork §131d: exclude a COMPROMISED agent's transitions from
                    # the policy objective. It runs the shared policy on a
                    # deliberately falsified input, so its rows teach the policy
                    # how to act while believing a phantom — and let it discover
                    # poses that make its own lie easy to detect, a harness
                    # artefact no strategic attacker would reproduce. The critic
                    # is NOT masked: the value function must still learn what a
                    # state with a compromised teammate in it is worth.
                    # (``lm`` was read above, where the advantage is scaled.)
                    if lm is None:
                        actor_loss = actor_loss.mean()
                        entropy = ent_per_agent.mean()
                    else:
                        w = lm.astype(actor_loss.dtype)
                        denom = jnp.sum(w) + 1e-8
                        actor_loss = jnp.sum(actor_loss * w) / denom
                        entropy = jnp.sum(ent_per_agent * w) / denom

                    ent_coef = _adaptive_ent_coef(
                        config.system.ent_coef,
                        entropy,
                        config.system.get("ent_target", None),
                        config.system.get("ent_target_gain", 7.0),
                    )
                    total_loss = actor_loss - ent_coef * entropy
                    aux_mse = jnp.float32(0.0)
                    if aux_coef:
                        # Fork §53: supervised aux loss — predict the TRUE
                        # evader position (global_state[..., 0:2], hindsight)
                        # ``k`` steps ahead. Valid where no episode boundary
                        # sits in (t, t+k]: done-count difference via cumsum.
                        k = int(config.system.get("aux_predict_horizon", 8))
                        gs = traj_batch.obs.global_state[..., 0:2]  # (T,B,N,2)
                        dcum = jnp.cumsum(traj_batch.done.astype(jnp.float32), axis=0)
                        tgt = gs[k:]  # (T-k, B, N, 2)
                        pred = aux_pred[:-k]
                        valid = (dcum[k:] - dcum[:-k]) == 0.0  # (T-k, B, N)
                        se = jnp.sum((pred - tgt) ** 2, axis=-1)
                        aux_mse = jnp.sum(se * valid) / (jnp.sum(valid) + 1e-8)
                        total_loss = total_loss + aux_coef * aux_mse

                    # Fork §138/WP3: the honest-episode KL budget, and the two
                    # gate statistics that say whether the residual is opening
                    # where it is supposed to. All exactly 0.0 unless a
                    # GatedResidualActor sowed and the coefficient is set, so
                    # every existing run's loss_info keys gain three zeros and
                    # nothing else.
                    honest_kl = jnp.float32(0.0)
                    gate_mean_honest = jnp.float32(0.0)
                    gate_mean_attacked = jnp.float32(0.0)
                    hm = traj_batch.honest_mask
                    if gate is not None:
                        g = gate[..., 0]
                        if hm is None:
                            gate_mean_honest = g.mean()
                        else:
                            hw = hm.astype(g.dtype)
                            gate_mean_honest = jnp.sum(g * hw) / (jnp.sum(hw) + 1e-8)
                            aw = 1.0 - hw
                            gate_mean_attacked = jnp.sum(g * aw) / (jnp.sum(aw) + 1e-8)
                    if kl_coef and champion_logits is not None:
                        honest_kl = masked_policy_kl(
                            actor_policy.distribution.logits,
                            champion_logits,
                            traj_batch.obs.action_mask,
                            hm,
                        )
                        total_loss = total_loss + kl_coef * honest_kl
                    return total_loss, (
                        actor_loss,
                        entropy,
                        aux_mse,
                        honest_kl,
                        gate_mean_honest,
                        gate_mean_attacked,
                    )

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: RNNPPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # Rerun network
                    obs_and_done = (traj_batch.obs, traj_batch.done)
                    _, value = critic_apply_fn(
                        critic_params, traj_batch.hstates.critic_hidden_state[0], obs_and_done
                    )

                    # Clipped value loss. ``config.system.critic_loss`` selects
                    # the per-element penalty: ``mse`` (stock, default) or
                    # ``smooth_l1`` (Huber) — the latter is more robust to the
                    # value outliers that sparse +/-10 terminal rewards produce.
                    # Defaults to ``mse`` via ``.get`` so other systems / older
                    # configs are unaffected. (trust_filter PERL port.)
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )
                    if config.system.get("critic_loss", "mse") == "smooth_l1":
                        # optax.huber_loss already includes the 0.5 in its
                        # quadratic region, so do NOT scale by 0.5 again.
                        huber_delta = config.system.get("huber_delta", 1.0)
                        # ``delta`` is keyword-only in optax >=0.2.6; passing it
                        # positionally breaks there. Keyword form works on all.
                        value_losses = optax.huber_loss(value, targets, delta=huber_delta)
                        value_losses_clipped = optax.huber_loss(
                            value_pred_clipped, targets, delta=huber_delta
                        )
                        value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()
                    else:
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    # §117 (D2) instrumentation. Two statistics that decide
                    # whether the Huber/clipping regime is actually costing us,
                    # and that a checkpoint replay CANNOT produce — clipping
                    # compares the freshly-updated value against the stored old
                    # value and GAE target DURING the PPO epochs, while a replay
                    # only ever sees final parameters.
                    #   huber_frac: fraction of residuals in Huber's LINEAR
                    #     region (|V-target| > delta), where gradients are
                    #     bounded and the objective stops being mean-seeking.
                    #     Terminal return RANGE does not establish this — it has
                    #     to be measured on the actual GAE targets.
                    #   clip_frac: fraction of samples whose loss comes from the
                    #     CLIPPED branch, i.e. where a large corrective step is
                    #     being suppressed.
                    resid = jnp.abs(value - targets)
                    huber_frac = (resid > config.system.get("huber_delta", 1.0)).mean()
                    clip_frac = (value_losses_clipped > value_losses).mean()

                    total_loss = config.system.vf_coef * value_loss
                    return total_loss, (value_loss, huber_frac, clip_frac)

                # Calculate actor loss
                key, entropy_key = jax.random.split(key)
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss_info, actor_grads = actor_grad_fn(
                    params.actor_params,
                    traj_batch,
                    advantages,
                    entropy_key,
                )

                # Calculate critic loss
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                value_loss_info, critic_grads = critic_grad_fn(
                    params.critic_params, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="batch"
                )
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
                )

                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="batch"
                )
                # pmean over devices.
                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="device"
                )

                # Update params and optimiser state
                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                new_params = Params(actor_new_params, critic_new_params)
                new_opt_state = OptStates(actor_new_opt_state, critic_new_opt_state)

                actor_loss, (
                    _,
                    entropy,
                    aux_mse,
                    honest_kl,
                    gate_mean_honest,
                    gate_mean_attacked,
                ) = actor_loss_info
                value_loss, (unscaled_value_loss, huber_frac, clip_frac) = (
                    value_loss_info
                )

                total_loss = actor_loss + value_loss
                loss_info = {
                    "total_loss": total_loss,
                    "value_loss": unscaled_value_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                    # Fork §53: 0.0 unless system.aux_predict_coef is set.
                    "aux_predict_loss": aux_mse,
                    # Fork §117 (D2): the value-target regime. huber_frac is the
                    # fraction of residuals in Huber's linear region (bounded
                    # gradients, median-seeking); clip_frac is the fraction
                    # taking the CLIPPED branch (a large correction suppressed).
                    # Both are training-time only — a replay cannot see them.
                    "huber_frac": huber_frac,
                    "clip_frac": clip_frac,
                    # Fork §138/WP3: the honest-cost budget and where the
                    # residual's gate actually opens. `honest_kl` is the
                    # measured price of the defence on liar-free episodes —
                    # the number that replaces §134f's bit-identity claim —
                    # and the two gate means say whether it is opening on the
                    # episodes that warrant it. 0.0 on every run without a
                    # soft-gated residual.
                    "honest_kl": honest_kl,
                    "gate_mean_honest": gate_mean_honest,
                    "gate_mean_attacked": gate_mean_attacked,
                }

                return (new_params, new_opt_state, entropy_key), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key, entropy_key = jax.random.split(key, 3)

            # Shuffle minibatches
            batch = (traj_batch, advantages, targets)
            num_recurrent_chunks = (
                config.system.rollout_length // config.system.recurrent_chunk_size
            )
            batch = tree.map(
                lambda x: x.reshape(
                    config.system.recurrent_chunk_size,
                    config.arch.num_envs * num_recurrent_chunks,
                    *x.shape[2:],
                ),
                batch,
            )
            permutation = jax.random.permutation(
                shuffle_key, config.arch.num_envs * num_recurrent_chunks
            )
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
            reshaped_batch = tree.map(
                lambda x: jnp.reshape(
                    x, (x.shape[0], config.system.num_minibatches, -1, *x.shape[2:])
                ),
                shuffled_batch,
            )
            minibatches = tree.map(lambda x: jnp.swapaxes(x, 1, 0), reshaped_batch)

            # Update minibatches
            (params, opt_states, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, entropy_key), minibatches
            )

            update_state = (
                params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                key,
            )
            return update_state, loss_info

        update_state = (
            params,
            opt_states,
            traj_batch,
            advantages,
            targets,
            key,
        )

        # Update epochs
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = RNNLearnerState(
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            last_done,
            hstates,
        )
        return learner_state, (episode_metrics, loss_info)

    def learner_fn(learner_state: RNNLearnerState) -> ExperimentOutput[RNNLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer states.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
                - dones (bool): Whether the initial timestep was a terminal state.
                - hstates (HiddenStates): The hidden state of the policy and critic RNN.

        """
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        # Number of updates per ``learn`` call = the train-metric logging window.
        # Defaults to the full eval window (one train-log per eval); set
        # ``system.num_updates_per_log`` to log training metrics more often than
        # eval (run_experiment splits each eval window into sub-windows).
        updates_per_call = config.system.get(
            "num_updates_per_log", config.system.num_updates_per_eval
        )
        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, updates_per_call
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[RNNLearnerState], Actor, RNNLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of agents.
    num_agents = env.num_agents
    config.system.num_agents = num_agents

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimiser.
    actor_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_post_torso = hydra.utils.instantiate(config.network.actor_network.post_torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
    critic_pre_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_post_torso = hydra.utils.instantiate(config.network.critic_network.post_torso)

    residual_cfg = config.network.get("residual", None)
    residual_on = residual_cfg is not None and residual_cfg.get("enabled", False)
    # Fork §138/WP3: the residual conditions on the champion's post-GRU core,
    # so the champion has to hand it back. Params are identical either way.
    residual_full_view = residual_on and bool(residual_cfg.get("full_view", False))

    actor_network = Actor(
        pre_torso=actor_pre_torso,
        post_torso=actor_post_torso,
        action_head=actor_action_head,
        hidden_state_dim=config.network.hidden_state_dim,
        # Fork §51: optional temporal-core swap (default "gru" = stock).
        temporal_core=config.network.get("temporal_core", "gru"),
        # Fork §53: aux evader-prediction head, on iff the loss uses it.
        aux_predict=bool(config.system.get("aux_predict_coef", 0.0)),
        return_core=residual_full_view,
    )
    # Fork §134 Layer 3: wrap the champion in a gated residual. The residual
    # is the only trainable part and it is multiplied by a hard 0/1 alarm the
    # environment sets, so on a team with no liars it contributes exactly zero
    # and the fine-tuned policy is bit-identical to the checkpoint it started
    # from. See GatedResidualActor.
    if residual_on:
        tail_start, tail_width = env.unwrapped.actor_tail_slice
        if tail_width <= 0:
            raise ValueError(
                "network.residual.enabled needs the env's trust tail: set "
                "env.kwargs.trust_tail=True (with trust_obs and belief_obs)"
            )
        config.system.tail_start = int(tail_start)
        config.system.tail_width = int(tail_width)
        actor_network = GatedResidualActor(
            champion=actor_network,
            residual_torso=hydra.utils.instantiate(residual_cfg.torso),
            action_dim=env.action_dim,
            tail_start=int(tail_start),
            tail_width=int(tail_width),
            hidden_state_dim=config.network.hidden_state_dim,
            freeze_champion=bool(residual_cfg.get("freeze_champion", True)),
            # Fork §138/WP3. "hard" is §134f verbatim and stays the default.
            gate=str(residual_cfg.get("gate", "hard")),
            gate_bias0=float(residual_cfg.get("gate_bias0", -4.0)),
            delta_clip=float(residual_cfg.get("delta_clip", 5.0)),
            full_view=residual_full_view,
        )

    critic_network = Critic(
        pre_torso=critic_pre_torso,
        post_torso=critic_post_torso,
        hidden_state_dim=config.network.hidden_state_dim,
        centralised_critic=True,
    )

    actor_lr = make_learning_rate(config.system.actor_lr, config)
    critic_lr = make_learning_rate(config.system.critic_lr, config)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    if residual_on and residual_cfg.get("freeze_champion", True):
        # Belt and braces. stop_gradient already zeroes the champion's
        # gradients, but a zero gradient still moves an Adam state, and a
        # future edit that removed the stop_gradient would silently start
        # training the champion. Partitioning says the intent in one place the
        # test can assert on.
        def _label(path, _leaf):
            return "freeze" if any("champion" in str(k) for k in path) else "train"

        # Zero the champion's gradients FIRST, then clip. The other order
        # would let the frozen half contribute to the global norm — and it
        # does contribute: the §53 aux head lives inside the champion and
        # produces a real gradient — which would silently shrink the
        # residual's effective step by a factor nobody chose.
        actor_optim = optax.chain(
            optax.multi_transform(
                {"train": optax.identity(), "freeze": optax.set_to_zero()},
                lambda params: flax.traverse_util.path_aware_map(_label, params),
            ),
            optax.clip_by_global_norm(config.system.max_grad_norm),
            optax.adam(actor_lr, eps=1e-5),
        )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise observation with obs of all agents.
    init_obs = env.observation_spec.generate_value()
    init_obs = tree.map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
        init_obs,
    )
    init_obs = tree.map(lambda x: x[jnp.newaxis, ...], init_obs)
    init_done = jnp.zeros((1, config.arch.num_envs, num_agents), dtype=bool)
    init_obs_done = (init_obs, init_done)

    # Initialise hidden state.
    init_policy_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )
    init_critic_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )

    # initialise params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_policy_hstate, init_obs_done)
    actor_opt_state = actor_optim.init(actor_params)
    critic_params = critic_network.init(critic_net_key, init_critic_hstate, init_obs_done)
    critic_opt_state = critic_optim.init(critic_params)

    # Fork §138/WP3: the REALISED parameter counts, logged once. The
    # "matched by parameter count" controls (ARM_*_BLACKBOX) are only matched
    # if someone checks, and until now nothing printed the number they are
    # supposed to match — the count went into result rows by hand, from the
    # config rather than from the built tree. Stashed on the config as well as
    # printed, so anything that serialises the config carries it.
    n_actor = int(sum(x.size for x in tree.leaves(actor_params)))
    n_critic = int(sum(x.size for x in tree.leaves(critic_params)))
    config.system.actor_param_count = n_actor
    config.system.critic_param_count = n_critic
    print(
        f"{Fore.CYAN}{Style.BRIGHT}Params: actor={n_actor:,} critic={n_critic:,}"
        f"{Style.RESET_ALL}"
    )

    # Get network apply functions and optimiser updates.
    apply_fns = (actor_network.apply, critic_network.apply)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    # Pack params and initial states.
    params = Params(actor_params, critic_params)
    hstates = HiddenStates(init_policy_hstate, init_critic_hstate)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        load_args = dict(config.logger.checkpointing.load_args)
        # Optional load_args.timestep: restore a SPECIFIC saved step (e.g. the
        # best-eval checkpoint) instead of the latest kept one. Popped here —
        # it is a restore_params() argument, not a Checkpointer kwarg.
        load_timestep = load_args.pop("timestep", None)
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, restored_hstates = loaded_checkpoint.restore_params(
            input_params=params,
            timestep=load_timestep,
            restore_hstates=True,
            THiddenState=HiddenStates,
        )
        # Update the params and hstates. Restored hidden states carry the
        # SOURCE run's batch dimensions (num_envs etc.); if the current run's
        # differ (e.g. warm-starting a 128-env checkpoint at 256 envs), keep
        # the freshly initialised hstates instead — they are transient
        # per-episode context, not learned state.
        # Fork §134: a champion checkpoint has no "champion" subtree, so graft
        # it into one. This is what lets the residual arm warm-start from the
        # existing T2 ladder instead of needing a fresh one.
        if residual_on and "champion" not in restored_params.actor_params["params"]:
            grafted = flax.core.unfreeze(actor_params)
            grafted["params"]["champion"] = flax.core.unfreeze(
                restored_params.actor_params
            )["params"]
            restored_params = restored_params._replace(actor_params=grafted)
        params = restored_params
        if restored_hstates is not None:
            fresh_shapes = jax.tree_util.tree_map(lambda x: x.shape, hstates)
            restored_shapes = jax.tree_util.tree_map(lambda x: x.shape, restored_hstates)
            if fresh_shapes == restored_shapes:
                hstates = restored_hstates

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    # Define params to be replicated across devices and batches.
    dones = jnp.zeros(
        (config.arch.num_envs, num_agents),
        dtype=bool,
    )
    key, step_keys = jax.random.split(key)
    opt_states = OptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states, hstates, step_keys, dones)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = replicate(replicate_learner, jax.devices())

    # Initialise learner state.
    params, opt_states, hstates, step_keys, dones = replicate_learner
    init_learner_state = RNNLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
        dones=dones,
        hstates=hstates,
    )
    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig, eval_callback=None) -> float:
    """Runs experiment.

    ``eval_callback`` (optional): called once per evaluation as
    ``eval_callback(actor_network, actor_params, env_step)`` with the same params
    the evaluator scored. Opaque to Mava — used by the pursuit entrypoint to log
    a greedy rollout GIF each eval. None (default) keeps the stock behaviour.
    """
    _config.logger.system_name = "rec_mappo"
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    # Set recurrent chunk size.
    if config.system.recurrent_chunk_size is None:
        config.system.recurrent_chunk_size = config.system.rollout_length
    else:
        assert config.system.rollout_length % config.system.recurrent_chunk_size == 0, (
            "Rollout length must be divisible by recurrent chunk size."
        )

        assert config.arch.num_envs % config.system.num_minibatches == 0, (
            "Number of envs must be divisibile by number of minibatches."
        )

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config=config, add_global_state=True)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Setup evaluator.
    # One key per device for evaluation.
    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_rec_eval_act_fn(actor_network.apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert config.system.num_updates > config.arch.num_evaluation, (
        "Number of updates per evaluation must be less than total number of updates."
    )

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    # Optionally log training metrics more often than eval: split each eval
    # window into ``num_logs_per_eval`` logged sub-windows; eval still runs once
    # per window. Defaults to 1 (original behaviour). Set arch.num_logs_per_eval.
    num_logs_per_eval = max(1, int(config.arch.get("num_logs_per_eval", 1)))
    config.system.num_updates_per_log = max(
        1, config.system.num_updates_per_eval // num_logs_per_eval
    )
    steps_per_log = (
        n_devices
        * config.system.num_updates_per_log
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )
    steps_per_rollout = steps_per_log * num_logs_per_eval
    # Logger setup
    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Create an initial hidden state used for resetting memory for evaluation
    eval_batch_size = get_num_eval_envs(config, absolute_metric=False)
    eval_hs = ScannedRNN.initialize_carry(
        (n_devices, eval_batch_size, config.system.num_agents),
        config.network.hidden_state_dim,
    )
    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    log_count = 0
    for eval_step in range(config.arch.num_evaluation):
        # Train in ``num_logs_per_eval`` sub-windows, logging the training
        # (ACTOR + TRAINER) metrics after each — more often than eval.
        for _ in range(num_logs_per_eval):
            start_time = time.time()
            learner_output = learn(learner_state)
            jax.block_until_ready(learner_output)
            elapsed_time = time.time() - start_time

            log_count += 1
            t = int(steps_per_log * log_count)
            episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
            episode_metrics["steps_per_second"] = steps_per_log / elapsed_time

            logger.log({"timestep": t}, t, log_count - 1, LogEvent.MISC)
            if ep_completed:
                logger.log(episode_metrics, t, log_count - 1, LogEvent.ACT)
            logger.log(learner_output.train_metrics, t, log_count - 1, LogEvent.TRAIN)

            learner_state = learner_output.learner_state

        # Evaluate once per eval window.
        trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)
        eval_metrics = evaluator(trained_params, eval_keys, {"hidden_state": eval_hs})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        # Optional render hook: log a greedy rollout GIF for this eval's policy.
        # ``learner_state.params`` here are the same params the evaluator scored,
        # so the GIF matches the logged numbers. No-op unless a callback was given.
        if eval_callback is not None:
            eval_callback(actor_network, learner_state.params.actor_params, t)

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=t,
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric.
    if config.arch.absolute_metric:
        eval_batch_size = get_num_eval_envs(config, absolute_metric=True)
        eval_hs = ScannedRNN.initialize_carry(
            (n_devices, eval_batch_size, config.system.num_agents),
            config.network.hidden_state_dim,
        )
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)

        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {"hidden_state": eval_hs})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="rec_mappo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Recurrent MAPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
