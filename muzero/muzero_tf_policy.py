import gym
import numpy as np
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.modelv2 import ModelV2
import ray.rllib.models.tf.tf_action_dist as rllib_tf_dist
from ray.rllib.policy import Policy
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils.types import AgentID, TensorType, TrainerConfigDict
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Any, Dict, List, Optional, Tuple, Union

from muzero.mcts import MCTS
from muzero.muzero import ATARI_DEFAULT_CONFIG
from muzero.muzero_tf_model import MuZeroTFModelV2
from muzero.replay_buffer import PRIO_WEIGHTS


def make_mu_zero_model(policy: Policy,
                       obs_space: gym.spaces.Space,
                       action_space: gym.spaces.Space,
                       config: TrainerConfigDict) -> ModelV2:
    return MuZeroTFModelV2(obs_space, action_space, config)

def scale_gradient(t: TensorType, scale: float) -> TensorType:
    """Retain the value of t while reducing its gradient"""
    return scale * t + (1 - scale) * tf.stop_gradient(t)

def board_value_loss(target: TensorType, output: TensorType, model: MuZeroTFModelV2) -> TensorType:
    return tf.math.reduce_sum(tf.math.squared_distance(target, output), axis=-1)

board_reward_loss = board_value_loss

def atari_scalar_loss(target: TensorType, output: TensorType, bound: int) -> TensorType:
    target = MuZeroTFModelV2.scalar_to_categorical(target, bound)
    return tf.nn.softmax_cross_entropy_with_logits(target, output, axis=-1)

def atari_reward_loss(target: TensorType, output: TensorType, model: MuZeroTFModelV2) -> TensorType:
    return atari_scalar_loss(target, output, model.reward_max)

def atari_value_loss(target: TensorType, output: TensorType, model: MuZeroTFModelV2) -> TensorType:
    return atari_scalar_loss(target, output, model.value_max)

def policy_loss_fn(target: TensorType, output: TensorType) -> TensorType:
    return tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output)
    
class MuZeroLoss:
    def __init__(self,
                 model,
                 dist_class,
                 reward_loss_fn,
                 value_loss_fn,
                 policy_loss_fn,
                 reward_targets,
                 value_targets,
                 policy_targets,
                 reward_preds,
                 value_preds,
                 policy_preds):
        self.mean_entropy = tf.identity(0.)
        # loop through time steps
        for policy_pred in policy_preds:
            action_dist = dist_class(tf.identity(policy_pred), model)
            self.mean_entropy += tf.reduce_mean(action_dist.entropy())
        # once done with looping, convert it to a tensor
        policy_preds = tf.transpose(tf.convert_to_tensor(policy_preds), perm=(1, 0, 2))
        policy_preds = tf.nn.softmax(policy_preds)

        K = int(reward_targets.shape[1])
        w = [1.]
        w.extend([1. / K] * (K - 1))
        w = tf.identity([w])

        #print('reward targets:', reward_targets.shape)
        #print('reward preds:', reward_preds.shape)
        #print('value targets:', value_targets.shape)
        #print('value preds:', value_preds.shape)
        #print('policy targets:', policy_targets.shape)
        #print('policy preds:', policy_preds.shape)
        self.total_reward_loss = reward_loss_fn(tf.identity(reward_targets), reward_preds, model)
        self.total_vf_loss = model.config['value_loss_weight'] * value_loss_fn(tf.identity(value_targets), value_preds, model)
        self.total_policy_loss = policy_loss_fn(tf.identity(policy_targets), policy_preds)
        #print('total reward:', self.total_reward_loss.shape)
        #print('total vf:', self.total_vf_loss.shape)
        #print('total_policy:', self.total_policy_loss.shape)
        self.total_loss = scale_gradient(self.total_reward_loss + self.total_vf_loss + self.total_policy_loss, w)

        for t in model.trainable_variables():
            self.total_loss += model.l2_reg * tf.nn.l2_loss(t)

        self.mean_reward_loss = tf.math.reduce_mean(self.total_reward_loss)
        self.mean_vf_loss = tf.math.reduce_mean(self.total_vf_loss)
        self.mean_policy_loss = tf.math.reduce_mean(self.total_policy_loss)
        self.sample_loss = tf.math.reduce_mean(self.total_loss, axis=-1)
        self.loss = tf.math.reduce_mean(self.total_loss)

# TODO: Check if I need to make things tf functions.

def mu_zero_loss(policy,
                 model: MuZeroTFModelV2,
                 dist_class: type,  # e.g., ray.rllib.models.tf.tf_action_dist.Categorical
                 train_batch: Dict[str, TensorType]) -> TensorType:
    obs = train_batch[SampleBatch.CUR_OBS]
    actions = train_batch[SampleBatch.ACTIONS]
    #if len(list(actions.shape)) < 2:
    #    actions = tf.convert_to_tensor([actions])
    
    reward_preds = []
    value_preds = []
    policy_preds = []
    hidden_state = model.representation(obs)
    for i in range(policy.loss_steps):
        value, action_probs = model.prediction(hidden_state)
        value_preds.append(value)
        policy_preds.append(action_probs)
        # j is a workaround for rllib passing in malformed batch at initialization.
        #j = min(i, actions.shape[1] - 1)
        hidden_state, reward = model.dynamics(hidden_state, actions[:, i])
        hidden_state = scale_gradient(hidden_state, 0.5)
        reward_preds.append(reward)

    reward_preds = tf.transpose(tf.convert_to_tensor(reward_preds), perm=(1, 0, 2))
    value_preds = tf.transpose(tf.convert_to_tensor(value_preds), perm=(1, 0, 2))

    if model.action_type == MuZeroTFModelV2.ATARI:
        value_loss_fn = atari_value_loss
        reward_loss_fn = atari_reward_loss
    elif model.action_type == MuZeroTFModelV2.BOARD:
        value_loss_fn = board_value_loss
        reward_loss_fn = board_reward_loss
    else:
        raise NotImplemented(f'action type "{model.action_type}" unknown')
    
    # Save the loss statistics in an object belonging to the policy.
    # The stats function will use it to return training statistics.
    policy.loss_obj = MuZeroLoss(
        model,
        dist_class,
        reward_loss_fn,
        value_loss_fn,
        policy_loss_fn,
        train_batch['rollout_rewards'],
        train_batch['rollout_values'],
        train_batch['rollout_policies'],
        reward_preds,
        value_preds,
        policy_preds)
    
    return policy.loss_obj.loss

def mu_zero_stats(policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    return {
        # TODO: 'cur_lr': tf.cast(policy.cur_lr, tf.float32),
        'total_loss': policy.loss_obj.loss,  # This is actually the mean loss
        'reward_loss': policy.loss_obj.mean_reward_loss,
        'vf_loss': policy.loss_obj.mean_vf_loss,
        'policy_loss': policy.loss_obj.mean_policy_loss,
        'entropy': policy.loss_obj.mean_entropy,
    }


def action_distribution_fn(
    policy: Policy,
    model: MuZeroTFModelV2,
    obs_batch: TensorType,
    state_batches=None,
    seq_lens=None,
    prev_action_batch=None,
    prev_reward_batch=None,
    explore=None,
    timestep=None,
    is_training=False):
    """
    Returns
    distribution inputs (parameters),
    dist-class to generate an action distribution from,
    internal state outputs (or empty list if N/A)
    """
    value, action_probs = model.forward_with_value(obs_batch, is_training)
    value = model.untransform(value)
    # Categorical assumes logits.
    action_dist = rllib_tf_dist.Categorical(tf.convert_to_tensor(action_probs), model)
    actions, logp = policy.exploration.get_exploration_action(
        action_distribution=action_dist,
        timestep=timestep,
        explore=False
    )
    t = tf.gather(action_probs, actions, axis=1, batch_dims=1)
    action_probs = tf.nn.softmax(tf.convert_to_tensor(action_probs), axis=-1)
    policy.last_extra_fetches = {
        SampleBatch.ACTION_PROB: tf.gather(action_probs, actions, axis=1, batch_dims=1),
        SampleBatch.VF_PREDS: tf.convert_to_tensor(value)
    }
    return action_probs, rllib_tf_dist.Categorical, []

def mu_zero_postprocess(
        policy,
        sample_batch,
        other_agent_batches = None,
        episode = None):
    """
    Called before compute_actions:
      https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/evaluation/sample_batch_builder.py#L193
      https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/evaluation/sample_batch_builder.py#L252
      https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/evaluation/sampler.py#L779

    policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[AgentID, Tuple[Policy, SampleBatch]]] = None,
    episode: Optional["MultiAgentEpisode"] = None) -> SampleBatch:
    """
    
    rewards = policy.model.untransform(sample_batch[SampleBatch.REWARDS])
    vf_preds = policy.model.untransform(
        policy.model.expectation_np(
            sample_batch[SampleBatch.VF_PREDS],
            policy.model.value_basis_np
        )
    )
    vf_preds = np.reshape(vf_preds, (-1,))
    
    # This calculates
    #   t[0:N] + gamma * t[1:N + 1] + ... + gamma^n_step * u
    # This will be
    #   r_0 + gamma * r_1 + ... + gamma^n_step * v_n_step
    # for the first entry when there are enough steps.
    # N = 7, k = 5
    # r0 + r1 + r2 + r3 + r4 + v4
    # r1 + r2 + r3 + r4 + r5 + v5
    # r2 + r3 + r4 + r5 + r6 + v6
    # r3 + r4 + r5 + r6 + v6
    # r4 + r5 + r6 + v6
    # r5 + r6 + v6
    # r6 + v6
    N = len(rewards)
    k = min(N, policy.n_step)
    gamma_k = policy.gamma ** k
    t = np.concatenate((rewards, vf_preds[-1:], [0] * (k - 1)))
    u = np.concatenate((vf_preds[k - 1:], [0] * (k - 1)))
    value_target = u * gamma_k
    while k > 0:
        k -= 1
        gamma_k /= policy.gamma
        value_target += t[k:N + k] * gamma_k
    value_target = value_target.astype(np.float32)
    sample_batch[Postprocessing.VALUE_TARGETS] = policy.model.transform(value_target)
    
    def rollout(values):
        """Matrix of shape (N, loss_steps) """
        arr = np.array([
            values if i == 0 else np.concatenate((values[i:], [values[-1] for _ in range(min(i, N))]))
            for i in range(policy.loss_steps)
        ])
        if len(arr.shape) == 3:
            return np.transpose(arr, axes=(1, 0, 2))
        else:
            return np.transpose(arr, axes=(1, 0))
    
    action_dist_inputs = sample_batch[SampleBatch.ACTION_DIST_INPUTS]
    sample_batch[SampleBatch.ACTIONS] = rollout(sample_batch[SampleBatch.ACTIONS])
    sample_batch['rollout_values'] = rollout(value_target)
    sample_batch['rollout_rewards'] = rollout(rewards)
    #sample_batch['rollout_policies'] = rollout(action_dist_inputs, default = [0] * len(action_dist_inputs[0]))
    s = np.sum(action_dist_inputs, axis=-1)
    assert np.sum(np.abs(s - np.ones(s.shape))) < 1e-3
    sample_batch['rollout_policies'] = rollout(action_dist_inputs)
    sample_batch['is_training'] = [True] * rewards.shape[0]

    # Setting the weight to -1 makes the weight be set to the max weight
    if PRIO_WEIGHTS not in sample_batch:
        sample_batch[PRIO_WEIGHTS] = -np.ones_like(sample_batch[SampleBatch.REWARDS])
    return sample_batch

def before_init(policy, obs_space, action_space, config):
    policy.n_step = config['n_step']
    policy.gamma = config['gamma']
    policy.batch_size = config['train_batch_size']
    # TODO: this is a hack to make the stats function happy before there is a loss object
    policy.loss_obj = None


def before_loss_init(policy, obs_space, action_space, config):
    ComputeTDErrorMixin.__init__(policy, obs_space, action_space, config)
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    MCTSMixin.__init__(policy, obs_space, action_space, config)

#from muzero.debug import tensor_stats

def clip_gradients(policy, optimizer, loss):
    #print('otpimizer:', type(optimizer))
    grads_and_vars = optimizer.compute_gradients(
        loss, policy.model.trainable_variables())
    #print('grad stats:', [(v.name, tensor_stats(g)) for (g, v) in grads_and_vars])
    b = policy.config['grad_clip']
    if b is not None:
        grads = [g for (g, v) in grads_and_vars]
        grads, _ = tf.clip_by_global_norm(grads, b)
        clipped_grads = list(zip(grads, policy.model.trainable_variables()))
    else:
        clipped_grads = grads_and_vars
    return clipped_grads

def make_optimizer(policy, config):
    nesterov = config['nesterov'] if 'nesterov' in config else False
    return tf.keras.optimizers.SGD(config['lr'], momentum=config['momentum'], nesterov=nesterov)

def vf_preds_fetches(policy):
    """Adds value function outputs to experience sample batches."""
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
    }

def td_error_fetches(policy):
    return {
        'td_error': policy.loss_obj.sample_loss,
    }

class ComputeTDErrorMixin:
    def __init__(self, obs_space, action_space, config):
        @make_tf_callable(self.get_session(), dynamic_shape=True)
        def compute_td_error(obs_t, act_t, rew_t, obs_tp1, done_mask,
                             importance_weights):
            # Do forward pass on loss to update td error attribute
            mu_zero_loss(
                self, self.model, rllib_tf_dist.Categorical, {
                    SampleBatch.CUR_OBS: obs_t,
                    SampleBatch.ACTIONS: act_t,
                    SampleBatch.REWARDS: rew_t,
                    SampleBatch.NEXT_OBS: obs_tp1,
                    SampleBatch.DONES: done_mask,
                    PRIO_WEIGHTS: importance_weights,
                })

            return self.q_loss.total_loss

        self.compute_td_error = compute_td_error

class ValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        pass

    def _value(self, obs, prev_action, prev_reward, *state):
        values, policies, actions = self.mcts.compute_action(obs)
        return values


class MCTSMixin:
    def __init__(self, obs_space, action_space, config):
        # Use a distinct model for MCTS so we don't modify the output of value_function() during searches
        self.mcts = MCTS(self.model, config)
        self.loss_steps = config['loss_steps']
        self.n_step = config['n_step']
        self.gamma = config['gamma']
        self.batch_size = max(1, config['train_batch_size'] // config['replay_sequence_length'])
        self.last_extra_fetches = {}
    
       
MuZeroTFPolicy = build_tf_policy('MuZeroTFPolicy',
                                 get_default_config=lambda: ATARI_DEFAULT_CONFIG,
                                 loss_fn=mu_zero_loss,
                                 stats_fn=mu_zero_stats,
                                 #before_init=before_init,
                                 before_loss_init=before_loss_init,
                                 gradients_fn=clip_gradients,
                                 optimizer_fn=make_optimizer,
                                 postprocess_fn=mu_zero_postprocess,
                                 extra_action_fetches_fn=vf_preds_fetches,
                                 extra_learn_fetches_fn=td_error_fetches,
                                 make_model=make_mu_zero_model,
                                 action_distribution_fn=action_distribution_fn,
                                 mixins=[
                                     ValueNetworkMixin,
                                     MCTSMixin,
                                     ComputeTDErrorMixin
                                 ])
