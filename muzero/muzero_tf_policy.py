from __future__ import annotations

from collections import OrderedDict
import logging
from typing import Any, Dict

import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

from muzero.mcts import MCTS
from muzero.muzero_tf_model import MuZeroTFModelV2
from muzero.policy import Policy, TFPolicy, traced_eager_policy, LEARNER_STATS_KEY, VALUE_TARGETS
from muzero.replay_buffer import PRIO_WEIGHTS
from muzero.sample_batch import SampleBatch


TensorType = Any
TrainerConfigDict = Dict

tf.compat.v1.enable_eager_execution()


def scale_gradient(t: TensorType, scale: float) -> TensorType:
    """Retain the value of t while reducing its gradient"""
    return scale * t + (1 - scale) * tf.stop_gradient(t)


def make_mu_zero_model(policy: Policy,
                       obs_space: gym.spaces.Space,
                       action_space: gym.spaces.Space,
                       config: TrainerConfigDict) -> MuZeroTFModelV2:
    return MuZeroTFModelV2(obs_space, action_space, config)


class MuZeroLoss:

    cross_entropy = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def __init__(
            self,
            policy,
            model,
            dist_class,
            train_batch):
        batch = self.prepare_batch(policy, model, dist_class, train_batch)
        self.loss(model, dist_class, *batch)

    def loss(self,
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
            policy_preds,
            prio_weights):
        self.mean_entropy = tf.identity(0.)
        # loop through time steps
        for policy_pred in policy_preds:
            action_dist = dist_class(probs=tf.identity(policy_pred))
            self.mean_entropy += tf.reduce_mean(action_dist.entropy())
        # once done with looping, convert it to a tensor
        policy_preds = tf.transpose(tf.convert_to_tensor(policy_preds), perm=(1, 0, 2))

        # Transform reward targets
        if model.config['transform_outputs']:
            reward_targets = model.transform(reward_targets)

        K = int(reward_targets.shape[1])
        w = [1.]
        w.extend([1. / K] * (K - 1))
        w = tf.identity([w])

        prio_weights = tf.identity(tf.convert_to_tensor(prio_weights, dtype=tf.float32))
        self.total_reward_loss = tf.math.reduce_mean(
            scale_gradient(
                reward_loss_fn(tf.identity(reward_targets), reward_preds, model), w
            ),
            axis=-1
        )
        self.total_vf_loss = tf.math.reduce_mean(
            scale_gradient(
                model.config['value_loss_weight'] * value_loss_fn(tf.identity(value_targets), value_preds, model), w
            ),
            axis=-1
        )
        self.total_policy_loss = tf.math.reduce_mean(
            scale_gradient(
                policy_loss_fn(tf.identity(policy_targets), policy_preds), w
            ),
            axis=-1
        )
        self.total_loss = self.total_reward_loss + self.total_vf_loss + self.total_policy_loss

        for t in model.trainable_variables():
            try:
                self.regularization += model.l2_reg * tf.nn.l2_loss(t)
            except AttributeError:
                self.regularization = model.l2_reg * tf.nn.l2_loss(t)

        self.reward_loss = tf.math.reduce_mean(self.total_reward_loss)
        self.vf_loss = tf.math.reduce_mean(self.total_vf_loss)
        self.policy_loss = tf.math.reduce_mean(self.total_policy_loss)
        self.weighted_reward_loss = tf.math.reduce_mean(self.total_reward_loss * prio_weights)
        self.total_weighted_vf_loss = self.total_vf_loss * prio_weights
        self.weighted_vf_loss = tf.math.reduce_mean(self.total_weighted_vf_loss)
        self.weighted_policy_loss = tf.math.reduce_mean(self.total_policy_loss * prio_weights)
        # self.loss = self.reward_loss + self.vf_loss + self.policy_loss + self.regularization
        self.loss = self.weighted_reward_loss + self.weighted_vf_loss + self.weighted_policy_loss + self.regularization

        return self.loss

    # TODO: Check if I need to make things tf functions.

    def prepare_batch(
            self,
            policy: MuZeroTFPolicy,
            model: MuZeroTFModelV2,
            dist_class: type,  # e.g., ray.rllib.models.tf.tf_action_dist.Categorical
            train_batch: Dict[str, TensorType]) -> TensorType:
        obs = train_batch[SampleBatch.CUR_OBS]
        actions = train_batch[SampleBatch.ACTIONS]
        
        reward_preds = []
        value_preds = []
        policy_preds = []
        hidden_state = model.representation(obs)
        for i in range(policy.loss_steps):
            value, action_probs = model.prediction(hidden_state)
            value_preds.append(value)
            policy_preds.append(action_probs)
            hidden_state, reward = model.dynamics(hidden_state, actions[:, i])
            hidden_state = scale_gradient(hidden_state, 0.5)
            reward_preds.append(reward)

        reward_preds = tf.transpose(tf.convert_to_tensor(reward_preds), perm=(1, 0, 2))
        value_preds = tf.transpose(tf.convert_to_tensor(value_preds), perm=(1, 0, 2))

        if model.action_type == MuZeroTFModelV2.ATARI:
            value_loss_fn = self.atari_value_loss
            reward_loss_fn = self.atari_reward_loss
        elif model.action_type == MuZeroTFModelV2.BOARD:
            value_loss_fn = self.board_value_loss
            reward_loss_fn = self.board_reward_loss
        else:
            raise NotImplemented(f'action type "{model.action_type}" unknown')
        
        # Save the loss statistics in an object belonging to the policy.
        # The stats function will use it to return training statistics.
        return (
            reward_loss_fn,
            value_loss_fn,
            self.policy_loss_fn,
            train_batch['rollout_rewards'],
            train_batch['rollout_values'],
            train_batch['rollout_policies'],
            reward_preds,
            value_preds,
            policy_preds,
            train_batch[PRIO_WEIGHTS]
        )

    @classmethod
    def board_value_loss(cls, target: TensorType, output: TensorType, model: MuZeroTFModelV2) -> TensorType:
        return tf.math.reduce_sum(tf.math.squared_distance(target, output), axis=-1)

    @classmethod
    def board_reward_loss(cls, target: TensorType, output: TensorType, model: MuZeroTFModelV2) -> TensorType:
        return tf.math.reduce_sum(tf.math.squared_distance(target, output), axis=-1)

    @classmethod
    def atari_scalar_loss(cls, target: TensorType, output: TensorType, bound: int) -> TensorType:
        target = MuZeroTFModelV2.scalar_to_categorical(target, bound)
        return cls.cross_entropy(target, output)

    @classmethod
    def atari_reward_loss(cls, target: TensorType, output: TensorType, model: MuZeroTFModelV2) -> TensorType:
        return cls.atari_scalar_loss(target, output, model.reward_max)

    @classmethod
    def atari_value_loss(cls, target: TensorType, output: TensorType, model: MuZeroTFModelV2) -> TensorType:
        return cls.atari_scalar_loss(target, output, model.value_max)

    @classmethod
    def policy_loss_fn(cls, target: TensorType, output: TensorType) -> TensorType:
        return cls.cross_entropy(target, output)


class MuZeroTFPolicy(TFPolicy):
    """
    Some parts based on https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/policy/eager_tf_policy.py
    """
    def __init__(self,
                 obs_space,
                 action_space,
                 config):
        super().__init__(obs_space, action_space, config)

        # The global timestep, broadcast down from time to time from the
        # driver.
        self.global_timestep = 0

        self.dist_class = tfp.distributions.Categorical

        self.model = make_mu_zero_model(self, obs_space, action_space, config)

        # Use a distinct model for MCTS so we don't modify the output of value_function() during searches
        self.mcts = MCTS(self.model, config)

        self._optimizer = self.make_optimizer(config)

        self.n_step = config['n_step']
        self.gamma = config['gamma']
        self.loss_steps = config['loss_steps']
        self.batch_size = config['train_batch_size']
        self.grad_clip = config['grad_clip']
        self.last_extra_fetches = {}

        # TODO: this is a hack to make the stats function happy before there is a loss object
        self.loss_obj = None

    def make_optimizer(self, config):
        nesterov = config['nesterov'] if 'nesterov' in config else False
        return tf.keras.optimizers.SGD(
            config['lr'], momentum=config['momentum'], nesterov=nesterov)

    def compute_actions(self,
                        obs_batch: TensorType,
                        is_training: bool = False,
                        **kwargs):
        actions, action_probs, value = self.get_actions(obs_batch, is_training)

        probs: np.ndarray = tf.gather(action_probs, actions, axis=1, batch_dims=1).numpy()
        batch = {
            SampleBatch.ACTION_PROB: probs,
            SampleBatch.ACTION_LOGP: np.log(probs),
            SampleBatch.VF_PREDS: value,
            'action_dist_probs': action_probs.numpy(),
        }

        if not is_training:
            self.global_timestep += 1

        return actions.numpy(), [], batch

    def compute_action(self, obs, greedy=False, mcts=True):
        value, action_probs = self.model.forward(tf.expand_dims(obs, 0), is_training=not mcts)

        action_probs = tf.convert_to_tensor(action_probs)
        if greedy:
            return tf.math.argmax(tf.squeeze(action_probs)).numpy()
        else:
            action_dist = tfp.distributions.Categorical(probs=action_probs)
            return tf.squeeze(action_dist.sample(1)).numpy()

    def get_actions(self, obs_batch: TensorType, is_training: bool):
        value, action_probs = self.model.forward(obs_batch, is_training=is_training)

        if not is_training and self.model.config['transform_outputs']:
            value = self.model.transform(value)

        action_probs = tf.convert_to_tensor(action_probs)
        action_dist = tfp.distributions.Categorical(probs=action_probs)
        actions = tf.squeeze(action_dist.sample(1), axis=0)

        return actions, action_probs, value

    def value_target(self,
                     rewards,
                     vf_preds):
        if self.model.config['transform_outputs']:
            #rewards = self.model.untransform(rewards)
            vf_preds = self.model.untransform(vf_preds)
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
        k = min(N, self.n_step)
        gamma_k = self.gamma ** k
        t = np.concatenate((rewards, vf_preds[-1:], [0] * (k - 1)))
        u = np.concatenate((vf_preds[k - 1:], [0] * (k - 1)))
        value_target = u * gamma_k
        while k > 0:
            k -= 1
            gamma_k /= self.gamma
            value_target += t[k:N + k] * gamma_k
        value_target = value_target.astype(np.float32)

        if self.model.config['transform_outputs']:
            value_target = self.model.transform(value_target)
        
        return value_target
      
    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch) -> SampleBatch:
        """
        This function cheats at rolling up the experience into "rollouts".
        It assumes that the batch experience is in order without missing
        frames and is from a single episode. It cheats by making the end of
        an episode run together with the beginning of the next episode.
        """
        
        rewards = sample_batch[SampleBatch.REWARDS]
        vf_preds = sample_batch[SampleBatch.VF_PREDS]
        value_target = self.value_target(rewards, vf_preds)
        
        N = len(rewards)
        
        def rollout(values):
            """Matrix of shape (N, loss_steps)"""
            arr = np.array([
                values if i == 0 else np.concatenate((values[i:], [values[-1] for _ in range(min(i, N))]), axis=0)
                for i in range(self.loss_steps)
            ])
            if len(arr.shape) == 3:
                return np.transpose(arr, axes=(1, 0, 2))
            else:
                return np.transpose(arr, axes=(1, 0))

        sample_batch[VALUE_TARGETS] = value_target

        actions = sample_batch[SampleBatch.ACTIONS]
        sample_batch[SampleBatch.ACTIONS] = rollout(actions)

        action_dist_inputs = sample_batch['action_dist_probs']
        sample_batch['rollout_policies'] = rollout(action_dist_inputs)

        sample_batch['rollout_values'] = rollout(value_target)

        if self.model.config['transform_outputs']:
            rewards = self.model.transform(rewards)
        sample_batch['rollout_rewards'] = rollout(rewards)

        return sample_batch

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        grads_and_vars, stats = self.compute_gradients(samples)
        self.apply_gradients(grads_and_vars)
        return stats

    def apply_gradients(self, grads_and_vars):
        self._optimizer.apply_gradients(grads_and_vars)

    def compute_gradients(self, samples):
        self._is_training = True
        with tf.GradientTape(persistent=True) as tape:
            _, _, batch = self.compute_actions(samples[SampleBatch.CUR_OBS], is_training=True)
            for key in batch:
                samples[key] = batch[key]
            # Not sure if this is necessary
            if self.loss_obj:
                del self.loss_obj
            self.loss_obj = MuZeroLoss(self, self.model, self.dist_class, samples)

        variables = self.trainable_variables()

        grads = tape.gradient(self.loss_obj.loss, variables)
        grads = self.clip_gradients(grads)

        assert len(variables) == len(grads)
        grads_and_vars = list(zip(grads, variables))
        stats = self._stats()
        for grad, var in grads_and_vars:
            if 'dense' in var.name and 'kernel' in var.name:
                if '_' not in var.name:
                    i = 0
                else:
                    i = int(var.name.split('/')[0].split('_')[1])
                grad_norm = tf.math.sqrt(tf.math.reduce_sum(grad * grad)).numpy()
                weight_norm = tf.math.sqrt(tf.math.reduce_sum(var.value() * var.value())).numpy()
                stats[LEARNER_STATS_KEY][f'dense_{i}_weights'] = weight_norm
                stats[LEARNER_STATS_KEY][f'dense_{i}_gradient'] = grad_norm

        return grads_and_vars, stats

    def clip_gradients(self, grads):
        b = self.grad_clip
        if b is not None:
            clipped_grads, _ = tf.clip_by_global_norm(grads, b)
        else:
            clipped_grads = grads
        return clipped_grads

    def _stats(self):
        stats = {}
        stats[LEARNER_STATS_KEY] = {
            # TODO: 'cur_lr': tf.cast(policy.cur_lr, tf.float32),
            'entropy': self.loss_obj.mean_entropy.numpy(),
            'total_loss': self.loss_obj.loss.numpy(),
            'reward_loss': self.loss_obj.reward_loss.numpy(),
            'vf_loss': self.loss_obj.vf_loss.numpy(),
            'policy_loss': self.loss_obj.policy_loss.numpy(),
            'weighted_reward_loss': self.loss_obj.weighted_reward_loss.numpy(),
            'weighted_vf_loss': self.loss_obj.weighted_vf_loss.numpy(),
            'weighted_policy_loss': self.loss_obj.weighted_policy_loss.numpy(),
            'regularization': self.loss_obj.regularization.numpy(),
        }
        stats['replay_p'] = self.loss_obj.total_weighted_vf_loss.numpy()
        return stats

    def variables(self):
        return self.model.variables()

    def trainable_variables(self):
        return self.model.trainable_variables()

    def get_weights(self, as_dict=False):
        variables = self.variables()
        if as_dict:
            return {v.name: v.numpy() for v in variables}
        return [v.numpy() for v in variables]

    def set_weights(self, weights):
        variables = self.variables()
        assert len(weights) == len(variables), (len(weights),
                                                len(variables))
        for v, w in zip(variables, weights):
            v.assign(w)

    def set_timestep(self, timestep):
        # TODO: Update learning rate based on timestep
        # Config['lr'] and self._optimizer
        pass

    def get_initial_state(self):
        return self.model.get_initial_state()

    def get_state(self):
        state = {
            "state": OrderedDict(
                (var.name, K.get_value(var))
                for var in self.model.variables()
            ),
            "optimizer_state": OrderedDict(
                (var.name, K.get_value(var))
                for var in self._optimizer.variables()
            ),
        }
        return state

    def set_state(self, state):
        for var in self.model.variables():
            if var.name in state['state']:
                var.assign(state['state'][var.name])
            else:
                logging.warning(f'Loading model saved state: no value for variable {var.name} was found in the saved state')
        for var in self._optimizer.variables():
            if var.name in state['optimizer_state']:
                var.assign(state['optimizer_state'][var.name])
            else:
                logging.warning(f'Loading optimizer saved state: no value for variable {var.name} was found in the saved state')

    @classmethod
    def with_tracing(cls):
        return traced_eager_policy(cls)
