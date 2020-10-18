import gym
import numpy as np
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
import ray.rllib.models.tf.tf_action_dist as rllib_tf_dist
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy import Policy, TFPolicy
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils.types import AgentID, TensorType, TrainerConfigDict
import tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple, Union

from muzero.mcts import MCTS


class MuZeroTFModelV2(TFModelV2):
    BOARD = 0
    ATARI = 1
    
    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 model_config: Dict[str, Any]):
        super(MuZeroTFModelV2, self).__init__(
            obs_space,
            action_space,
            action_space.n,
            model_config,
            'MuZeroTFModelV2')
        self.config = model_config
        
        self.input_shape = obs_space.shape
        obs_input = tf.keras.Input(self.input_shape)
        state_output = self._build_model(obs_input, model_config['conv_filters']['representation'])
        self.representation_net = tf.keras.Model(obs_input, [state_output])

        self.state_shape = state_output.shape[1:]
        self.action_shape = self.state_shape[:-1] + (action_space.n,)
        self.action_space_size = action_space.n
        if model_config['action_type'] == 'board':
            self.action_type = self.BOARD
        elif model_config['action_type'] == 'atari':
            self.action_type = self.ATARI
        else:
            raise NotImplemented(f'action_type "{model_config["action_type"]}" not implemented in MuZeroTFModelV2')
            
        self.is_reward_categorical = model_config['reward_type'] == 'categorical'
        self.is_value_categorical = model_config['value_type'] == 'categorical'
        
        state_input = tf.keras.Input(self.state_shape)
        action_input = tf.keras.Input(self.action_shape)
        dynamics_input = tf.keras.layers.Concatenate(axis=-1)([state_input, action_input])
        next_state_output = self._build_model(dynamics_input, model_config['conv_filters']['dynamics'])
        reward_output = self._scalar_head(next_state_output, model_config, 'reward')
        self.dynamics_net = tf.keras.Model([state_input, action_input], [next_state_output, reward_output])
        
        prediction_input = tf.keras.Input(self.state_shape)
        prediction_output = self._build_model(prediction_input, model_config['conv_filters']['prediction'])
        value_output = self._scalar_head(prediction_output, model_config, 'value')
        policy_output = self._policy_head(prediction_output, model_config)
        self.prediction_net = tf.keras.Model(prediction_input, [value_output, policy_output])
        
        if model_config['value_type'] == 'categorical':
            self.value_basis = tf.transpose(tf.reshape(tf.range(-model_config['value_max'], model_config['value_max'] + 1, 1.0), (1, -1)))
            self.value_basis_np = np.transpose(np.reshape(np.arange(-model_config['value_max'], model_config['value_max'] + 1, 1.0), (1, -1)))
        if model_config['reward_type'] == 'categorical':
            self.reward_basis = tf.transpose(tf.reshape(tf.range(-model_config['reward_max'], model_config['reward_max'] + 1, 1.0), (1, -1)))
            self.reward_basis_np = np.transpose(np.reshape(np.arange(-model_config['reward_max'], model_config['reward_max'] + 1, 1.0), (1, -1)))
    
        self.input_steps = model_config['input_steps']
        self.K = model_config['loss_steps']
        self.train_n_channels = self.input_steps + self.K
        self.l2_reg = model_config['l2_reg']
        self.value_max = model_config['value_max']
        self.reward_max = model_config['reward_max']

        self._value_out = None
        
        self.register_variables(
            self.representation_net.variables
            + self.dynamics_net.variables
            + self.prediction_net.variables)
        
        self.mcts = MCTS(self, model_config)
    
    def trainable_variables(self) -> List[tf.Variable]:
        return (
            self.representation_net.trainable_variables
            + self.dynamics_net.trainable_variables
            + self.prediction_net.trainable_variables
        )

    def _policy_head(self, last_layer: tf.keras.layers.Layer, config: Dict[str, Any]) -> tf.keras.layers.Layer:
        output = last_layer
        output = tf.keras.layers.Conv2D(256, (3, 3), (1, 1))(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.ReLU()(output)
        
        num_nodes = np.prod(tuple(output.shape[1:]))
        output = tf.keras.layers.Reshape((num_nodes,))(output)
        output = tf.keras.layers.Dense(self.num_outputs)(output)
        output = tf.keras.layers.Softmax()(output)
        return output
    
    def _scalar_head(self, last_layer: tf.keras.layers.Layer, config: Dict[str, Any], name: str = 'value') -> tf.keras.layers.Layer:
        output = last_layer
        output = tf.keras.layers.Conv2D(1, (1, 1), (1, 1))(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.ReLU()(output)
        
        num_nodes = np.prod(tuple(output.shape[1:]))
        output = tf.keras.layers.Reshape((num_nodes,))(output)
        output = tf.keras.layers.Dense(256, activation='relu')(output)
        if config[f'{name}_type'] == 'categorical':
            output = tf.keras.layers.Dense(2 * config[f'{name}_max'] + 1)(output)
        elif config[f'{name}_type'] == 'scalar' and not infinite_max:
            output = tf.keras.layers.Dense(1)(output)
        else:
            raise NotImplemented(f'{name}_type "{config[name + "_type"]}" unknown')
        return output
        
    def _build_model(self, input_layer: tf.keras.layers.Layer, config: Dict[str, Any]) -> tf.keras.layers.Layer:
        last_layer = input_layer
        for layer_count, layer_type, num_filters, kernel_size, stride in config:
            if layer_type == 'conv':
                for _ in range(layer_count):
                    last_layer = tf.keras.layers.Conv2D(num_filters, kernel_size, stride, padding='same')(last_layer)
                    last_layer = tf.keras.layers.BatchNormalization()(last_layer)
                    last_layer = tf.keras.layers.ReLU()(last_layer)
            elif layer_type == 'res':
                for _ in range(layer_count):
                    init_layer = last_layer
                    last_layer = tf.keras.layers.Conv2D(num_filters, kernel_size, stride, padding='same')(last_layer)
                    last_layer = tf.keras.layers.BatchNormalization()(last_layer)
                    last_layer = tf.keras.layers.ReLU()(last_layer) 
                    last_layer = tf.keras.layers.Conv2D(num_filters, kernel_size, stride, padding='same')(last_layer)
                    last_layer = tf.keras.layers.BatchNormalization()(last_layer)
                    last_layer = tf.keras.layers.Add()([last_layer, init_layer])
                    last_layer = tf.keras.layers.ReLU()(last_layer)
            elif layer_type == 'avg_pool':
                for _ in range(layer_count):
                    last_layer = tf.keras.layers.AveragePooling2D(kernel_size, stride, padding='same')(last_layer)
            elif layer_type == 'max_pool':
                for _ in range(layer_count):
                    last_layer = tf.keras.layers.MaxPool2D(kernel_size, stride, padding='same')(last_layer)
        return last_layer
    
    
    def _encode_atari_actions(self, actions: TensorType) -> TensorType:
        """Create one frame per action and encode using one hot"""
        channel_shape = self.state_shape[:-1].as_list()
        actions = tf.reshape(actions, tf.concat([tf.shape(actions), tf.convert_to_tensor([1] * len(channel_shape))], 0))
        tile = tf.tile(actions, tf.constant([1] + channel_shape))
        one_hot = tf.one_hot(tile, self.num_outputs)
        return one_hot

    def value_function(self) -> TensorType:
        return self._value_out

    @staticmethod
    def expectation(categorical: TensorType, basis: TensorType) -> TensorType:
        return tf.tensordot(categorical, basis, axes=[[1], [0]])
    
    @staticmethod
    def expectation_np(categorical: np.ndarray, basis: np.ndarray) -> np.ndarray:
        return np.tensordot(categorical, basis, axes=[[1], [0]])

    @staticmethod
    def scalar_to_categorical(t: TensorType, bound: int) -> TensorType:
        """
        a + b == 1
        a * glb(t) + b * lub(t) == t

        b == 1 - a
        a * glb(t) + (1 - a) * lub(t) == t
        a * (glb(t) - lub(t)) == t - lub(t)
        a == (t - lub(t)) / (glb(t) - lub(t))
        a == lub(t) - t
        b == t - glb(t)
        """
        #print('t shape:', t.shape)
        shape = tf.concat([tf.shape(t), tf.constant([2 * bound + 1])], 0)
        t_clipped = tf.clip_by_value(t, -bound, bound)
        # Negative numbers round toward zero (up). Make them non-negative to fix.
        indices_l = tf.clip_by_value(
            tf.cast(t_clipped + tf.cast(tf.identity(bound), dtype=t.dtype), tf.int32),
            -bound,
            bound) - tf.identity(bound)
        indices_u = tf.clip_by_value(indices_l + tf.identity(1), -bound, bound)

        # TODO: precompute tile and repeat
        dtype = t_clipped.dtype
        left = tf.reshape(tf.cast(indices_u, dtype) - t_clipped, (-1,))
        right = tf.reshape(t_clipped - tf.cast(indices_l, dtype), (-1,))

        def zip_with_indices(u, x, y):
            return tf.transpose(tf.stack([
                tf.tile(tf.range(x), tf.reshape(y, (1,))),
                tf.repeat(tf.range(y), tf.reshape(x, (1,))),
                tf.reshape(u, (-1,))
            ]))
        
        indices_l = zip_with_indices(indices_l + bound, tf.shape(t)[0], tf.shape(t)[1])
        indices_u = zip_with_indices(indices_u + bound, tf.shape(t)[0], tf.shape(t)[1])
        return tf.scatter_nd(indices_l, left, shape) + tf.scatter_nd(indices_u, right, shape)
    
    def forward(self, input_dict: Dict[str, Any], state: List[Any], seq_lens: Any) -> Tuple[TensorType, List[Any]]:
        """
        WARNING: This outputs policy as probabilities if training and as logits if not training.
        
        Arguments:
            input_dict (dict): dictionary of input tensors, including "obs",
                "prev_action", "prev_reward", "is_training"
            state (list): list of state tensors with sizes matching those
                returned by get_initial_state + the batch dimension
            seq_lens (Tensor): 1d tensor holding input sequence lengths
        Returns:
            (outputs, state): The model output tensor of size
                [BATCH, output_spec.size] or a list of tensors corresponding to
                output_spec.shape_list, and a list of state tensors of
                [BATCH, state_size_i].
        """
        # Observations need to have self.input_steps steps for each batch.
        is_training = input_dict['is_training'] if 'is_training' in input_dict else False
        # Convert boolean tensor to Python bool
        if isinstance(is_training, tf.Tensor):
            is_training = tf.keras.backend.eval(is_training)
        
        if is_training:
            value, policy, actions = self.mcts.compute_action(input_dict[SampleBatch.CUR_OBS])
        else:
            # For Atari, obs should be of size (batch_size, screen_x, screen_y, self.input_steps*4).
            hidden_state = self.representation(input_dict[SampleBatch.CUR_OBS])
            value, policy = self.prediction(hidden_state)
        return policy, state

    def forward_with_value(self, obs: TensorType, is_training: bool = False) -> Tuple[TensorType, TensorType]:
        """
        WARNING: This outputs policy as probabilities if training and as logits if not training.
        """
        # Convert boolean tensor to Python bool
        if isinstance(is_training, tf.Tensor):
            is_training = tf.keras.backend.eval(is_training)

        if is_training:
            value, policy, actions = self.mcts.compute_action(obs)
        else:
            # For Atari, obs should be of size (batch_size, screen_x, screen_y, self.input_steps*4).
            hidden_state = self.representation(obs)
            value, policy = self.prediction(hidden_state)
        return value, policy
        
    def representation(self, obs_batch: TensorType) -> TensorType:
        """obs should be of shape (batch_size, 32*4, screen_x, screen_y)"""
        return self.representation_net(obs_batch)
    
    def prediction(self, hidden_state: TensorType) -> Tuple[TensorType, TensorType]:
        """hidden_state should be of shape (batch_size,) + self.state_shape"""
        self._value_out, policy = self.prediction_net(hidden_state)
        return self._value_out, policy
    
    def dynamics(self, hidden_state: TensorType, action_batch: TensorType) -> Tuple[TensorType, TensorType]:
        """action should be of shape (batch_size) + self.state_shape[:-1]"""
        action_t = self._encode_atari_actions(action_batch)
        return self.dynamics_net((hidden_state, action_t))
