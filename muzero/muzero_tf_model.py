from __future__ import annotations

from typing import Any

import gym
import numpy as np
import tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple, Union

from muzero.mcts import MCTS
from muzero.sample_batch import SampleBatch

TensorType = Any


class MuZeroTFModelV2:
    BOARD = 0
    ATARI = 1
    
    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 model_config: Dict[str, Any]):
        self.config = model_config

        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import InteractiveSession

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

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
        reward_output = self._scalar_head(next_state_output, model_config['conv_filters']['reward'], model_config['reward_type'], model_config['reward_max'])
        self.dynamics_net = tf.keras.Model([state_input, action_input], [next_state_output, reward_output])
        
        prediction_input = tf.keras.Input(self.state_shape)
        prediction_output = self._build_model(prediction_input, model_config['conv_filters']['prediction'])
        value_output = self._scalar_head(prediction_output, model_config['conv_filters']['value'], model_config['value_type'], model_config['value_max'])
        policy_output = self._policy_head(prediction_output, model_config['conv_filters']['policy'])
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
        # the epsilon used in the formula for scaling the targets
        self.scaling_epsilon = model_config['scaling_epsilon']
        self.value_max = model_config['value_max']
        self.reward_max = model_config['reward_max']

        self.var_list = (
            self.representation_net.variables
            + self.dynamics_net.variables
            + self.prediction_net.variables
        )
        self.train_var_list = (
            self.representation_net.trainable_variables
            + self.dynamics_net.trainable_variables
            + self.prediction_net.trainable_variables
        )
        
        self.mcts = MCTS(self, model_config)

    def variables(self) -> List[tf.Variable]:
        return self.var_list

    def trainable_variables(self) -> List[tf.Variable]:
        return self.train_var_list

    def _policy_head(self, last_layer: tf.keras.layers.Layer, config: Dict[str, Any]) -> tf.keras.layers.Layer:
        output = last_layer
        output = self._build_model(output, config)
       
        num_nodes = np.prod(tuple(output.shape[1:]))
        output = tf.keras.layers.Reshape((num_nodes,))(output)
        output = tf.keras.layers.Dense(self.action_space_size)(output)
        output = tf.keras.layers.Softmax()(output)
        return output
    
    def _scalar_head(self, last_layer: tf.keras.layers.Layer, config: Dict[str, Any], head_type: str = 'scalar', head_max: int = 300) -> tf.keras.layers.Layer:
        output = last_layer
        output = self._build_model(output, config)
       
        if head_type == 'categorical':
            output = tf.keras.layers.Dense(2 * head_max + 1)(output)
            output = tf.keras.layers.Softmax()(output)
        elif head_type == 'scalar':
            output = tf.keras.layers.Dense(1, activation='tanh')(output)
        else:
            raise NotImplemented(f'scalar head type "{head_type}" unknown')
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
            elif layer_type == 'fc':
                if len(last_layer.shape) > 2:
                    last_layer = tf.keras.layers.Reshape((-1,))(last_layer)
                for _ in range(layer_count):
                    last_layer = tf.keras.layers.Dense(num_filters)(last_layer)
                    last_layer = tf.keras.layers.ReLU()(last_layer)
        return last_layer
    
    
    def _encode_atari_actions(self, actions: TensorType) -> TensorType:
        """Create one frame per action and encode using one hot"""
        channel_shape = self.state_shape[:-1].as_list()
        actions = tf.reshape(actions, tf.concat([tf.shape(actions), tf.convert_to_tensor([1] * len(channel_shape))], 0))
        tile = tf.tile(actions, tf.constant([1] + channel_shape))
        one_hot = tf.one_hot(tile, self.action_space_size)
        return one_hot

    def transform(self, t: TensorType) -> TensorType:
        if isinstance(t, np.ndarray):
            return (
                np.sign(t) * (
                    np.sqrt(np.abs(t) + 1) - 1
                ) + self.scaling_epsilon * t
            )
        else:
            return (
                tf.math.sign(t) * (
                    tf.math.sqrt(tf.math.abs(t) + 1) - 1
                ) + self.scaling_epsilon * t
            )

    def untransform(self, t: TensorType) -> TensorType:
        """
        Valid for self.scaling_epsilon < 0.25.

        x >= 0 => y >= 0
        y = s - 1 + eps * x
        x = s^2 - 1
        y = s - 1 + eps * s^2 - eps
        y = eps * s^2 + s - (1 + eps)
        0 = eps * s^2 + s - (1 + eps + y)
        s = (-1 +/- sqrt(1 + 4 * eps * (1 + eps + y))) / (2 * eps)
        s = (-1 + sqrt(1 + 4 * eps * (1 + eps + y)))  / (2 * eps)
        x = (-1 + sqrt(1 + 4 * eps * (y + eps + 1)))^2 / (4 * eps^2) - 1

        x < 0 => y < 0, eps < 1, y > 1 - eps - 1 / (4 * eps)
        #eps < 0.25 => y > 1 - eps - 1 / (4 * eps)
        1 + eps < y
        y = -s + 1 + eps * x
        x = -s^2 + 1
        y = -s + 1 + eps * (-s^2 + 1)
        y = -eps * s^2 - s + (1 + eps)
        0 = -eps * s^2 - s + (1 + eps - y)
        s = (1 +/- sqrt(1 - 4 * (-eps) * (1 + eps - y))) / (2 * (-eps))
        s = (-1 + sqrt(1 + 4 * eps * (1 + eps - y))) / (2 * eps)
        x = -(1 - sqrt(1 - 4 * eps * (1 - eps - y)))^2 / (4 * eps^2) + 1
        x = -((-1 + sqrt(1 + 4 * eps * (y + eps - 1)))^2 / (4 * eps^2) - 1)

        x = sign(y) * (sqrt(1 + 4 * eps * (y + eps + sign(y))) - 1)^2 / (4 * eps^2) - 1)
        """
        eps = self.scaling_epsilon
        if isinstance(t, np.ndarray):
            return np.sign(t) * (
                np.square(np.divide(
                    np.sqrt(1 + 4 * eps * (np.abs(t) + 1 + eps)) - 1,
                    2 * eps
                )) - 1
            )
        else:
            return tf.math.sign(t) * (
                tf.math.square(tf.math.divide(
                    tf.math.sqrt(1 + 4 * eps * (tf.math.abs(t) + 1 + eps)) - 1,
                    2 * eps
                )) - 1
            )

    @staticmethod
    def expectation(categorical: TensorType, basis: TensorType) -> TensorType:
        return tf.tensordot(categorical, basis, axes=[[-1], [0]])
    
    @staticmethod
    def expectation_np(categorical: np.ndarray, basis: np.ndarray) -> np.ndarray:
        return np.tensordot(categorical, basis, axes=[[-1], [0]])

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
        t_clipped = tf.clip_by_value(t, -bound, bound)
        shape = tf.concat([tf.shape(t), tf.constant([2 * bound + 1])], 0)
        dtype = t_clipped.dtype

        # Negative numbers round toward zero (up). Make them non-negative to fix.
        indices_l = tf.cast(t_clipped + tf.cast(tf.identity(bound), dtype=dtype), tf.int32) - tf.identity(bound)
        indices_u = indices_l + 1

        # TODO: precompute tile and repeat
        left = tf.reshape(tf.cast(indices_u, dtype) - t_clipped, (-1,))
        right = tf.reshape(t_clipped - tf.cast(indices_l, dtype), (-1,))

        def zip_with_indices(u, x, y):
            return tf.transpose(tf.stack([
                tf.repeat(tf.range(x), tf.reshape(y, (1,))),
                tf.tile(tf.range(y), tf.reshape(x, (1,))),
                tf.reshape(u, (-1,))
            ]))

        indices_l = zip_with_indices(indices_l + bound, tf.shape(t)[0], tf.shape(t)[1])
        indices_u = zip_with_indices(indices_u + bound, tf.shape(t)[0], tf.shape(t)[1])

        in_bounds = indices_u[:, 2] < 2 * bound + 1
        indices_u = tf.boolean_mask(indices_u, in_bounds)
        right = tf.boolean_mask(right, in_bounds)

        return tf.scatter_nd(indices_l, left, shape) + tf.scatter_nd(indices_u, right, shape)

    def __call__(self, input_dict: Dict[str, Any], is_training: bool = True) -> Tuple[TensorType, List[Any]]:
        return self.forward(input_dict, is_training) 

    def forward(self, input_dict: Dict[str, Any], is_training: bool = True) -> Tuple[TensorType, List[Any]]:
        """
        WARNING: This outputs policy as probabilities.

        This is called by the learner thread.
        
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
        # Convert boolean tensor to Python bool
        if isinstance(is_training, tf.Tensor):
            is_training = tf.keras.backend.eval(is_training)

        if is_training:
            # For Atari, obs should be of size (batch_size, screen_x, screen_y, self.input_steps*4).
            hidden_state = self.representation(input_dict[SampleBatch.CUR_OBS])
            value, policy = self.prediction(hidden_state)
            value = self.expectation(value, self.value_basis)
        else:
            value, policy, actions = self.mcts.compute_action(input_dict[SampleBatch.CUR_OBS])
        return policy, value

    def forward_with_value(self, obs: TensorType, is_training: bool = False) -> Tuple[TensorType, TensorType]:
        """
        WARNING: This outputs policy as probabilities.

        This is called by the rollout workers.
        """
        # Convert boolean tensor to Python bool
        if isinstance(is_training, tf.Tensor):
            is_training = tf.keras.backend.eval(is_training)

        if is_training:
            # For Atari, obs should be of size (batch_size, screen_x, screen_y, self.input_steps*4).
            hidden_state = self.representation(obs)
            value, policy = self.prediction(hidden_state)
            value = self.expectation(value, self.value_basis)
        else:
            value, policy, actions = self.mcts.compute_action(obs)
        return value, policy
        
    def representation(self, obs_batch: TensorType) -> TensorType:
        """obs should be of shape (batch_size, 32*4, screen_x, screen_y)"""
        hidden_state = self.representation_net(obs_batch)
        # See Appendix G.
        s_max = tf.math.reduce_max(hidden_state)
        s_min = tf.math.reduce_min(hidden_state)
        hidden_state = (hidden_state - s_min) / (s_max - s_min)
        return hidden_state
    
    def prediction(self, hidden_state: TensorType) -> Tuple[TensorType, TensorType]:
        """hidden_state should be of shape (batch_size,) + self.state_shape"""
        value, policy = self.prediction_net(hidden_state)
        return value, policy
    
    def dynamics(self, hidden_state: TensorType, action_batch: TensorType) -> Tuple[TensorType, TensorType]:
        """action should be of shape (batch_size) + self.state_shape[:-1]"""
        action_t = self._encode_atari_actions(action_batch)
        new_hidden_state, reward = self.dynamics_net((hidden_state, action_t))
        # See Appendix G.
        s_max = tf.math.reduce_max(new_hidden_state)
        s_min = tf.math.reduce_min(new_hidden_state)
        new_hidden_state = (new_hidden_state - s_min) / (s_max - s_min)
        return new_hidden_state, reward
