import gym
import numpy as np
import pytest
import tensorflow as tf

from .conftest import random_obs
from muzero.env import wrap_muzero
from muzero.muzero import ATARI_DEFAULT_CONFIG
from muzero.muzero_tf_model import MuZeroTFModelV2

@pytest.fixture(scope='module')
def atari_config():
    config = ATARI_DEFAULT_CONFIG
    config['value_max'] = 6
    return config

@pytest.fixture(scope='module')
def atari_model(atari_config: dict):
    env = wrap_muzero(gym.make('BreakoutNoFrameskip-v4'))
    return MuZeroTFModelV2(env.observation_space, env.action_space, atari_config)

def test_scalar_to_categorical(atari_model: MuZeroTFModelV2, atari_config: dict):
    t = np.array([[0.75, 4.5, 3], [1.7, 2, 3.2]], dtype='float32')
    u = atari_model.scalar_to_categorical(t, atari_config['value_max'])
    v = atari_model.expectation(u, atari_model.value_basis)
    w = atari_model.scalar_to_categorical(v, atari_config['value_max'])
    assert tf.experimental.numpy.allclose(t, v)
    assert tf.experimental.numpy.allclose(u, w)

def test_transform_np(atari_model: MuZeroTFModelV2):
    t = np.array([[0.75, 4.5, 3], [1.7, 2, 3.2]], dtype='float32')
    u = atari_model.transform(t)
    v = atari_model.untransform(u)
    w = atari_model.transform(v)
    assert np.allclose(t, v, atol=1e-3)
    assert np.allclose(u, w, atol=1e-3)

def test_transform_tf(atari_model: MuZeroTFModelV2):
    t = tf.convert_to_tensor([[0.75, 4.5, 3], [1.7, 2, 3.2]], dtype=tf.float32)
    u = atari_model.transform(t)
    v = atari_model.untransform(u)
    w = atari_model.transform(v)
    assert tf.experimental.numpy.allclose(t, v, atol=1e-3)
    assert tf.experimental.numpy.allclose(u, w, atol=1e-3)

def test_encode_actions(atari_model: MuZeroTFModelV2):
    t = atari_model._encode_atari_actions([0, 1])
    assert t.shape == (2, 6, 6, 4)
    assert tf.experimental.numpy.allclose(t[0,:,:,0], tf.ones((6, 6)))
    assert tf.experimental.numpy.allclose(t[0,:,:,1], tf.zeros((6, 6)))
    assert tf.experimental.numpy.allclose(t[0,:,:,2], tf.zeros((6, 6)))
    assert tf.experimental.numpy.allclose(t[0,:,:,3], tf.zeros((6, 6)))

def test_forward(atari_tensor_spec, atari_model: MuZeroTFModelV2):
    batch_size = 3
    t = random_obs(atari_tensor_spec[0], batch_size=batch_size, frames_per_obs=32)

    # Training.
    value, policy = atari_model.forward(t, is_training=True)
    assert value.shape == (batch_size,)
    assert policy.shape == (batch_size, atari_model.action_space_size)

    # Not training.
    value, policy = atari_model.forward(t, is_training=False)
    assert value.shape == (batch_size,)
    assert policy.shape == (batch_size, atari_model.action_space_size)
