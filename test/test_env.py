import gym
import numpy as np
import pytest

from muzero.env import wrap_cartpole, FrameStackWithAction1D, FrameStackWithAction2D

def test_env_reproducibility():
    random_seed = 1

    env = wrap_cartpole(gym.make('CartPole-v0'))
    env.seed(random_seed)
    env.action_space.seed(random_seed)
    obs1 = env.reset()

    env = wrap_cartpole(gym.make('CartPole-v0'))
    env.seed(random_seed)
    env.action_space.seed(random_seed)
    obs2 = env.reset()

    assert obs1.tolist() == obs2.tolist()

    env.seed(random_seed)
    env.action_space.seed(random_seed)
    obs3 = env.reset()

    assert obs3.tolist() == obs2.tolist()

def framestack_assertions(env_name, framestack_cls):
    random_seed = 1
    frames = 3

    env = gym.make(env_name)
    env.seed(random_seed)
    env.action_space.seed(random_seed)
    env = framestack_cls(env, k=frames)
    obs1a = env.reset()
    assert obs1a.shape == env.frame_shape[:-1] + (env.frame_shape[-1] * frames,)

    env.step(1)
    obs1c, _, _, _ = env.step(1)
    assert obs1a.shape == obs1c.shape

    env2 = gym.make(env_name)
    env2.seed(random_seed)
    env2.action_space.seed(random_seed)
    obs21 = env2.reset()
    if len(obs21.shape) < len(env.frame_shape):
        obs21 = np.expand_dims(obs21, axis=-1)
    padding_shape = env.frame_shape[:-1] + (env.frame_shape[-1] * (frames - 1),)
    obs2a = np.concatenate([np.zeros(padding_shape), obs21, np.zeros(env.channel_shape)], axis=-1)
    assert np.allclose(obs1a, obs2a)

    obs22, _, _, _ = env2.step(1)
    obs23, _, _, _ = env2.step(1)
    if len(obs22.shape) < len(env.frame_shape):
        obs22 = np.expand_dims(obs22, axis=-1)
        obs23 = np.expand_dims(obs23, axis=-1)
    action0 = np.zeros(env.channel_shape)
    action1 = (np.ones(env.channel_shape) + 1) / env.action_count
    obs2c = np.concatenate([obs21, action0, obs22, action1, obs23, action1], axis=-1)
    assert np.allclose(obs1c, obs2c)

    return obs1a

def test_framestack_cartpole():
    obs = framestack_assertions('CartPole-v0', FrameStackWithAction1D)
    # Sanity check because test code is complex.
    assert obs.shape == (4, 6)

def test_framestack_breakout():
    obs = framestack_assertions('BreakoutNoFrameskip-v4', FrameStackWithAction2D)
    # Sanity check because test code is complex.
    assert obs.shape == (210, 160, 12)
