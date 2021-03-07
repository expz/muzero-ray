import gym
import numpy as np
import pytest

from muzero.env import wrap_cartpole

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