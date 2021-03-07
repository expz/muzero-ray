from test.mcts_unoptimized import cartpole_mcts_unoptimized
from typing import Any, Dict
import gym
import numpy as np
import pytest

from muzero.env import wrap_cartpole
from muzero.mcts import Node, RootNode, MCTS
from muzero.muzero import CARTPOLE_DEFAULT_CONFIG
from muzero.muzero_tf_model import MuZeroTFModelV2

"""
import gym
import numpy as np

from muzero.env import wrap_cartpole
from muzero.mcts import Node, RootNode, MCTS
from muzero.muzero import CARTPOLE_DEFAULT_CONFIG
from muzero.muzero_tf_model import MuZeroTFModelV2

config = CARTPOLE_DEFAULT_CONFIG
env = wrap_cartpole(gym.make('CartPole-v0'))
model = MuZeroTFModelV2(env.observation_space, env.action_space, cartpole_config)
mcts = MCTS(cartpole_model, cartpole_config)

"""

@pytest.fixture()
def cartpole_config():
    config = CARTPOLE_DEFAULT_CONFIG
    return config

@pytest.fixture()
def cartpole_model(cartpole_config):
    env = wrap_cartpole(gym.make('CartPole-v0'))
    return MuZeroTFModelV2(env.observation_space, env.action_space, cartpole_config)

def test_node(cartpole_model, cartpole_config):
    action = 1
    reward = 2
    mcts = MCTS(cartpole_model, cartpole_config)
    root = RootNode(reward=1, state=None, mcts=mcts)
    node = Node(action=action, reward=reward, state=None, mcts=mcts, parent=root)
    assert node.reward == reward
    assert root.children_r[action] == reward

    node.backup(3)
    assert node.total_value == 3.0
    assert node.value == 3.0
    assert node.visit_count == 1

    mcts.normalize_q(np.array([1, 2]))

def cartpole_mcts_optimized(config: Dict[str, Any]):
    env = wrap_cartpole(gym.make('CartPole-v0'))
    env.seed(config['random_seed'])
    env.action_space.seed(config['random_seed'])
    current_observation = env.reset()
    current_observation = np.expand_dims(current_observation, axis=0)

    cartpole_model = MuZeroTFModelV2(env.observation_space, env.action_space, config)

    mcts = MCTS(cartpole_model, config, random_seed=config['random_seed'])
    values, policies, actions, roots = mcts.compute_action(current_observation, debug=True)

    assert len(roots) == 1
    return roots[0]


def test_cartpole_mcts(cartpole_config: Dict[str, Any]):
    # Make sure the only use of randomness is the call to dirichlet.
    cartpole_config['argmax_tree_policy'] = True
    cartpole_config['train_batch_size'] = 1
    cartpole_config['replay_batch_size'] = 1

    optimized_root = cartpole_mcts_optimized(cartpole_config)
    unoptimized_root = cartpole_mcts_unoptimized(cartpole_config)

    optimized_nodes = [optimized_root] 
    unoptimized_nodes = [unoptimized_root]
    while optimized_nodes:
        opt_node = optimized_nodes.pop()
        unopt_node = unoptimized_nodes.pop()
        if opt_node.is_expanded:
            for child in opt_node.children:
                if child is not None:
                    optimized_nodes.append(child)
            assert len([child for child in unopt_node.children.values() if child.visit_count > 0]) == len([child for child in opt_node.children if child is not None])
            for child in unopt_node.children.values():
                if child.visit_count > 0:
                    unoptimized_nodes.append(child)
        assert opt_node.visit_count == unopt_node.visit_count
        assert pytest.approx(opt_node.reward, 1e-4) == unopt_node.reward
        assert pytest.approx(opt_node.total_value, 1e-4) == unopt_node.value_sum

    assert len(unoptimized_nodes) == 0

    cartpole_config['argmax_tree_policy'] = False
