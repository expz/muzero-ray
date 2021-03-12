"""
This code is mostly from the pseudocode posted on arxiv at

https://arxiv.org/src/1911.08265v2/anc/pseudocode.py
"""

import collections
import gym
import math
import typing
from typing import Any, Dict, List, Optional

import numpy
import pytest
import tensorflow as tf

from muzero.env import wrap_cartpole
from muzero.muzero import CARTPOLE_DEFAULT_CONFIG
from muzero.tf_model import MuZeroTFModelV2


MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value

class MuZeroConfig(object):

  def __init__(self,
               action_space_size: int,
               max_moves: int,
               discount: float,
               dirichlet_alpha: float,
               num_simulations: int,
               batch_size: int,
               td_steps: int,
               num_actors: int,
               lr_init: float,
               lr_decay_steps: float,
               visit_softmax_temperature_fn,
               known_bounds: Optional[KnownBounds] = None):
    ### Self-Play
    self.action_space_size = action_space_size
    self.num_actors = num_actors

    self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
    self.max_moves = max_moves
    self.num_simulations = num_simulations
    self.discount = discount

    # Root prior exploration noise.
    self.root_dirichlet_alpha = dirichlet_alpha
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
    self.known_bounds = known_bounds

    ### Training
    self.training_steps = int(1000e3)
    self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = batch_size
    self.num_unroll_steps = 5
    self.td_steps = td_steps

    self.weight_decay = 1e-4
    self.momentum = 0.9

    # Exponential learning rate schedule
    self.lr_init = lr_init
    self.lr_decay_rate = 0.1
    self.lr_decay_steps = lr_decay_steps


def make_board_game_config(action_space_size: int, max_moves: int,
                           dirichlet_alpha: float,
                           lr_init: float) -> MuZeroConfig:

  def visit_softmax_temperature(num_moves, training_steps):
    if num_moves < 30:
      return 1.0
    else:
      return 0.0  # Play according to the max.

  return MuZeroConfig(
      action_space_size=action_space_size,
      max_moves=max_moves,
      discount=1.0,
      dirichlet_alpha=dirichlet_alpha,
      num_simulations=800,
      batch_size=2048,
      td_steps=max_moves,  # Always use Monte Carlo return.
      num_actors=3000,
      lr_init=lr_init,
      lr_decay_steps=400e3,
      visit_softmax_temperature_fn=visit_softmax_temperature,
      known_bounds=KnownBounds(-1, 1))


def make_go_config() -> MuZeroConfig:
  return make_board_game_config(
      action_space_size=362, max_moves=722, dirichlet_alpha=0.03, lr_init=0.01)


def make_chess_config() -> MuZeroConfig:
  return make_board_game_config(
      action_space_size=4672, max_moves=512, dirichlet_alpha=0.3, lr_init=0.1)


def make_shogi_config() -> MuZeroConfig:
  return make_board_game_config(
      action_space_size=11259, max_moves=512, dirichlet_alpha=0.15, lr_init=0.1)


def make_atari_config() -> MuZeroConfig:

  def visit_softmax_temperature(num_moves, training_steps):
    if training_steps < 500e3:
      return 1.0
    elif training_steps < 750e3:
      return 0.5
    else:
      return 0.25

  return MuZeroConfig(
      action_space_size=18,
      max_moves=27000,  # Half an hour at action repeat 4.
      discount=0.997,
      dirichlet_alpha=0.25,
      num_simulations=15,
      batch_size=48,
      td_steps=10,
      num_actors=4,
      lr_init=0.001,
      lr_decay_steps=350e3,
      visit_softmax_temperature_fn=visit_softmax_temperature)

def make_cartpole_config(config) -> MuZeroConfig:
  return MuZeroConfig(
    action_space_size=2,
    max_moves=300,
    discount=config['gamma'],
    dirichlet_alpha=config['mcts']['dirichlet_alpha'],
    num_simulations=config['mcts']['num_simulations'],
    batch_size=config['replay_batch_size'],
    td_steps=config['n_step'],
    num_actors=4,
    lr_init=config['lr'],
    lr_decay_steps=500e3,
    visit_softmax_temperature_fn=lambda x, y: 1)


class Action(object):

  def __init__(self, index: int):
    self.index = index

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    return self.index == other.index

  def __gt__(self, other):
    return self.index > other.index


class Player(object):
  pass


class Node(object):

  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.hidden_state = None
    self.reward = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history: List[Action], action_space_size: int):
    self.history = list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: Action):
    self.history.append(action)

  def last_action(self) -> Action:
    return self.history[-1]

  def action_space(self) -> List[Action]:
    return [Action(i) for i in range(self.action_space_size)]

  def to_play(self) -> Player:
    return Player()


class NetworkOutput(typing.NamedTuple):
  value: float
  reward: float
  policy: Dict[Action, float]
  hidden_state: tf.Tensor


class Network:
  def __init__(self, env: gym.Env, config: Dict[str, Any]):
    self.action_space_size = env.action_space.n
    self.model = MuZeroTFModelV2(
      env.observation_space,
      env.action_space,
      config)

  def initial_inference(self, observation: tf.Tensor) -> NetworkOutput:
    observation = numpy.expand_dims(observation, axis=0)
    hidden_state = self.model.representation(observation)
    value, policy = self.model.prediction(hidden_state)
    value = self.model.expectation(value, self.model.value_basis).numpy()[0]
    policy = policy.numpy()[0]
    policy_dict = {
        Action(i): policy[i] for i in range(self.action_space_size)
    }
    return NetworkOutput(
      value,
      0,
      policy_dict,
      hidden_state)


  def recurrent_inference(self, hidden_state: tf.Tensor, action: Action) -> NetworkOutput:
    next_state, reward = self.model.dynamics(hidden_state, [action.index])
    value, policy = self.model.prediction(next_state)
    value = self.model.expectation(value, self.model.value_basis).numpy()[0]
    reward = self.model.expectation(reward, self.model.reward_basis).numpy()[0]
    policy = policy.numpy()[0]
    policy_dict = {
        Action(i): policy[i] for i in range(self.action_space_size)
    }
    return NetworkOutput(
      value,
      reward,
      policy_dict,
      next_state)


def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory,
             network: Network):
  min_max_stats = MinMaxStats(config.known_bounds)

  for _ in range(config.num_simulations):
    history = action_history.clone()
    node = root
    search_path = [node]

    while node.expanded():
      action, node = select_child(config, node, min_max_stats)
      history.add_action(action)
      search_path.append(node)

    parent = search_path[-2]
    network_output = network.recurrent_inference(parent.hidden_state,
                                                 history.last_action())

    expand_node(node, history.to_play(), history.action_space(), network_output)

    backpropagate(search_path, network_output.value, history.to_play(),
                  config.discount, min_max_stats)


# Select the child with the highest UCB score.
def select_child(config: MuZeroConfig, node: Node,
                 min_max_stats: MinMaxStats):
  child_idx = int(numpy.argmax([ucb_score(config, node, child, min_max_stats) for child in node.children.values()]))
  return Action(child_idx), node.children[Action(child_idx)]

  #_, action, child = max(
  #    (ucb_score(config, node, child, min_max_stats), action,
  #     child) for action, child in node.children.items())
  #return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: MuZeroConfig, parent: Node, child: Node,
              min_max_stats: MinMaxStats) -> float:
  pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  if child.visit_count > 0:
    value_score = child.reward + config.discount * min_max_stats.normalize(
        child.value())
  else:
    value_score = 0
  return prior_score + value_score


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(node: Node, to_play: Player, actions: List[Action],
                network_output: NetworkOutput):
  node.to_play = to_play
  node.hidden_state = network_output.hidden_state
  node.reward = network_output.reward
  policy = network_output.policy
  assert pytest.approx(1) == sum(policy.values())
  for action, p in policy.items():
    node.children[action] = Node(p)


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play: Player,
                  discount: float, min_max_stats: MinMaxStats):
  for node in reversed(search_path):
    node.value_sum += value # TODO: if node.to_play == to_play else -value
    node.visit_count += 1
    min_max_stats.update(node.value())
    value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MuZeroConfig, node: Node):
  actions = list(node.children.keys())
  noise = numpy.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def cartpole_mcts_unoptimized(network_config: Dict[str, Any]) -> Node:
  numpy.random.seed(network_config['random_seed'])
  env = wrap_cartpole(gym.make('CartPole-v0'))
  env.seed(network_config['random_seed'])
  env.action_space.seed(network_config['random_seed'])
  network = Network(env, network_config)
  config = make_cartpole_config(network_config)
  model = MuZeroTFModelV2(env.observation_space, env.action_space, network_config)

  root = Node(0)
  current_observation = env.reset()
  action_history = ActionHistory([], 2)
  expand_node(root, Player(), action_history.action_space(),
              network.initial_inference(current_observation))
  add_exploration_noise(config, root)

  # We then run a Monte Carlo Tree Search using only action sequences and the
  # model learned by the network.
  run_mcts(config, root, action_history, network)
  return root
