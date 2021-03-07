from __future__ import annotations

from typing import Any, List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class Node:
    def __init__(self, action, reward, state, mcts, parent):
        action_space_size = mcts.model.action_space_size
        self.action = action  # Action used to go to this state

        self.is_expanded = False
        self.parent = parent
        self.children = np.empty(action_space_size, dtype=np.object_)

        # Use floats all around, because we need float division later on.
        self.children_q = np.zeros([action_space_size], dtype=np.float32)  # Q-values
        self.children_p = np.zeros([action_space_size], dtype=np.float32)  # Priors
        self.children_n = np.zeros([action_space_size], dtype=np.float32)  # Number of visits
        self.children_r = np.zeros([action_space_size], dtype=np.float32)  # Rewards

        if parent is not None:
            self.reward = reward
        self.state = state

        self.mcts = mcts

    @property
    def reward(self) -> float:
        return self.parent.children_r[self.action]
    
    @reward.setter
    def reward(self, r: float):
        self.parent.children_r[self.action] = r
    
    @property
    def visit_count(self) -> int:
        return self.parent.children_n[self.action]

    @visit_count.setter
    def visit_count(self, value: int):
        self.parent.children_n[self.action] = value

    @property
    def total_value(self) -> float:
        return self.parent.children_q[self.action]

    @total_value.setter
    def total_value(self, value: float):
        self.parent.children_q[self.action] = value

    @property
    def value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count else 0.

    @property
    def children_values(self) -> np.ndarray:
        return (self.children_n > 0).astype(np.float32) * np.divide(self.children_q, np.maximum(self.children_n, 1))

    def children_ucb_scores(self) -> float:
        pb_c = np.log((self.visit_count + self.mcts.puct_c2 + 1) / self.mcts.puct_c2) + self.mcts.puct_c1
        pb_c = np.multiply(pb_c, np.divide(np.sqrt(self.visit_count), self.children_n + 1))
        prior_scores = np.multiply(pb_c, self.children_p)
        
        value_scores = (self.children_n > 0).astype(np.float32) * (
            self.children_r + self.mcts.gamma * self.mcts.normalize_q(self.children_values)
        )
        
        return prior_scores + value_scores

    def best_action(self):
        """
        :return: action
        """
        return np.argmax(self.children_ucb_scores())

    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node

    def expand(self, children_priors):
        self.is_expanded = True
        self.children_p = children_priors

    def get_child(self, action):
        if self.children_n[action] == 0:
            self.children[action] = Node(
                state=None,
                action=action,
                parent=self,
                reward=0,
                mcts=self.mcts)
        return self.children[action]

    def backup(self, value):
        current = self
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            self.mcts.update_q_bounds(current.total_value / current.visit_count)

            value = current.reward + self.mcts.gamma * value

            current = current.parent


class RootNode(Node):
    def __init__(self, reward, state, mcts):
        super(RootNode, self).__init__(None, reward, state, mcts, None)
        self._reward = reward
        self._n = 0
        self._q = 0.
        self.is_expanded = True
    
    @property
    def reward(self) -> float:
        return self._reward
    
    @reward.setter
    def reward(self, r):
        self._reward = r

    @property
    def visit_count(self) -> int:
        return self._n

    @visit_count.setter
    def visit_count(self, value: int):
        self._n = value

    @property
    def total_value(self) -> float:
        return self._q

    @total_value.setter
    def total_value(self, value: float):
        self._q = value


def array_to_list(arr: np.ndarray) -> List[Any]:
    return [row.squeeze() for row in np.split(arr, arr.shape[0])]


class MCTS:

    def __init__(self, model, config, temperature=1.0, random_seed=1):
        np.random.seed(random_seed)

        self.model = model
        self.transform = config['transform_outputs']
        self.gamma = config['gamma']

        mcts_config = config['mcts']
        self.reset_q_bounds_per_node = mcts_config['reset_q_bounds_per_node']
        self.temperature = temperature
        self.num_sims = mcts_config["num_simulations"]
        self.exploit = mcts_config["argmax_tree_policy"]
        self.puct_c1 = mcts_config["puct_c1"]
        self.puct_c2 = mcts_config["puct_c2"]
        self.reset_q_bounds()
        
        if config['action_type'] == 'atari':
            self._extract_action = lambda a: a[0][0]
        elif config['action_type'] == 'board':
            self._extract_action = lambda a: a
        else:
            raise NotImplemented(f'Action type "{config["action_type"]}" unknown')

        self.add_dirichlet_noise = mcts_config["add_dirichlet_noise"]
        self.dir_epsilon = mcts_config["dirichlet_epsilon"] if 'dirichlet_epsilon' in mcts_config else 0
        self.dir_alpha = mcts_config["dirichlet_alpha"] if 'dirichlet_alpha' in mcts_config else None

        self._stats = {
            'action_count': 0,
        }
        for i in range(self.model.action_space_size):
            self._stats[f'action_{i}_count'] = 0
    
    @property
    def temperature(self):
        return self._temperature
    
    @temperature.setter
    def temperature(self, temperature):
        self._temperature = temperature

    def reset_q_bounds(self):
        self.min_q = np.array([float('inf')])
        self.max_q = np.array([-float('inf')])

    def normalize_q(self, q):
        if self.min_q >= self.max_q:
            return q
        return (q - self.min_q) / (self.max_q - self.min_q)

    def update_q_bounds(self, q):
        self.min_q = np.minimum(q, self.min_q)
        self.max_q = np.maximum(q, self.max_q)

    def model_prediction(self, batch_state_tf: tf.Tensor):
        # Superstitious introduction of extra variables so they can be explicitly deleted.
        batch_value_raw_tf, children_priors_tf = self.model.prediction(batch_state_tf)
        batch_value_tf = batch_value_raw_tf
        if self.model.is_value_categorical:
            batch_value_tf = self.model.expectation(batch_value_tf, self.model.value_basis)
        if self.transform:
            batch_value_tf = self.model.untransform(batch_value_tf)

        batch_value = batch_value_tf.numpy()
        children_priors = children_priors_tf.numpy()

        # This is required to prevent GPU OOM
        del batch_state_tf
        del batch_value_raw_tf
        del batch_value_tf
        del children_priors_tf

        return children_priors, batch_value

    def model_dynamics(self, leaves: List[Node]) -> tf.Tensor:
        prev_batch_state = tf.convert_to_tensor([leaf.parent.state for leaf in leaves])
        batch_action = tf.convert_to_tensor([leaf.action for leaf in leaves])
        batch_state_tf, batch_reward_tf = self.model.dynamics(prev_batch_state, batch_action)

        # Convert to Numpy arrays
        batch_state = batch_state_tf.numpy()
        batch_reward = batch_reward_tf.numpy()

        # This is required to prevent GPU OOM
        del prev_batch_state
        del batch_action
        del batch_reward_tf

        if self.model.is_reward_categorical:
            batch_reward = self.model.expectation_np(batch_reward, self.model.reward_basis_np)
        if self.transform:
            batch_reward = self.model.untransform(batch_reward)

        # Save the state and reward
        for j, state in enumerate(array_to_list(batch_state)):
            leaves[j].state = state
            leaves[j].reward = batch_reward[j]
        return batch_state_tf

    def expand_roots(self, nodes: List[Node], batch_state_tf: tf.Tensor):
        children_priors, _ = self.model_prediction(batch_state_tf)
        for node, p in zip(nodes, array_to_list(children_priors)):
            node.children_p = p

        if self.add_dirichlet_noise:
            self.add_noise(nodes)

    def expand_nodes(self, leaves: List[Node], batch_state_tf: tf.Tensor) -> None:
        children_priors, batch_value = self.model_prediction(batch_state_tf)
        batch_value = array_to_list(batch_value)
        children_priors = array_to_list(children_priors)
        for leaf, priors, value in zip(leaves, children_priors, batch_value):
            leaf.expand(priors)
            leaf.backup(value)

    def add_noise(self, nodes: List[Node]):
        for node in nodes:
            noise = np.random.dirichlet([self.dir_alpha] * self.model.action_space_size)
            node.children_p = (1 - self.dir_epsilon) * node.children_p + self.dir_epsilon * noise

    def compute_action(self, obs, debug=False):
        if self.reset_q_bounds_per_node:
            self.reset_q_bounds()
        batch_state_tf = self.model.representation(obs)
        batch_state = batch_state_tf.numpy()
        states = array_to_list(batch_state)
        nodes = [
            RootNode(
                reward=0.,
                state=state,
                mcts=self
            )
            for state in states
        ]
        self.expand_roots(nodes, batch_state_tf)

        for _ in range(self.num_sims):
            # Traverse to a leaf using choices based on upper confidence bounds
            leaves = [node.select() for node in nodes]
            for leaf in leaves:
                self._stats[f'action_{leaf.action}_count'] += 1
                self._stats['action_count'] += 1
            self.expand_nodes(leaves, self.model_dynamics(leaves))

        values = np.array([node.value for node in nodes])
        tree_policies, actions = self.select_action(nodes)

        if debug:
            return values, tree_policies, actions, nodes
        else:
            # Not sure if this is necessary
            for node in nodes:
                del node
            return values, tree_policies, actions
    
    def select_action(self, nodes: List[Node]) -> np.ndarray:
        # From Appendix D
        tree_policies = np.array([
            np.power(node.children_n, 1 / self.temperature)
            for node in nodes
        ])
        tree_policies = np.divide(tree_policies, np.sum(tree_policies, axis=1, keepdims=True))
        if self.exploit:
            # If exploit then choose action that has the max probability
            actions = np.argmax(tree_policies, axis=1)
        else:
            # Sample from the categorical distributions of tree_policies
            b, _ = tree_policies.shape
            cum_prob = np.cumsum(tree_policies, axis=-1)
            r = np.random.uniform(size=(b, 1))
            # Returns the indices of the first occurences of True
            actions = np.argmax(cum_prob > r, axis=-1)
        return tree_policies, actions

    def stats(self):
        return self._stats


class TFMCTS:
    """
    Broken. Needs to be updated to match MCTS which works.
    """

    def __init__(self, model, config, temperature=1.0):
        self.model = model
        self.gamma = config['gamma']

        mcts_config = config['mcts']
        self.reset_q_bounds_per_node = mcts_config['reset_q_bounds_per_node']
        self.temperature = temperature
        self.num_sims = mcts_config["num_simulations"]
        self.exploit = mcts_config["argmax_tree_policy"]
        self.puct_c1 = mcts_config["puct_c1"]
        self.puct_c2 = mcts_config["puct_c2"]
        self.reset_q_bounds()
        
        if config['action_type'] == 'atari':
            self._extract_action = lambda a: a[0][0]
        elif config['action_type'] == 'board':
            self._extract_action = lambda a: a
        else:
            raise NotImplemented(f'Action type "{config["action_type"]}" unknown')

        self.add_dirichlet_noise = mcts_config["add_dirichlet_noise"]
        self.dir_epsilon = mcts_config["dirichlet_epsilon"] if 'dirichlet_epsilon' in mcts_config else 0
        self.dir_alpha = mcts_config["dirichlet_alpha"] if 'dirichlet_alpha' in mcts_config else None
        if self.add_dirichlet_noise:
            self.dirichlet = tfp.distributions.Dirichlet([self.dir_alpha] * self.model.action_space_size)
    
    @property
    def temperature(self):
        return self._temperature
    
    @temperature.setter
    def temperature(self, temperature):
        if not tf.is_tensor(temperature):
            temperature = tf.constant(temperature, dtype=tf.float32)
        self._temperature = temperature

    def reset_q_bounds(self):
        self.min_q = tf.convert_to_tensor([float('inf')])
        self.max_q = tf.convert_to_tensor([-float('inf')])

    def normalize_q(self, q):
        if self.min_q >= self.max_q:
            return q
        return (q - self.min_q) / (self.max_q - self.min_q)

    def update_q_bounds(self, q):
        self.min_q = tf.math.maximum(q, self.min_q)
        self.max_q = tf.math.maximum(q, self.max_q)

    def compute_action(self, obs):
        if self.reset_q_bounds_per_node:
            self.reset_q_bounds()
        batch_state = self.model.representation(obs)
        states = tf.unstack(batch_state)
        nodes = [
            RootNode(
                reward=0.,
                state=state,
                mcts=self
            )
            for state in states
        ]
        for i in range(self.num_sims):
            # Traverse to a leaf using choices based on upper confidence bounds
            leaves = [node.select() for node in nodes]
            if i > 0:
                batch_state = tf.convert_to_tensor([leaf.parent.state for leaf in leaves])
                batch_action = tf.convert_to_tensor([leaf.action for leaf in leaves])
                batch_state, batch_reward = self.model.dynamics(batch_state, batch_action)
                if self.model.is_reward_categorical:
                    batch_reward = self.model.expectation(batch_reward, self.model.reward_basis)
                if self.transform:
                    batch_reward = self.model.untransform(batch_reward)
                for j, state in enumerate(tf.unstack(batch_state)):
                    leaves[j].state = state
                    leaves[j].reward = batch_reward[j]
            batch_value, children_priors = self.model.prediction(batch_state)
            if self.model.is_value_categorical:
                batch_value = self.model.expectation(batch_value, self.model.value_basis)
            if self.transform:
                batch_value = self.model.untransform(batch_value)
            if self.add_dirichlet_noise:
                noise = self.dirichlet.sample(1)
                children_priors = (1 - self.dir_epsilon) * children_priors + self.dir_epsilon * noise
            for leaf, priors, value in zip(leaves, tf.unstack(children_priors), tf.unstack(batch_value)):
                leaf.expand(priors)
                leaf.backup(value)

        values = tf.convert_to_tensor([node.value for node in nodes])
        # From Appendix D
        tree_policies = tf.convert_to_tensor([
            np.power(node.children_n, 1 / self.temperature)
            for node in nodes
        ])
        tree_policies = tf.math.divide(tree_policies, tf.math.reduce_sum(tree_policies, axis=1, keepdims=True))
        if self.exploit:
            # if exploit then choose action that has the maximum
            # tree policy probability
            actions = tf.math.argmax(tree_policies, axis=1)
        else:
            # otherwise sample an action according to tree policy probabilities
            categorical = tfp.distributions.Categorical(probs=tree_policies)
            actions = categorical.sample(len(values))
        return values, tree_policies, actions
