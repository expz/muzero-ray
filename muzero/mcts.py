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
        #print('visit count:', self.visit_count.shape if hasattr(self.visit_count, 'shape') else 'no shape')
        #print('viis count type:', type(self.visit_count))
        return self.total_value / self.visit_count if self.visit_count else 0.

    def children_ucb_scores(self) -> float:
        pb_c = np.log((self.visit_count + self.mcts.puct_c2 + 1) / self.mcts.puct_c2) + self.mcts.puct_c1
        #print('pb_c', pb_c)
        pb_c = np.multiply(pb_c, np.divide(np.sqrt(self.visit_count), self.children_n + 1))
        #print('pb_c 2', pb_c)
        prior_scores = np.multiply(pb_c, self.children_p)
        #print('prior scores', prior_scores)
        
        #print('children_rewards', self.children_r)
        #print('children q', self.children_q)
        value_scores = (self.children_n > 0).astype(np.float32) * (
            self.children_r + self.mcts.gamma * self.mcts.normalize_q(self.children_q)
        )
        #print('value scores', value_scores)
        
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
                reward=None,
                mcts=self.mcts)
        return self.children[action]

    def backup(self, value):
        current = self
        while current is not None:
            self.mcts.update_q_bounds(value)
            current.visit_count += 1
            current.total_value += value
            
            value = current.reward + self.mcts.gamma * value
            current = current.parent


class RootNode(Node):
    def __init__(self, reward, state, mcts):
        super(RootNode, self).__init__(None, reward, state, mcts, None)
        self._reward = reward
        self._n = 0
        self._q = 0.
    
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


class MCTS:

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

        #self.bomb_fuse = 10
    
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
        self.min_q = np.maximum(q, self.min_q)
        self.max_q = np.maximum(q, self.max_q)

    def compute_action(self, obs):
        #print('in compute_action', self.bomb_fuse)
        if self.reset_q_bounds_per_node:
            self.reset_q_bounds()
        batch_state = self.model.representation(obs)
        #print('original batch state', batch_state.shape)
        states = tf.unstack(batch_state)
        #print('states', len(states), states[0].shape)
        nodes = [
            RootNode(
                reward=0.,
                state=state,
                mcts=self
            )
            for state in states
        ]
        #print('before loop')
        for i in range(self.num_sims):
            # Traverse to a leaf using choices based on upper confidence bounds
            leaves = [node.select() for node in nodes]
            if i > 0:
                batch_state = tf.convert_to_tensor([leaf.parent.state for leaf in leaves])
                batch_action = tf.convert_to_tensor([leaf.action for leaf in leaves])
                batch_state, batch_reward = self.model.dynamics(batch_state, batch_action)
                if self.model.is_reward_categorical:
                    batch_reward = self.model.expectation(batch_reward, self.model.reward_basis)
                batch_reward = self.model.untransform(batch_reward)
                #print('batch_state:', batch_state.shape)
                #print('batch_reward:', batch_reward.shape)
                #print('batchreward[0]:', batch_reward[0].shape if hasattr(batch_reward[0], 'shape') else type(batch_reward[0]))
                for j, state in enumerate(tf.unstack(batch_state)):
                    leaves[j].state = state
                    leaves[j].reward = batch_reward[j]
            batch_value, children_logits = self.model.prediction(batch_state)
            if self.model.is_value_categorical:
                batch_value = self.model.expectation(batch_value, self.model.value_basis)
            batch_value = np.array(self.model.untransform(batch_value))
            children_priors = tf.nn.softmax(children_logits, axis=-1)
            children_priors = np.array(children_priors)
            #print('batch_value:', batch_value.shape)
            #print('children priors:', children_priors.shape)
            if self.add_dirichlet_noise:
                noise = np.random.dirichlet([self.dir_alpha] * self.model.action_space_size)
                #print("noise:", noise.shape)
                children_priors = (1 - self.dir_epsilon) * children_priors + self.dir_epsilon * noise
                #print('new children priors:', children_priors.shape, 'vals:', children_priors)
            batch_value = [value.squeeze() for value in np.split(batch_value, batch_value.shape[0])]
            children_priors = [priors.squeeze() for priors in np.split(children_priors, children_priors.shape[0])]
            for leaf, priors, value in zip(leaves, children_priors, batch_value):
                leaf.expand(priors)
                #print('value for backup:', value.shape)
                leaf.backup(value)

        values = np.array([node.value for node in nodes])
        tree_policies = np.array([
            #np.power(node.children_n / node.visit_count, self.temperature)
            np.power(node.children_n, 1 / self.temperature)
            for node in nodes
        ])
        #tree_policies = tf.nn.softmax(tree_policies)
        tree_policies = np.divide(tree_policies, np.sum(tree_policies, axis=1, keepdims=True))
        #print(tree_policies)
        #self.bomb_fuse -= 1
        #if not self.bomb_fuse:
        #    raise Exception('boom! this thread exploded')
        if self.exploit:
            # if exploit then choose action that has the maximum
            # tree policy probability
            actions = np.argmax(tree_policies, axis=1)
        else:
            # otherwise sample an action according to tree policy probabilities
            #categorical = tfp.distributions.Categorical(logits=tree_policies)
            # Sample from the categorical distributions of tree_policies
            b, _ = tree_policies.shape
            cum_prob = np.cumsum(tree_policies, axis=-1)
            r = np.random.uniform(size=(b, 1))
            actions = np.argmax(cum_prob > r, axis=-1)
        #print('mcts compute actions:', values.shape, tree_policies.shape, actions)
        return values, tree_policies, actions


class TFMCTS:

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

        #self.bomb_fuse = 10
    
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
        #print('in compute_action', self.bomb_fuse)
        if self.reset_q_bounds_per_node:
            self.reset_q_bounds()
        batch_state = self.model.representation(obs)
        #print('original batch state', batch_state.shape)
        states = tf.unstack(batch_state)
        #print('states', len(states), states[0].shape)
        nodes = [
            RootNode(
                reward=0.,
                state=state,
                mcts=self
            )
            for state in states
        ]
        #print('before loop')
        for i in range(self.num_sims):
            # Traverse to a leaf using choices based on upper confidence bounds
            leaves = [node.select() for node in nodes]
            if i > 0:
                batch_state = tf.convert_to_tensor([leaf.parent.state for leaf in leaves])
                batch_action = tf.convert_to_tensor([leaf.action for leaf in leaves])
                batch_state, batch_reward = self.model.dynamics(batch_state, batch_action)
                if self.model.is_reward_categorical:
                    batch_reward = self.model.expectation(batch_reward, self.model.reward_basis)
                batch_reward = self.model.untransform(batch_reward)
                #print('batch_state:', batch_state.shape)
                #print('batch_reward:', batch_reward.shape)
                #print('batchreward[0]:', batch_reward[0].shape if hasattr(batch_reward[0], 'shape') else type(batch_reward[0]))
                for j, state in enumerate(tf.unstack(batch_state)):
                    leaves[j].state = state
                    leaves[j].reward = batch_reward[j]
            batch_value, children_priors_logits = self.model.prediction(batch_state)
            if self.model.is_value_categorical:
                batch_value = self.model.expectation(batch_value, self.model.value_basis)
            batch_value = self.model.untransform(batch_value)
            #print('batch_value:', batch_value.shape)
            children_priors = tf.nn.softmax(children_priors_logits)
            #print('children priors:', children_priors.shape)
            if self.add_dirichlet_noise:
                noise = self.dirichlet.sample(1)
                #print("noise:", noise.shape)
                children_priors = (1 - self.dir_epsilon) * children_priors + self.dir_epsilon * noise
                #print('new children priors:', children_priors.shape, 'vals:', children_priors)
            for leaf, priors, value in zip(leaves, tf.unstack(children_priors), tf.unstack(batch_value)):
                leaf.expand(priors)
                #print('value for backup:', value.shape)
                leaf.backup(value)

        values = tf.convert_to_tensor([node.value for node in nodes])
        tree_policies = tf.convert_to_tensor([
            #np.power(node.children_n / node.visit_count, self.temperature)
            np.power(node.children_n, 1 / self.temperature)
            for node in nodes
        ])
        #tree_policies = tf.nn.softmax(tree_policies)
        tree_policies = tf.math.divide(tree_policies, tf.math.reduce_sum(tree_policies, axis=1, keepdims=True))
        #print(tree_policies)
        #self.bomb_fuse -= 1
        #if not self.bomb_fuse:
        #    raise Exception('boom! this thread exploded')
        if self.exploit:
            # if exploit then choose action that has the maximum
            # tree policy probability
            actions = tf.math.argmax(tree_policies, axis=1)
        else:
            # otherwise sample an action according to tree policy probabilities
            #categorical = tfp.distributions.Categorical(logits=tree_policies)
            categorical = tfp.distributions.Categorical(probs=tree_policies)
            actions = categorical.sample(len(values))
        #print('mcts compute actions:', values.shape, tree_policies.shape, actions)
        return values, tree_policies, actions
