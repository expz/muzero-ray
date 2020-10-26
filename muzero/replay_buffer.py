import collections
from collections import namedtuple
import logging
import numpy as np
import platform
import tensorflow as tf
from typing import List

import ray
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, \
    DEFAULT_POLICY_ID
from ray.rllib.utils.types import SampleBatchType
from ray.rllib.utils.timer import TimerStat
from ray.util.iter import ParallelIteratorWorker

# Constant that represents all policies in lockstep replay mode.
_ALL_POLICIES = "__all__"

logger = logging.getLogger(__name__)


def log2(n):
  """
  Smallest integer greater than or equal to log_2(n).
  """
  i = 1
  log = 0
  while n > i:
    log += 1
    i *= 2
  return log


class StructureList(tf.Module):
  """
  Some code copied from
  https://github.com/tensorflow/agents/blob/master/tf_agents/replay_buffers/table.py
  """
  def __init__(self, length, tensor_spec, scope='structure_list'):
    super(StructureList, self).__init__(name='StructureList')
    self._tensor_spec = tensor_spec
    self.length = length

    def _create_unique_slot_name(spec):
      return tf.compat.v1.get_default_graph().unique_name(spec.name or 'slot')

    self._slots = tf.nest.map_structure(_create_unique_slot_name,
                                        self._tensor_spec)

    def _create_storage(spec, slot_name):
      """Create storage for a slot, track it."""
      shape = [self.length] + spec.shape.as_list()
      new_storage = tf.Variable(
          name=slot_name,
          initial_value=tf.zeros(shape, dtype=spec.dtype),
          shape=None,
          dtype=spec.dtype)
      return new_storage

    with tf.compat.v1.variable_scope(scope):
      self._storage = tf.nest.map_structure(_create_storage, self._tensor_spec,
                                            self._slots)

    self._slot2storage_map = dict(
        zip(tf.nest.flatten(self._slots), tf.nest.flatten(self._storage)))

  def add_batch(self, values, rows, slots=None):
    slots = slots or self._slots
    flattened_slots = tf.nest.flatten(slots)
    flattened_values = tf.nest.flatten(values)
    for slot, value in zip(flattened_slots, flattened_values):
      update = tf.IndexedSlices(value, rows)
      self._slot2storage_map[slot].scatter_update(update, use_locking=True)

  def select(self, indices, slots=None):
    def _gather(tensor):
      return tf.gather(tensor,
                       tf.convert_to_tensor(indices, dtype=tf.int32),
                       axis=0)

    slots = slots or self._slots
    return tf.nest.map_structure(_gather, self._storage)

  def size_bytes(self):
    storage_size = 0
    storage_bytes = 0
    for var in tf.nest.flatten(self._storage):
      var_size = var.shape.num_elements()
      storage_size += var_size
      storage_bytes += var_size * var.dtype.size
    return storage_size, storage_bytes

Node = namedtuple('Node', ['depth', 'num'])

class SumTree:
  def __init__(self, capacity):
    self.capacity = capacity
    self._tree_depth = log2(capacity)

  @property
  def size(self):
    pass

  @property
  def count(self):
    pass

  def _root(self):
    return Node(0, 0)

  def _is_leaf(self, node):
    return node.depth == self._tree_depth
  
  def _left_child(self, node):
    new_num = 2 * node.num
    return Node(node.depth + 1, new_num) if not self._is_leaf(node) else None

  def _right_child(self, node):
    new_num = 2 * node.num + 1
    return Node(node.depth + 1, new_num) if not self._is_leaf(node) else None

  def _parent(self, node):
    return Node(node.depth - 1, node.num // 2) if node.depth > 0 else None

  def _node_to_index(self, node):
    return (1 << node.depth) + node.num - 1

  def _get_node_for_id(self, id_):
    return Node(self._tree_depth, (id_ % self.capacity))

  def _get_nodes_for_id(self, id_, batch_size):
    return [
      Node(self._tree_depth, (id_ % self.capacity) + i)
      for i in range(batch_size)
    ]

  def add_batch(self, items, probs):
    pass

  def sample_batch(self, batch_size):
    pass

class IntPairSumTree(SumTree):
  def __init__(self, capacity):
    SumTree.__init__(self, capacity)
    self._leaf_offset = (1 << self._tree_depth) - 1
    self._id_tree = [0] * ((1 << (self._tree_depth + 1)) - 1)
    self._data_list = np.full((capacity, 2), -1, dtype=np.int32)
    self._next_element_id = 0

  def add(self, a, b, prob):
    node = self._get_node_for_id(self._next_element_id)
    idx = node.num
    k = self._node_to_index(node)
    delta_p = prob - self._id_tree[k]
    self._id_tree[k] = prob
    node = self._parent(node)
    while node is not None:
      self._id_tree[self._node_to_index(node)] += delta_p
      node = self._parent(node)
    overwritten = None
    if self._next_element_id >= self.capacity:
      overwritten = self._data_list[idx, :]
    self._data_list[idx, :] = [a, b]
    self._next_element_id += 1
    return overwritten

  def add_batch(self, items, probs):
    batch_size = items.shape[0]
    leaves = self._get_nodes_for_id(self._next_element_id, batch_size)
    indices = [node.num for node in leaves]
    for node, p in zip(leaves, probs):
      k = self._node_to_index(node)
      delta_p = p - self._id_tree[k]
      self._id_tree[k] = p
      node = self._parent(node)
      while node is not None:
        self._id_tree[self._node_to_index(node)] += delta_p
        node = self._parent(node)
    overwritten = None
    if self._next_element_id >= self.capacity:
      overwritten = self._data_list[indices, :]
    self._data_list[indices, :] = items
    self._next_element_id += batch_size
    return overwritten
 
  def _sample_probs(self, batch_size):
    max_p = self._id_tree[0]
    return np.random.uniform(low=0.0, high=max_p, size=[batch_size])

  def sample_batch(self, batch_size):
    if not self._id_tree or not self._id_tree[0]:
      return None
    indices = [0] * batch_size
    ps = self._sample_probs(batch_size)
    for i in range(batch_size):
      node = self._root()
      p = ps[i]
      while not self._is_leaf(node):
        left = self._left_child(node)
        right = self._right_child(node)
        left_val = self._id_tree[self._node_to_index(left)]
        if p <= left_val:
          node = left
        else:
          node = right
          p -= left_val
      indices[i] = node.num
    return self._data_list[indices]

  def update(self, idx, weight):
    node = Node(self._tree_depth, idx)
    delta = weight - self._id_tree[self._node_to_index(node)]
    self._id_tree[self._node_to_index(node)] += delta
    while self._parent(node):
      node = self._parent(node)
      self._id_tree[self._node_to_index(node)] += delta


class StructureSumTree(tf.Module, SumTree):
  def __init__(self,
               capacity,
               tensor_spec,
               scope='sum_tree',
               device='cpu:*'):
    tf.Module.__init__(self, name='SumTree')
    SumTree.__init__(self, capacity)

    def _add_batch_dim(spec):
      return tf.TensorSpec([None] + spec.shape, spec.dtype)

    self._data_spec = tensor_spec
    self._batch_spec = tf.nest.map_structure(_add_batch_dim, tensor_spec)
    self._capacity = tf.int64(capacity)
    self._scope = scope
    self._device = device
    self._leaf_offset = 1 << self._tree_depth - 1
    id_spec = tf.TensorSpec([1], tf.float32, 'p')
    self._id_tree = [0] * ((1 << (self._tree_depth + 1)) - 1)
    self._data_list = StructureList(tensor_spec, capacity, scope)
    self._next_element_id = tf.Variable(tf.convert_to_tensor(0, dtype=tf.int64), name='next_batch_id')
    self._random = tf.random.Generator.from_seed(42)

  def variables(self):
      return (
        self._data_list.variables(),
        self._next_element_id)

  @property
  def device(self):
    return self._device

  @property
  def scope(self):
    return self._scope

  @property
  def size(self):
    return tf.minimum(self._next_element_id, self._capacity)

  @property
  def count(self):
    return self._next_element_id

  def add_batch(self, items, probs):
    # Validate tuple structure against spec's tuple structure
    tf.nest.assert_same_structure(items, self._batch_spec)
    # Validate tensors against specs
    flattened_specs = tf.nest.flatten(self._batch_spec)
    flattened_items = tf.nest.flatten(items)
    if any(
        not spec.is_compatible_with(tensor)
        for spec, tensor in zip(flattened_specs, flattened_items)):
      raise Exception(
          f'TensorSpec not compatible with Tensor. '
          f'Specs: {flattened_specs}. '
          f'Tensors: {[tf.TensorSpec.from_tensor(t) for t in flattened_items]}.'
      )

    batch_size = flattened_items[0].shape[0]
    with tf.device(self._device), tf.name_scope(self._scope):
      leaves = self._get_nodes_for_id(self._next_element_id, batch_size)
      for node, p in zip(leaves, probs):
        k = self._node_to_index(node)
        delta_p = p - self._id_tree[k]
        self._id_tree[k] = p
        node = self._parent(node)
        while node is not None:
          self._id_tree[self._node_to_index(node)] += delta_p
          node = self._parent(node)
      indices = [node.num for node in leaves]
      overwritten = None
      if self._next_element_id >= self._capacity:
        overwritten = self._data_list.select(indices)
      self._data_list.add_batch(items, indices)
      self._next_batch_id += batch_size
      return overwritten
 
  def _sample_probs(self, batch_size):
    max_p = self._id_tree[0]
    return self._random.uniform(shape=[batch_size], maxval=max_p)

  def sample_batch(self, batch_size):
    if not self._id_tree or not self._id_tree[0]:
      return None
    indices = [0] * batch_size
    ps = self._sample_probs()
    for i in range(batch_size):
      node = self._root()
      p = ps[i]
      while not self._is_leaf(node):
        left = self._left_child(node)
        right = self._right_child(node)
        left_val = self._id_tree[self._node_to_index(left)]
        if p <= left_val:
          node = left
        else:
          node = right
          p -= left_val
      indices[i] = node.num
    return self._data_list.select(indices)

class ReplayBuffer(tf.Module):

  def __init__(self, capacity, tensor_spec, scope='replay_buffer', name='ReplayBuffer'):
    super(ReplayBuffer, self).__init__(name=name)
    self._tensor_spec = tensor_spec
    self._capacity = capacity
    self._scope = scope

  def add(self, item: SampleBatchType, weight: float):
    pass

  def sample(self, num_items: int):
    pass


class PrioritizedReplayBuffer(ReplayBuffer):
  def __init__(self, capacity, tensor_spec,
               scope='prioritized_replay_buffer', alpha=1,
               frames_per_obs=32):
    super(PrioritizedReplayBuffer, self).__init__(
      capacity,
      tensor_spec,
      scope,
      'PrioritizedReplayBuffer')
    self.alpha = alpha
    self._frame_idx = 0
    # Tree of (episode, frame) pairs with priorities
    self._tree = IntPairSumTree(capacity)
    # Buffer of frames
    full_spec = (
      tensor_spec,
      tf.TensorSpec((1,), dtype=tf.int32),
      tf.TensorSpec((1,), dtype=tf.float32),
      tf.TensorSpec((1,), dtype=tf.int32),
      tf.TensorSpec((1,), dtype=tf.float32),
      tf.TensorSpec((1,), dtype=tf.int32),
      tf.TensorSpec((1,), dtype=tf.int32),
    )
    self._frames = StructureList(capacity, full_spec, scope='frame_buffer')
    # (episode, steps) => index of frame in self._steps
    self._episode_steps = {}
    self._batches_added = 0
    self._batches_sampled = 0
    self.channels_per_frame = tensor_spec.shape[-1]
    self.frames_per_obs = 32

  def add(self, item: SampleBatchType, weight: float):
    assert len(item[SampleBatch.EPS_ID]) == 1
    assert len(item['t']) == 1
    episode, step = item[SampleBatch.EPS_ID][0], item['t'][0]
    wt = item['weights'][0]  # = weight
    wt = np.abs(wt)
    prob = np.pow(wt, self.alpha) if self.alpha != 1 else wt
    overwritten = self._tree.add(episode, step, prob)
    if overwritten is not None:
      eps, st = overwritten
      del self._episode_steps[(eps, st)]
    obs = np.expand_dims(item[SampleBatch.CUR_OBS][0], axis=0)
    cnt = obs.shape[-1] // self.channels_per_frame
    simple_obs = [
      np.take(obs, list(range(i * 4, i * 4 + 4)), axis=-1)
      for i in range(cnt)
    ]
    os = []
    o_steps = []
    rows = []
    for i, o in enumerate(simple_obs):
      o_step = step - (cnt - i - 1)
      if (episode, o_step) not in self._episode_steps:
        os.append(o)
        o_steps.append(o_step)
        rows.append(self._frame_idx % self._frames.length)
        self._episode_steps[(episode, o_step)] = self._frame_idx
        self._frame_idx += 1
    if os:
      actions = np.full((len(os), 1), item[SampleBatch.ACTIONS])
      rewards = np.full((len(os), 1), item[SampleBatch.REWARDS])
      dones = np.full((len(os), 1), item[SampleBatch.DONES])
      action_prob = np.full((len(os), 1), item[SampleBatch.ACTION_PROB])
      eps_id = np.full((len(os), 1), item[SampleBatch.EPS_ID])
      unroll_id = np.expand_dims(np.array(o_steps), axis=-1)
      self._frames.add_batch((np.vstack(os), actions, rewards, dones, action_prob, eps_id, unroll_id), rows)
 
  def add_batch(self, item: SampleBatchType):
    episodes, steps = item[SampleBatch.EPS_ID], item['t']
    ws = np.abs(item['weights'])
    prob = np.pow(ws, self.alpha) if self.alpha != 1 else ws
    overwritten = self._tree.add_batch(np.transpose(np.vstack([episodes, steps])), prob)
    if overwritten is not None:
      for ep, st in overwritten:
        del self._episode_steps[(ep, st)]
    obs_batch = item[SampleBatch.CUR_OBS]
    for episode, step, obs in zip(episodes, steps, np.split(obs_batch, obs_batch.shape[0], axis=0)):
      cnt = obs.shape[-1] // self.channels_per_frame
      #n = len(obs.shape)
      #perm = [n] + list(range(n))
      #simple_obs = np.transpose(np.reshape(obs, obs.shape[:-1] + (self.channels_per_frame, cnt)), axes=perm)
      #rows = list(range(self._frame_idx, self._frame_idx + cnt))
      simple_obs = [
        np.take(obs, list(range(i * 4, i * 4 + 4)), axis=-1)
        for i in range(cnt)
      ]
      os = []
      rows = []
      for i, o in enumerate(simple_obs):
        o_step = step - (cnt - i - 1)
        if (episode, o_step) not in self._episode_steps:
          os.append(o)
          rows.append(self._frame_idx % self._frames.length)
          self._episode_steps[(episode, o_step)] = self._frame_idx
          self._frame_idx += 1
      if os:
        self._frames.add_batch(np.vstack(os), rows)
    self._batches_added += 1
  
  def sample(self, num_items: int, beta: float, trajectory_len: int = None) -> SampleBatchType:
    if trajectory_len is None:
      trajectory_len = self.frames_per_obs
    episode_steps = self._tree.sample_batch(num_items)
    frames = []
    actions = []
    rewards = []
    dones = []
    action_probs = []
    eps_ids = []
    unroll_ids = []
    for episode, step in episode_steps:
      indices = [
        self._episode_steps[(episode, s)]
        for s in range(step, step - trajectory_len, -1)
      ]
      obs, a, r, d, ap, eid, uid = self._frames.select(indices)
      dims = len(obs.shape)
      if self.channels_per_frame > 1:
        axes = list(range(1, dims - 1)) + [0, dims - 1]
        obs = np.transpose(obs, axes=axes)
        obs = np.reshape(obs, obs.shape[:-2] + (obs.shape[-2] * obs.shape[-1],))
      else:
        axes = list(range(1, dims)) + [0]
        obs = np.transpose(obs, axes=axes)
      frames.append(np.expand_dims(obs, axis=0))
      actions.append(a[0][0])
      rewards.append(r[0][0])
      dones.append(d[0][0])
      action_probs.append(ap[0][0])
      eps_ids.append(eid[0][0])
      unroll_ids.append(uid[0][0])

    # TODO: This doesn't work if frames are general structures (not tensors)
    sample = SampleBatch(**{
      SampleBatch.CUR_OBS: np.vstack(frames),
      SampleBatch.ACTIONS: actions,
      SampleBatch.REWARDS: rewards,
      SampleBatch.DONES: dones,
      SampleBatch.ACTION_PROB: action_probs,
      SampleBatch.EPS_ID: eps_ids,
      SampleBatch.UNROLL_ID: unroll_ids,
    })
    self._batches_sampled += 1
    return sample

  def update_priorities(self, idxes, priorities):
    assert len(idxes) == len(priorities)
    for idx, priority in zip(idxes, priorities):
      assert priority > 0
      self._tree.update(idx, priority**self._alpha)

  def stats(self, debug=False):
    data = {
      "frame_count": self._frame_idx,
      "sampled_count": self._batches_sampled,
      "est_size_bytes": self._frames.size_bytes()[1],
      "num_entries": self._frames.length,
    }
    return data


# Visible for testing.
_local_replay_buffer = None


class LocalReplayBuffer(ParallelIteratorWorker):
    """A replay buffer shard.
    Ray actors are single-threaded, so for scalability multiple replay actors
    may be created to increase parallelism."""

    def __init__(self,
                 tensor_spec,
                 num_shards,
                 learning_starts,
                 buffer_size,
                 replay_batch_size,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta=0.4,
                 prioritized_replay_eps=1e-6,
                 replay_mode="independent",
                 replay_sequence_length=1):
        self.replay_starts = learning_starts // num_shards
        self.buffer_size = buffer_size // num_shards
        self.replay_batch_size = replay_batch_size
        self.prioritized_replay_beta = prioritized_replay_beta
        self.prioritized_replay_eps = prioritized_replay_eps
        self.replay_mode = replay_mode
        self.replay_sequence_length = replay_sequence_length

        if replay_sequence_length > 1:
            self.replay_batch_size = int(
                max(1, replay_batch_size // replay_sequence_length))
            logger.info(
                "Since replay_sequence_length={} and replay_batch_size={}, "
                "we will replay {} sequences at a time.".format(
                    replay_sequence_length, replay_batch_size,
                    self.replay_batch_size))

        if replay_mode not in ["lockstep", "independent"]:
            raise ValueError("Unsupported replay mode: {}".format(replay_mode))

        def gen_replay():
            while True:
                yield self.replay()

        ParallelIteratorWorker.__init__(self, gen_replay, False)

        def new_buffer():
            return PrioritizedReplayBuffer(
                self.buffer_size, tensor_spec, alpha=prioritized_replay_alpha)

        self.replay_buffers = collections.defaultdict(new_buffer)

        # Metrics
        self.add_batch_timer = TimerStat()
        self.replay_timer = TimerStat()
        self.update_priorities_timer = TimerStat()
        self.num_added = 0

        # Make externally accessible for testing.
        global _local_replay_buffer
        _local_replay_buffer = self
        # If set, return this instead of the usual data for testing.
        self._fake_batch = None

    @staticmethod
    def get_instance_for_testing():
        global _local_replay_buffer
        return _local_replay_buffer

    def get_host(self):
        return platform.node()

    def get_default_buffer(self):
        return self.replay_buffers[DEFAULT_POLICY_ID]

    def add_batch(self, batch):
        # Make a copy so the replay buffer doesn't pin plasma memory.
        batch = batch.copy()
        # Handle everything as if multiagent
        if isinstance(batch, SampleBatch):
            batch = MultiAgentBatch({DEFAULT_POLICY_ID: batch}, batch.count)
        with self.add_batch_timer:
            if self.replay_mode == "lockstep":
                # Note that prioritization is not supported in this mode.
                for s in batch.timeslices(self.replay_sequence_length):
                    self.replay_buffers[_ALL_POLICIES].add(s, weight=None)
            else:
                for policy_id, b in batch.policy_batches.items():
                    for s in b.timeslices(self.replay_sequence_length):
                        if "weights" in s:
                            weight = np.mean(s["weights"])
                        else:
                            weight = None
                        self.replay_buffers[policy_id].add(s, weight=weight)
        self.num_added += batch.count

    def replay(self):
        if self._fake_batch:
            fake_batch = SampleBatch(self._fake_batch)
            return MultiAgentBatch({
                DEFAULT_POLICY_ID: fake_batch
            }, fake_batch.count)

        if self.num_added < self.replay_starts:
            return None

        with self.replay_timer:
            if self.replay_mode == "lockstep":
                return self.replay_buffers[_ALL_POLICIES].sample(
                    self.replay_batch_size, beta=self.prioritized_replay_beta)
            else:
                samples = {}
                for policy_id, replay_buffer in self.replay_buffers.items():
                    samples[policy_id] = replay_buffer.sample(
                        self.replay_batch_size,
                        beta=self.prioritized_replay_beta)
                return MultiAgentBatch(samples, self.replay_batch_size)

    def update_priorities(self, prio_dict):
        with self.update_priorities_timer:
            for policy_id, (batch_indexes, td_errors) in prio_dict.items():
                new_priorities = (
                    np.abs(td_errors) + self.prioritized_replay_eps)
                self.replay_buffers[policy_id].update_priorities(
                    batch_indexes, new_priorities)

    def stats(self, debug=False):
        stat = {
            "add_batch_time_ms": round(1000 * self.add_batch_timer.mean, 3),
            "replay_time_ms": round(1000 * self.replay_timer.mean, 3),
            "update_priorities_time_ms": round(
                1000 * self.update_priorities_timer.mean, 3),
        }
        for policy_id, replay_buffer in self.replay_buffers.items():
            stat.update({
                "policy_{}".format(policy_id): replay_buffer.stats(debug=debug)
            })
        return stat

ReplayActor = ray.remote(num_cpus=0)(LocalReplayBuffer)