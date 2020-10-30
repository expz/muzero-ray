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

PRIO_WEIGHTS = 'weights'


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
  def total(self):
    pass

  @property
  def size(self):
    pass

  @property
  def min(self):
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
      Node(self._tree_depth, (id_ + i) % self.capacity)
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
    # tree of priorities
    self._id_tree = [0] * ((1 << (self._tree_depth + 1)) - 1)
    # list of integer pairs. pair at index i corresponds to leaf i of tree.
    self._data_list = np.full((capacity, 2), -1, dtype=np.int32)
    # set of integer pairs
    self._pairs = set([])
    self._next_element_id = 0
    self._total = 0
    self._min = float('inf')
    self._max = 0

  @property
  def next_id(self):
    return self._next_element_id

  @property
  def total(self):
    return self._total

  @property
  def size(self):
    return min(self._next_element_id, self.capacity)

  @property
  def min(self):
    return self._min

  @property
  def max(self):
    return self._max

  def add(self, a, b, prob):
    node = self._get_node_for_id(self._next_element_id)
    #s = 'node num:' + str(node.num) + ' next id:' + str(self._next_element_id) + ' index:' + str(self._node_to_index(node)) + '\n'
    id_ = node.num
    overwritten = None
    if self._next_element_id >= self.capacity:
      overwritten = self._data_list[id_, :]
      self._pairs.remove(tuple(overwritten))
    self._data_list[id_, :] = [a, b]
    self._pairs.add((a, b))
    k = self._node_to_index(node)
    delta_p = prob - self._id_tree[k]
    #s += 'k: ' + str(k) + ' delta_p: ' + str(delta_p) + '\n'
    self._id_tree[k] = prob
    node = self._parent(node)
    while node is not None:
    #  s += 'node: ' + str((node.depth, node.num)) + ' index: ' + str(self._node_to_index(node)) + ' | '
      self._id_tree[self._node_to_index(node)] += delta_p
      node = self._parent(node)
    self._total += delta_p
    self._min = min(self._min, prob)
    self._max = max(self._max, prob)
    #if not all([self._id_tree[i] == 0 for i in range(k + 1, k + 500)]):
    #  raise Exception(s)
    self._next_element_id += 1
    return overwritten

  def add_batch(self, items, probs):
    batch_size = items.shape[0]
    leaves = self._get_nodes_for_id(self._next_element_id, batch_size)
    indices = [node.num for node in leaves]
    for node, p in zip(leaves, probs):
      k = self._node_to_index(node)
      delta_p = p - self._id_tree[k]
      self._total += delta_p
      self._min = min(self._min, p)
      self._max = max(self._max, p)
      self._id_tree[k] = p
      node = self._parent(node)
      while node is not None:
        self._id_tree[self._node_to_index(node)] += delta_p
        node = self._parent(node)
    overwritten = None
    if self._next_element_id >= self.capacity:
      overwritten = self._data_list[indices, :]
      for pair in np.split(overwritten, overwritten.shape[0], axis=0):
        self._pairs.add(tuple(pair))
    self._data_list[indices, :] = items
    for pair in np.split(items, items.shape[0], axis=0):
      self._pairs.add(tuple(pair))
    self._next_element_id += batch_size
    return overwritten
 
  def _sample_probs(self, batch_size):
    max_val = self._id_tree[0]
    return np.random.uniform(low=0.0, high=max_val, size=[batch_size])

  def sample_batch(self, batch_size):
    if self._next_element_id == 0:
      return None, None
    #print('next element id:', self._next_element_id, 'capacity:', self.capacity, 'last mass:', self._id_tree[self._next_element_id - 1], 'last data:', self._data_list[self._next_element_id - 1])
    indices = [0] * batch_size
    ps = self._sample_probs(batch_size)
    #s = ''
    #bad = False
    for i in range(batch_size):
      node = self._root()
      p = ps[i]
      while not self._is_leaf(node):
      #  s += 'node:' + str((node.depth, node.num)) + 'p:' + str(p) + '\n'
        left = self._left_child(node)
        right = self._right_child(node)
      #  s += 'l child:' + str((left.depth, left.num)) + 'r child:' + str((right.depth, right.num)) + '\n'
        left_val = self._id_tree[self._node_to_index(left)]
      #  s += 'l index:' + str(self._node_to_index(left)) + 'l val:' + str(left_val) + '\n'
        if p <= left_val:
      #    s += 'went left\n'
          node = left
        else:
      #    s += 'went right\n'
          node = right
          p -= left_val
      #if node.num >= self._next_element_id:
      #  s += '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
      #  x = self._node_to_index(node)
      #  s += str(self._id_tree[max(0, x - 700):x]) + '\n'
      #  bad = True
      indices[i] = node.num
      #s += 'p:' + str(p) + 'id:' + str(node.num) + 'data:' + str(self._data_list[node.num]) + '\n'
      next_node = self._get_node_for_id(self._next_element_id)
      next_index = self._node_to_index(next_node)
      #s += 'next id: ' + str(self._next_element_id) + ' next node: ' + str((next_node.depth, next_node.num)) + ' next index: ' + str(next_index) + '\n'
      #s += str(self._id_tree[next_index:next_index + 100]) + '\n'
      #s += str(self._data_list[self._next_element_id:self._next_element_id + 100]) + '\n'
      #s += '=================================\n'
      #if bad:
      #  raise Exception(s)
    return self._data_list[indices], indices

  def get_by_index(self, idx):
    return self._id_tree[self._node_to_index(self._get_node_for_id(idx))]

  def update(self, id_, weight):
    node = self._get_node_for_id(id_)
    #print('update ', (node.depth, node.num), ' id:', id_)
    delta = weight - self._id_tree[self._node_to_index(node)]
    self._total += delta
    self._min = min(self._min, weight)
    self._max = max(self._max, weight)
    self._id_tree[self._node_to_index(node)] += delta
    while self._parent(node):
      node = self._parent(node)
      self._id_tree[self._node_to_index(node)] += delta
    
  def contains(self, a, b):
    return (a, b) in self._pairs


class ReplayBuffer(tf.Module):

  def __init__(self, capacity, tensor_spec, scope='replay_buffer', name='ReplayBuffer'):
    super(ReplayBuffer, self).__init__(name=name)
    self._tensor_spec = tensor_spec
    self._capacity = capacity
    self._scope = scope

    if isinstance(tensor_spec, tuple):
      self._fields = [spec.name for spec in tensor_spec]
    else:
      self._fields = None

  def add(self, item: SampleBatchType, weight: float):
    pass

  def sample(self, num_items: int):
    pass


class PrioritizedReplayBuffer(ReplayBuffer):

  def __init__(self, capacity, tensor_spec,
               scope='prioritized_replay_buffer', alpha=1, beta=1,
               frames_per_obs=32):
    super(PrioritizedReplayBuffer, self).__init__(
      capacity,
      tensor_spec,
      scope,
      'PrioritizedReplayBuffer')
    self.alpha = alpha
    self.beta = beta
    self._step_idx = 0
    # Tree of (episode, frame) pairs with priorities
    self._tree = IntPairSumTree(capacity)
    # Buffer of frames
    self._steps = StructureList(capacity, tensor_spec, scope='frame_buffer')
    # (episode, step) => (index of step in self._steps, id of step in self._tree)
    self._episode_steps = {}
    self._batches_added = 0
    self._batches_sampled = 0
    self.channels_per_frame = 1
    for i, field in enumerate(self._fields):
      if field == SampleBatch.CUR_OBS:
        self.channels_per_frame = tensor_spec[i].shape[-1]
    self.frames_per_obs = frames_per_obs

  def add(self, item: SampleBatchType, weight: float):
    assert len(item[SampleBatch.EPS_ID]) == 1
    assert len(item['t']) == 1
    #print('sample keys:', sorted([(field, item[field].shape) for field in item.data]))
    episode, step = item[SampleBatch.EPS_ID][0], item['t'][0]
    if weight == -1:
      weight = self._tree.max if self._tree.max else 1
    else:
      assert weight >= 0
    prob = np.pow(weight, self.alpha) if self.alpha != 1 else weight
    id_ = self._tree.next_id
    overwritten = self._tree.add(episode, step, prob)
    if overwritten is not None:
      eps, st = overwritten
      del self._episode_steps[(eps, st)]
    obs = np.expand_dims(item[SampleBatch.CUR_OBS][0], axis=0)
    cnt = obs.shape[-1] // self.channels_per_frame
    simple_obs = [
      np.take(obs, list(range(i * self.channels_per_frame, (i + 1) * self.channels_per_frame)), axis=-1)
      for i in range(cnt)
    ]
    os = []
    o_steps = []
    rows = []
    for i, o in enumerate(simple_obs):
      o_step = step - (cnt - i - 1)
      frame_idx, _ = self._episode_steps.get((episode, o_step), (None, None))
      # don't re-save secondary frames that were saved from a previous call
      if frame_idx is None or o_step == step:
        os.append(o)
        o_steps.append(o_step)
        # TODO: Get rid of rows argument of list add_batch and have it
        # keep track of next id so we don't need self._step_idx
        rows.append(self._step_idx % self._steps.length)
        if o_step == step:
          if (episode, o_step) in self._episode_steps:
            logging.warn(
              f'Episode {episode} step {o_step} is being added to the replay '
              'buffer a second time. It will overwrite and have a priority '
              'which is the sum of the previous and current priorities.')
          self._episode_steps[(episode, o_step)] = (self._step_idx % self._steps.length, id_)
        else:
          self._episode_steps[(episode, o_step)] = (self._step_idx % self._steps.length, None)
        self._step_idx += 1

    def field_val(item, field, step_count, i):
      """
      Store zeros for steps that are included in the current step
      data but are not the primary step (since we don't know anything
      about them except their observation).

      This has a hack to match rllib behavior whereby tensors of
      shape (1,) are expanded to (1, 1) and then concatenated to (32, 1)
      but tensors of shape (1, 5) are kept the same and concatenated to
      (32, 5).
      """
      if len(item[field].shape) > 1 and item[field].shape[0] == 1:
        zeros_shape = (step_count - 1,) + tuple(item[field].shape)[1:]
        data = item[field]
      else:
        zeros_shape = (step_count - 1,) + tuple(item[field].shape)
        data = np.expand_dims(item[field], axis=0)
      zeros = np.full(zeros_shape, 0, dtype=self._tensor_spec[i].dtype.as_numpy_dtype)
      return np.concatenate([zeros, data])
 
    step_count = len(os)
    assert step_count > 0
    data = {
      SampleBatch.CUR_OBS: np.vstack(os),
      SampleBatch.EPS_ID: np.full((step_count, 1), item[SampleBatch.EPS_ID]),
      SampleBatch.UNROLL_ID: np.expand_dims(np.array(o_steps), axis=-1),
    }
    exclude_fields = set([
      SampleBatch.CUR_OBS, SampleBatch.EPS_ID, SampleBatch.UNROLL_ID,
    ])
    #print('_fields', [(field, type(field)) for field in self._fields])
    data.update({
      field: field_val(item, field, step_count, i)
      for i, field in enumerate(self._fields)
      if field not in exclude_fields
    })
    batch = tuple(data[field] for field in self._fields)
    #print('batch', [(field, batch[i].shape) for i, field in enumerate(self._fields)])
    self._steps.add_batch(batch, rows)

  def sample(self, num_items: int, trajectory_len: int = None) -> SampleBatchType:
    if trajectory_len is None:
      trajectory_len = self.frames_per_obs
    episode_steps, tree_indices = self._tree.sample_batch(num_items)
    if episode_steps is None:
      raise Exception('Tried to sample from empty replay buffer')
    # indexes are ids in integer pair tree
    batch_indexes = [
      self._episode_steps[(ep, step)][1]
      for ep, step in episode_steps
    ]
    #raise Exception(f"""
    #  print('get by idnex', {self._tree.get_by_index(tree_indices[0])})
    #  print('tree total', {self._tree.total})
    #  print('tree size', {self._tree.size})
    #  print('tree min', {self._tree.min})
    #  """)
    max_weight = (self._tree.min * self._tree.size)**(-self.beta)
    weights = [
      ((self._tree.get_by_index(idx) / self._tree.total) * self._tree.size)**(-self.beta) / max_weight
      for idx in tree_indices
    ]
    data = { field: [] for field in self._fields }
    for episode, step in episode_steps:
      indices = [
        self._episode_steps[(episode, s)][0]
        for s in range(step, step - trajectory_len, -1)
      ]
      item = self._steps.select(indices)
      obs = item[self._fields.index(SampleBatch.CUR_OBS)]
      dims = len(obs.shape)
      if self.channels_per_frame > 1:
        axes = list(range(1, dims - 1)) + [0, dims - 1]
        obs = np.transpose(obs, axes=axes)
        obs = np.reshape(obs, obs.shape[:-2] + (obs.shape[-2] * obs.shape[-1],))
      else:
        axes = list(range(1, dims)) + [0]
        obs = np.transpose(obs, axes=axes)
      for i, field in enumerate(self._fields):
        if field == SampleBatch.CUR_OBS:
          data[field].append(np.expand_dims(obs, axis=0))
        elif len(item[i][0].shape) == 1 and item[i][0].shape[0] == 1:
          data[field].append(item[i][0][0])
        else:
          data[field].append(item[i][0])

    data_np = {
      field: np.vstack(data[field]) if isinstance(data[field][0], np.ndarray) else np.array(data[field])
      for field in self._fields
    }
    #print('data_np', [(field, data_np[field].shape) for field in data_np])
    # TODO: This doesn't work if frames are general structures (not tensors)
    sample = SampleBatch(**data_np)
    sample['batch_indexes'] = batch_indexes
    sample[PRIO_WEIGHTS] = weights
    self._batches_sampled += 1
    return sample

  def update_priorities(self, idxes, priorities):
    print('next id: ', self._tree.next_id, 'first priority:', priorities[0])
    assert len(idxes) == len(priorities)
    for idx, priority in zip(idxes, priorities):
      assert priority > 0
      self._tree.update(idx, priority**self.alpha)

  def stats(self, debug=False):
    data = {
      "frame_count": self._step_idx,
      "sampled_count": self._batches_sampled,
      "est_size_bytes": self._steps.size_bytes()[1],
      "num_entries": self._steps.length,
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
                self.buffer_size,
                tensor_spec,
                alpha=prioritized_replay_alpha,
                beta=prioritized_replay_beta)

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
                        if PRIO_WEIGHTS in s:
                            weight = np.mean(s[PRIO_WEIGHTS])
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

        # TODO: This might break training intensity
        if self.num_added < max(self.replay_starts, self.replay_batch_size):
            return None

        with self.replay_timer:
            if self.replay_mode == "lockstep":
                return self.replay_buffers[_ALL_POLICIES].sample(self.replay_batch_size)
            else:
                samples = {}
                for policy_id, replay_buffer in self.replay_buffers.items():
                    samples[policy_id] = replay_buffer.sample(self.replay_batch_size)
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