import collections
from collections import namedtuple
import logging
import numpy as np
import platform
import tensorflow as tf
from typing import List

import ray
from ray.util.iter import ParallelIteratorWorker
from ray.util.timer import _Timer as TimerStat

from muzero.sample_batch import SampleBatch, DEFAULT_POLICY_ID
from muzero.structure_list import NPStructureList


PRIO_WEIGHTS = 'weights'
PRIORITIES = 'priorities'
MIN_ALLOWED_PRIORITY = 0.1

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

  def update(self, id_, new_weight):
    node = self._get_node_for_id(id_)
    old_weight = self._id_tree[self._node_to_index(node)]
    delta = new_weight - old_weight

    self._total += delta
    self._id_tree[self._node_to_index(node)] += delta
    while self._parent(node):
      node = self._parent(node)
      self._id_tree[self._node_to_index(node)] += delta

    # If the weight we are updating used to be min or max, then find new min or max
    if self._min == old_weight:
      self._min = new_weight
      for i in range(min(self._next_element_id, self.capacity)):
        # TODO: This hack shouldn't be necessary
        if self._id_tree[self._leaf_offset + i] > 0:
          self._min = min(self._min, self._id_tree[self._leaf_offset + i])
    else:
      self._min = min(self._min, new_weight)
    if self._max == old_weight:
      self._max = new_weight
      for i in range(min(self._next_element_id, self.capacity)):
        self._max = max(self._max, self._id_tree[self._leaf_offset + i])
    else:
      self._max = max(self._max, new_weight)
    
  def contains(self, a, b):
    return (a, b) in self._pairs


class ReplayBuffer(tf.Module):

  def __init__(self, capacity, array_spec, scope='replay_buffer', name='ReplayBuffer'):
    super(ReplayBuffer, self).__init__(name=name)
    self._array_spec = array_spec
    self._capacity = capacity
    self._scope = scope

    if isinstance(array_spec, tuple):
      self._fields = [spec.name for spec in array_spec]
    else:
      self._fields = None

  def add(self, item: SampleBatch, weight: float):
    pass

  def sample(self, num_items: int):
    pass


class PrioritizedReplayBuffer(ReplayBuffer):

  def __init__(self, capacity, array_spec,
               scope='prioritized_replay_buffer', alpha=1, beta=1,
               frames_per_obs=32):
    super(PrioritizedReplayBuffer, self).__init__(
      capacity,
      array_spec,
      scope,
      'PrioritizedReplayBuffer')

    # Check that the spec has the minimal required fields.
    names = [spec.name for spec in array_spec]
    assert SampleBatch.EPS_ID in names
    assert SampleBatch.CUR_OBS in names

    self.alpha = alpha
    self.beta = beta
    self._step_idx = 0
    # Tree of (episode, frame) pairs with priorities
    self._tree = IntPairSumTree(capacity)
    # Buffer of frames
    self._steps = NPStructureList(capacity, array_spec, scope='frame_buffer')
    # (episode, step) => (index of step in self._steps, id of step in self._tree)
    self._episode_steps = {}
    self._batches_added = 0
    self._batches_sampled = 0
    self.channels_per_frame = 1
    for i, field in enumerate(self._fields):
      if field == SampleBatch.CUR_OBS:
        self.channels_per_frame = array_spec[i].shape[-1]
    self.frames_per_obs = frames_per_obs

  def add(self, item: SampleBatch, p: float):
    """
    The item is added with priority `p**self.alpha`.
    """
    assert len(item[SampleBatch.EPS_ID]) == 1
    assert len(item['t']) == 1
    # Silently ignore
    if 0 <= p < MIN_ALLOWED_PRIORITY:
      p = MIN_ALLOWED_PRIORITY
    #print('sample keys:', sorted([(field, item[field].shape) for field in item.data]))
    episode, step = item[SampleBatch.EPS_ID][0], item["t"][0]
    if p == -1:
      p = self._tree.max if self._tree.max else 1
    else:
      assert p >= 0
    priority = np.pow(p, self.alpha) if self.alpha != 1 else p
    id_ = self._tree.next_id
    overwritten = self._tree.add(episode, step, priority)
    if overwritten is not None:
      eps, st = overwritten
      if (eps, st) in self._episode_steps:
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
      zeros = np.full(zeros_shape, 0, dtype=self._array_spec[i].dtype)
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
    data.update({
      field: field_val(item, field, step_count, i)
      for i, field in enumerate(self._fields)
      if field not in exclude_fields
    })
    batch = tuple(data[field] for field in self._fields)
    self._steps.add_batch(batch, rows)
    # Not sure if this is necessary
    for row in rows:
      del row
    del batch
    del data
    return priority

  def sample(self, num_items: int, trajectory_len: int = None) -> SampleBatch:
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
    steps = [step for ep, step in episode_steps]
    #raise Exception(f"""
    #  print('get by idnex', {self._tree.get_by_index(tree_indices[0])})
    #  print('tree total', {self._tree.total})
    #  print('tree size', {self._tree.size})
    #  print('tree min', {self._tree.min})
    #  """)
    max_weight = ((self._tree.min / self._tree.total) * self._tree.size)**(-self.beta)
    weights = [
      ((self._tree.get_by_index(idx) / self._tree.total) * self._tree.size)**(-self.beta) / max_weight
      for idx in tree_indices
    ]
    #idx = tree_indices[0]
    #print('priority', self._tree.get_by_index(idx), 'total', self._tree.total, 'size', self._tree.size, 'beta', self.beta, 'max weight', max_weight, 'min', self._tree.min, 'weight', weights[0])
    data = { field: [] for field in self._fields }
    for episode, step in episode_steps:
      indices = [
        self._episode_steps[(episode, max(s, 0))][0]
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
          data[field].append(obs)
        elif len(item[i][0].shape) == 1 and item[i][0].shape[0] == 1:
          data[field].append(item[i][0][0])
        else:
          data[field].append(item[i][0])

    #print(data['rollout_policies'][0].shape)
    #print(len(data['rollout_policies']))
    #import time
    #time.sleep(3)
    data_np = {
      field: np.array(data[field]) if isinstance(data[field][0], np.ndarray) else np.array(data[field])
      for field in self._fields
    }
    #print('data_np', [(field, data_np[field].shape) for field in data_np])
    # TODO: This doesn't work if frames are general structures (not tensors)
    sample = SampleBatch(**data_np)
    sample['batch_indexes'] = np.array(batch_indexes)
    sample['t'] = np.array(steps)
    sample[PRIO_WEIGHTS] = weights
    self._batches_sampled += 1
    return sample

  def update_priorities(self, idxes, priorities):
    #print('next id: ', self._tree.next_id, 'first priority:', priorities[0])
    assert len(idxes) == len(priorities)
    for idx, priority in zip(idxes, priorities):
      if priority < MIN_ALLOWED_PRIORITY:
        priority = MIN_ALLOWED_PRIORITY
      self._tree.update(idx, priority**self.alpha)

  def get_priorities(self, idxes):
    return [self._tree.get_by_index(idx) for idx in idxes]

  def stats(self, debug=False):
    data = {
      "frame_count": self._step_idx,
      "sampled_count": self._batches_sampled,
      "est_size_bytes": self._steps.size_bytes()[1],
      "num_entries": self._steps.length,
      "tree_min": self._tree.min,
      "tree_max": self._tree.max,
    }
    return data


# Visible for testing.
_local_replay_buffer = None


class LocalReplayBuffer(ParallelIteratorWorker):
    """A replay buffer shard.
    Ray actors are single-threaded, so for scalability multiple replay actors
    may be created to increase parallelism."""

    def __init__(self,
                 array_spec,
                 num_shards,
                 learning_starts,
                 buffer_size,
                 replay_batch_size,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta=0.4,
                 prioritized_replay_eps=1e-6,
                 replay_mode="independent",
                 replay_sequence_length=1,
                 frames_per_obs=32):
        self.replay_starts = learning_starts // num_shards
        self.buffer_size = buffer_size // num_shards
        self.replay_batch_size = replay_batch_size
        self.prioritized_replay_alpha = prioritized_replay_alpha
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
                array_spec,
                alpha=prioritized_replay_alpha,
                beta=prioritized_replay_beta,
                frames_per_obs=frames_per_obs)

        self.replay_buffers = collections.defaultdict(new_buffer)

        # Metrics
        self.add_batch_timer = TimerStat()
        self.replay_timer = TimerStat()
        self.update_priorities_timer = TimerStat()
        self.num_added = 0
        self.num_updated = 0
        self.priority_total = 0
        self.updated_priority_total = 0

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

    def add_batch(self, batch: SampleBatch):
        # Make a copy so the replay buffer doesn't pin plasma memory.
        b = batch.copy()
        del batch
        with self.add_batch_timer:
            for s in b.timeslices(1):
                if PRIORITIES in s:
                    p = s[PRIORITIES][0]
                else:
                    p = -1  # -1 means unknown
                self.priority_total += self.replay_buffers[DEFAULT_POLICY_ID].add(s, p=p)
                self.num_added += 1
        # Not sure if this is necessary
        del b

    def replay(self):
        if self._fake_batch:
            fake_batch = SampleBatch(self._fake_batch)
            return fake_batch

        # TODO: This might break training intensity
        if self.num_added < max(self.replay_starts, self.replay_batch_size):
            return None

        with self.replay_timer:
            return self.replay_buffers[DEFAULT_POLICY_ID].sample(self.replay_batch_size)

    def update_priorities(self, prio_dict):
        with self.update_priorities_timer:
            for policy_id, (batch_indexes, ps) in prio_dict.items():
                new_priorities = np.abs(ps) + self.prioritized_replay_eps
                self.replay_buffers[policy_id].update_priorities(
                    batch_indexes, new_priorities)
                self.num_updated += new_priorities.shape[0]
                self.updated_priority_total += new_priorities.sum()

    def stats(self, debug=False):
        stat = {
            "add_batch_time_ms": round(1000 * self.add_batch_timer.mean, 3),
            "replay_time_ms": round(1000 * self.replay_timer.mean, 3),
            "update_priorities_time_ms": round(
                1000 * self.update_priorities_timer.mean, 3),
            "average_priority": self.priority_total / self.num_added,
            "update_count": self.num_updated,
            "average_updated_priority": self.updated_priority_total / self.num_updated,
        }
        for policy_id, replay_buffer in self.replay_buffers.items():
            stat.update({
                "policy_{}".format(policy_id): replay_buffer.stats(debug=debug)
            })
        # Reset count and priority stats after every call
        self.num_added = 0
        self.num_updated = 0
        self.priority_total = 0
        self.updated_priority_total = 0
        return stat

ReplayActor = ray.remote(num_cpus=0)(LocalReplayBuffer)
