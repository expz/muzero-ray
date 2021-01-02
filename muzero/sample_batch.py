"""
From https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/utils/compression.py
and https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/utils/memory.py
and https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/policy/sample_batch.py
"""
from __future__ import annotations

import base64
import collections
import itertools
import lz4
import numpy as np
import pickle
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Union


TensorType = Any

# Default policy id for single agent environments
DEFAULT_POLICY_ID = "default_policy"

# TODO(ekl) reuse the other id def once we fix imports
PolicyID = Any

def pack(data):
    data = pickle.dumps(data)
    data = lz4.frame.compress(data)
    # TODO(ekl) we shouldn't need to base64 encode this data, but this
    # seems to not survive a transfer through the object store if we don't.
    data = base64.b64encode(data).decode("ascii")
    return data

def unpack(data):
    data = base64.b64decode(data)
    data = lz4.frame.decompress(data)
    data = pickle.loads(data)
    return data

def is_compressed(data):
    return isinstance(data, bytes) or isinstance(data, str)

def aligned_array(size, dtype, align=64):
    """Returns an array of a given size that is 64-byte aligned.
    The returned array can be efficiently copied into GPU memory by TensorFlow.
    """

    n = size * dtype.itemsize
    empty = np.empty(n + (align - 1), dtype=np.uint8)
    data_align = empty.ctypes.data % align
    offset = 0 if data_align == 0 else (align - data_align)
    if n == 0:
        # stop np from optimising out empty slice reference
        output = empty[offset:offset + 1][0:0].view(dtype)
    else:
        output = empty[offset:offset + n].view(dtype)

    assert len(output) == size, len(output)
    assert output.ctypes.data % align == 0, output.ctypes.data
    return output

def concat_aligned(items, time_major=None):
    """Concatenate arrays, ensuring the output is 64-byte aligned.
    We only align float arrays; other arrays are concatenated as normal.
    This should be used instead of np.concatenate() to improve performance
    when the output array is likely to be fed into TensorFlow.
    Args:
        items (List(np.ndarray)): The list of items to concatenate and align.
        time_major (bool): Whether the data in items is time-major, in which
            case, we will concatenate along axis=1.
    Returns:
        np.ndarray: The concat'd and aligned array.
    """

    if len(items) == 0:
        return []
    elif len(items) == 1:
        # we assume the input is aligned. In any case, it doesn't help
        # performance to force align it since that incurs a needless copy.
        return items[0]
    elif (isinstance(items[0], np.ndarray)
          and items[0].dtype in [np.float32, np.float64, np.uint8]):
        dtype = items[0].dtype
        flat = aligned_array(sum(s.size for s in items), dtype)
        if time_major is not None:
            if time_major is True:
                batch_dim = sum(s.shape[1] for s in items)
                new_shape = (
                    items[0].shape[0],
                    batch_dim,
                ) + items[0].shape[2:]
            else:
                batch_dim = sum(s.shape[0] for s in items)
                new_shape = (
                    batch_dim,
                    items[0].shape[1],
                ) + items[0].shape[2:]
        else:
            batch_dim = sum(s.shape[0] for s in items)
            new_shape = (batch_dim, ) + items[0].shape[1:]
        output = flat.reshape(new_shape)
        assert output.ctypes.data % 64 == 0, output.ctypes.data
        np.concatenate(items, out=output, axis=1 if time_major else 0)
        return output
    else:
        return np.concatenate(items, axis=1 if time_major else 0)


class SampleBatch:
    """Wrapper around a dictionary with string keys and array-like values.
    For example, {"obs": [1, 2, 3], "reward": [0, -1, 1]} is a batch of three
    samples, each with an "obs" and "reward" attribute.
    """

    # Outputs from interacting with the environment
    CUR_OBS = "obs"
    NEXT_OBS = "new_obs"
    ACTIONS = "actions"
    REWARDS = "rewards"
    PREV_ACTIONS = "prev_actions"
    PREV_REWARDS = "prev_rewards"
    DONES = "dones"
    INFOS = "infos"

    # Extra action fetches keys.
    ACTION_DIST_INPUTS = "action_dist_inputs"
    ACTION_PROB = "action_prob"
    ACTION_LOGP = "action_logp"

    # Uniquely identifies an episode
    EPS_ID = "eps_id"

    # Uniquely identifies a sample batch. This is important to distinguish RNN
    # sequences from the same episode when multiple sample batches are
    # concatenated (fusing sequences across batches can be unsafe).
    UNROLL_ID = "unroll_id"

    # Uniquely identifies an agent within an episode
    AGENT_INDEX = "agent_index"

    # Value function predictions emitted by the behaviour policy
    VF_PREDS = "vf_preds"

    def __init__(self, *args, **kwargs):
        """Constructs a sample batch (same params as dict constructor)."""

        self._initial_inputs = kwargs.pop("_initial_inputs", {})

        self.data = dict(*args, **kwargs)
        lengths = []
        for k, v in self.data.copy().items():
            assert isinstance(k, str), self
            lengths.append(len(v))
            self.data[k] = np.array(v, copy=False)
        if not lengths:
            raise ValueError("Empty sample batch")
        assert len(set(lengths)) == 1, ("data columns must be same length",
                                        self.data, lengths)
        self.count = lengths[0]

    @staticmethod
    def concat_samples(samples: List[Dict[str, TensorType]]) -> \
            "SampleBatch":
        """Concatenates n data dicts or MultiAgentBatches.
        Args:
            samples (List[Dict[TensorType]]]): List of dicts of data (numpy).
        Returns:
            Union[SampleBatch, MultiAgentBatch]: A new (compressed)
                SampleBatch or MultiAgentBatch.
        """
        out = {}
        samples = [s for s in samples if s.count > 0]
        for k in samples[0].keys():
            out[k] = concat_aligned([s[k] for s in samples])
        return SampleBatch(out)

    def concat(self, other: "SampleBatch") -> "SampleBatch":
        """Returns a new SampleBatch with each data column concatenated.
        Args:
            other (SampleBatch): The other SampleBatch object to concat to this
                one.
        Returns:
            SampleBatch: The new SampleBatch, resulting from concating `other`
                to `self`.
        Examples:
            >>> b1 = SampleBatch({"a": [1, 2]})
            >>> b2 = SampleBatch({"a": [3, 4, 5]})
            >>> print(b1.concat(b2))
            {"a": [1, 2, 3, 4, 5]}
        """

        if self.keys() != other.keys():
            raise ValueError(
                "SampleBatches to concat must have same columns! {} vs {}".
                format(list(self.keys()), list(other.keys())))
        out = {}
        for k in self.keys():
            out[k] = concat_aligned([self[k], other[k]])
        return SampleBatch(out)

    def copy(self) -> "SampleBatch":
        """Creates a (deep) copy of this SampleBatch and returns it.
        Returns:
            SampleBatch: A (deep) copy of this SampleBatch object.
        """
        return SampleBatch(
            {k: np.array(v, copy=True)
             for (k, v) in self.data.items()})

    def rows(self) -> Dict[str, TensorType]:
        """Returns an iterator over data rows, i.e. dicts with column values.
        Yields:
            Dict[str, TensorType]: The column values of the row in this
                iteration.
        Examples:
            >>> batch = SampleBatch({"a": [1, 2, 3], "b": [4, 5, 6]})
            >>> for row in batch.rows():
                   print(row)
            {"a": 1, "b": 4}
            {"a": 2, "b": 5}
            {"a": 3, "b": 6}
        """

        for i in range(self.count):
            row = {}
            for k in self.keys():
                row[k] = self[k][i]
            yield row

    def columns(self, keys: List[str]) -> List[any]:
        """Returns a list of the batch-data in the specified columns.
        Args:
            keys (List[str]): List of column names fo which to return the data.
        Returns:
            List[any]: The list of data items ordered by the order of column
                names in `keys`.
        Examples:
            >>> batch = SampleBatch({"a": [1], "b": [2], "c": [3]})
            >>> print(batch.columns(["a", "b"]))
            [[1], [2]]
        """

        out = []
        for k in keys:
            out.append(self[k])
        return out

    def shuffle(self) -> None:
        """Shuffles the rows of this batch in-place."""

        permutation = np.random.permutation(self.count)
        for key, val in self.items():
            self[key] = val[permutation]

    def split_by_episode(self) -> List["SampleBatch"]:
        """Splits this batch's data by `eps_id`.
        Returns:
            List[SampleBatch]: List of batches, one per distinct episode.
        """

        slices = []
        cur_eps_id = self.data["eps_id"][0]
        offset = 0
        for i in range(self.count):
            next_eps_id = self.data["eps_id"][i]
            if next_eps_id != cur_eps_id:
                slices.append(self.slice(offset, i))
                offset = i
                cur_eps_id = next_eps_id
        slices.append(self.slice(offset, self.count))
        for s in slices:
            slen = len(set(s["eps_id"]))
            assert slen == 1, (s, slen)
        assert sum(s.count for s in slices) == self.count, (slices, self.count)
        return slices

    def slice(self, start: int, end: int) -> "SampleBatch":
        """Returns a slice of the row data of this batch (w/o copying).
        Args:
            start (int): Starting index.
            end (int): Ending index.
        Returns:
            SampleBatch: A new SampleBatch, which has a slice of this batch's
                data.
        """

        return SampleBatch({k: v[start:end] for k, v in self.data.items()})

    def timeslices(self, k: int) -> List["SampleBatch"]:
        """Returns SampleBatches, each one representing a k-slice of this one.
        Will start from timestep 0 and produce slices of size=k.
        Args:
            k (int): The size (in timesteps) of each returned SampleBatch.
        Returns:
            List[SampleBatch]: The list of (new) SampleBatches (each one of
                size k).
        """
        out = []
        i = 0
        while i < self.count:
            out.append(self.slice(i, i + k))
            i += k
        return out

    def keys(self) -> Iterable[str]:
        """
        Returns:
            Iterable[str]: The keys() iterable over `self.data`.
        """
        return self.data.keys()

    def items(self) -> Iterable[TensorType]:
        """
        Returns:
            Iterable[TensorType]: The values() iterable over `self.data`.
        """
        return self.data.items()

    def get(self, key: str) -> Optional[TensorType]:
        """Returns one column (by key) from the data or None if key not found.
        Args:
            key (str): The key (column name) to return.
        Returns:
            Optional[TensorType]: The data under the given key. None if key
                not found in data.
        """
        return self.data.get(key)

    def size_bytes(self) -> int:
        """
        Returns:
            int: The overall size in bytes of the data buffer (all columns).
        """
        return sum(sys.getsizeof(d) for d in self.data)

    def __getitem__(self, key: str) -> TensorType:
        """Returns one column (by key) from the data.
        Args:
            key (str): The key (column name) to return.
        Returns:
            TensorType]: The data under the given key.
        """
        return self.data[key]

    def __setitem__(self, key, item) -> None:
        """Inserts (overrides) an entire column (by key) in the data buffer.
        Args:
            key (str): The column name to set a value for.
            item (TensorType): The data to insert.
        """
        self.data[key] = item

    def compress(
            self,
            bulk: bool = False,
            columns: Set[str] = frozenset(["obs", "new_obs"])) -> None:
        """Compresses the data buffers (by column) in place.
        Args:
            bulk (bool): Whether to compress across the batch dimension (0)
                as well. If False will compress n separate list items, where n
                is the batch size.
            columns (Set[str]): The columns to compress. Default: Only
                compress the obs and new_obs columns.
        """
        for key in columns:
            if key in self.data:
                if bulk:
                    self.data[key] = pack(self.data[key])
                else:
                    self.data[key] = np.array(
                        [pack(o) for o in self.data[key]])

    def decompress_if_needed(
            self,
            columns: Set[str] = frozenset(
                ["obs", "new_obs"])) -> "SampleBatch":
        """Decompresses data buffers (per column if not compressed) in place.
        Args:
            columns (Set[str]): The columns to decompress. Default: Only
                decompress the obs and new_obs columns.
        Returns:
            SampleBatch: This very SampleBatch.
        """
        for key in columns:
            if key in self.data:
                arr = self.data[key]
                if is_compressed(arr):
                    self.data[key] = unpack(arr)
                elif len(arr) > 0 and is_compressed(arr[0]):
                    self.data[key] = np.array(
                        [unpack(o) for o in self.data[key]])
        return self

    def __str__(self):
        return "SampleBatch({})".format(str(self.data))

    def __repr__(self):
        return "SampleBatch({})".format(str(self.data))

    def __iter__(self):
        return self.data.__iter__()

    def __contains__(self, x):
        return x in self.data


def to_float_array(v: List[Any]) -> np.ndarray:
    arr = np.array(v)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)  # save some memory
    return arr


class SampleBatchBuilder:
    """Util to build a SampleBatch incrementally.
    For efficiency, SampleBatches hold values in column form (as arrays).
    However, it is useful to add data one row (dict) at a time.
    """

    _next_unroll_id = 0  # disambiguates unrolls within a single episode

    def __init__(self):
        self.buffers: Dict[str, List] = collections.defaultdict(list)
        self.count = 0

    def add_values(self, **values: Dict[str, Any]) -> None:
        """Add the given dictionary (row) of values to this batch."""

        for k, v in values.items():
            self.buffers[k].append(v)
        self.count += 1

    def add_batch(self, batch: SampleBatch) -> None:
        """Add the given batch of values to this batch."""

        for k, column in batch.items():
            self.buffers[k].extend(column)
        self.count += batch.count

    def build_and_reset(self) -> SampleBatch:
        """Returns a sample batch including all previously added values."""

        batch = SampleBatch(
            {k: to_float_array(v)
             for k, v in self.buffers.items()})
        if SampleBatch.UNROLL_ID not in batch.data:
            batch.data[SampleBatch.UNROLL_ID] = np.repeat(
                SampleBatchBuilder._next_unroll_id, batch.count)
            SampleBatchBuilder._next_unroll_id += 1
        self.buffers.clear()
        self.count = 0
        return batch


class MultiAgentBatch:
    """A batch of experiences from multiple agents in the environment.
    Attributes:
        policy_batches (Dict[PolicyID, SampleBatch]): Mapping from policy
            ids to SampleBatches of experiences.
        count (int): The number of env steps in this batch.
    """

    def __init__(self,
                 policy_batches: Dict[PolicyID, SampleBatch],
                 env_steps: int):
        """Initialize a MultiAgentBatch object.
        Args:
            policy_batches (Dict[PolicyID, SampleBatch]): Mapping from policy
                ids to SampleBatches of experiences.
            env_steps (int): The number of timesteps in the environment this
                batch contains. This will be less than the number of
                transitions this batch contains across all policies in total.
        """

        for v in policy_batches.values():
            assert isinstance(v, SampleBatch)
        self.policy_batches = policy_batches
        # Called count for uniformity with SampleBatch. Prefer to access this
        # via the env_steps() method when possible for clarity.
        self.count = env_steps

    def env_steps(self) -> int:
        """The number of env steps (there are >= 1 agent steps per env step).
        Returns:
            int: The number of environment steps contained in this batch.
        """
        return self.count

    def agent_steps(self) -> int:
        """The number of agent steps (there are >= 1 agent steps per env step).
        Returns:
            int: The number of agent steps total in this batch.
        """
        ct = 0
        for batch in self.policy_batches.values():
            ct += batch.count
        return ct

    def timeslices(self, k: int) -> List["MultiAgentBatch"]:
        """Returns k-step batches holding data for each agent at those steps.
        For examples, suppose we have agent1 observations [a1t1, a1t2, a1t3],
        for agent2, [a2t1, a2t3], and for agent3, [a3t3] only.
        Calling timeslices(1) would return three MultiAgentBatches containing
        [a1t1, a2t1], [a1t2], and [a1t3, a2t3, a3t3].
        Calling timeslices(2) would return two MultiAgentBatches containing
        [a1t1, a1t2, a2t1], and [a1t3, a2t3, a3t3].
        This method is used to implement "lockstep" replay mode. Note that this
        method does not guarantee each batch contains only data from a single
        unroll. Batches might contain data from multiple different envs.
        """

        # Build a sorted set of (eps_id, t, policy_id, data...)
        steps = []
        for policy_id, batch in self.policy_batches.items():
            for row in batch.rows():
                steps.append((row[SampleBatch.EPS_ID], row["t"],
                              row["agent_index"], policy_id, row))
        steps.sort()

        finished_slices = []
        cur_slice = collections.defaultdict(SampleBatchBuilder)
        cur_slice_size = 0

        def finish_slice():
            nonlocal cur_slice_size
            assert cur_slice_size > 0
            batch = MultiAgentBatch(
                {k: v.build_and_reset()
                 for k, v in cur_slice.items()}, cur_slice_size)
            cur_slice_size = 0
            finished_slices.append(batch)

        # For each unique env timestep.
        for _, group in itertools.groupby(steps, lambda x: x[:2]):
            # Accumulate into the current slice.
            for _, _, _, policy_id, row in group:
                cur_slice[policy_id].add_values(**row)
            cur_slice_size += 1
            # Slice has reached target number of env steps.
            if cur_slice_size >= k:
                finish_slice()
                assert cur_slice_size == 0

        if cur_slice_size > 0:
            finish_slice()

        assert len(finished_slices) > 0, finished_slices
        return finished_slices

    @staticmethod
    def wrap_as_needed(
            policy_batches: Dict[PolicyID, SampleBatch],
            env_steps: int) -> Union[SampleBatch, "MultiAgentBatch"]:
        """Returns SampleBatch or MultiAgentBatch, depending on given policies.
        Args:
            policy_batches (Dict[PolicyID, SampleBatch]): Mapping from policy
                ids to SampleBatch.
            env_steps (int): Number of env steps in the batch.
        Returns:
            Union[SampleBatch, MultiAgentBatch]: The single default policy's
                SampleBatch or a MultiAgentBatch (more than one policy).
        """
        if len(policy_batches) == 1 and DEFAULT_POLICY_ID in policy_batches:
            return policy_batches[DEFAULT_POLICY_ID]
        return MultiAgentBatch(policy_batches, env_steps)

    @staticmethod
    def concat_samples(samples: List["MultiAgentBatch"]) -> "MultiAgentBatch":
        """Concatenates a list of MultiAgentBatches into a new MultiAgentBatch.
        Args:
            samples (List[MultiAgentBatch]): List of MultiagentBatch objects
                to concatenate.
        Returns:
            MultiAgentBatch: A new MultiAgentBatch consisting of the
                concatenated inputs.
        """
        policy_batches = collections.defaultdict(list)
        env_steps = 0
        for s in samples:
            if not isinstance(s, MultiAgentBatch):
                raise ValueError(
                    "`MultiAgentBatch.concat_samples()` can only concat "
                    "MultiAgentBatch types, not {}!".format(type(s).__name__))
            for key, batch in s.policy_batches.items():
                policy_batches[key].append(batch)
            env_steps += s.env_steps()
        out = {}
        for key, batches in policy_batches.items():
            out[key] = SampleBatch.concat_samples(batches)
        return MultiAgentBatch(out, env_steps)

    def copy(self) -> "MultiAgentBatch":
        """Deep-copies self into a new MultiAgentBatch.
        Returns:
            MultiAgentBatch: The copy of self with deep-copied data.
        """
        return MultiAgentBatch(
            {k: v.copy()
             for (k, v) in self.policy_batches.items()}, self.count)

    def size_bytes(self) -> int:
        """
        Returns:
            int: The overall size in bytes of all policy batches (all columns).
        """
        return sum(b.size_bytes() for b in self.policy_batches.values())

    def compress(
            self,
            bulk: bool = False,
            columns: Set[str] = frozenset(
                ["obs", "new_obs"])) -> None:
        """Compresses each policy batch (per column) in place.
        Args:
            bulk (bool): Whether to compress across the batch dimension (0)
                as well. If False will compress n separate list items, where n
                is the batch size.
            columns (Set[str]): Set of column names to compress.
        """
        for batch in self.policy_batches.values():
            batch.compress(bulk=bulk, columns=columns)

    def decompress_if_needed(
            self,
            columns: Set[str] = frozenset(
                ["obs", "new_obs"])) -> "MultiAgentBatch":
        """Decompresses each policy batch (per column), if already compressed.
        Args:
            columns (Set[str]): Set of column names to decompress.
        Returns:
            MultiAgentBatch: This very MultiAgentBatch.
        """
        for batch in self.policy_batches.values():
            batch.decompress_if_needed(columns)
        return self

    def __str__(self):
        return "MultiAgentBatch({}, env_steps={})".format(
            str(self.policy_batches), self.count)

    def __repr__(self):
        return "MultiAgentBatch({}, env_steps={})".format(
            str(self.policy_batches), self.count)

    # Deprecated.
    def total(self):
        return self.agent_steps()