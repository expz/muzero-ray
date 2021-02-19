import numpy as np
import pytest

from muzero.sample_batch import SampleBatch
from muzero.structure_list import ArraySpec, NPStructureList


@pytest.fixture
def atari_tensor_spec():
    n_channels = 4
    frame_shape = (96, 96)
    loss_steps = 6
    action_count = 4
    tensor_spec = (
        ArraySpec(frame_shape + (n_channels,), np.float32, SampleBatch.CUR_OBS),
        ArraySpec((1,), np.int32, SampleBatch.EPS_ID),
        ArraySpec((1,), np.int32, SampleBatch.UNROLL_ID),
        ArraySpec((loss_steps,), np.int32, SampleBatch.ACTIONS),
        ArraySpec((action_count,), np.float32, 'action_dist_probs'),
        ArraySpec((1,), np.float32, SampleBatch.ACTION_PROB),
        ArraySpec((1,), np.float32, SampleBatch.ACTION_LOGP),
        ArraySpec((1,), np.bool, SampleBatch.DONES),
        ArraySpec((1,), np.float32, SampleBatch.REWARDS),
        ArraySpec((loss_steps, action_count), np.float32, 'rollout_policies'),
        ArraySpec((loss_steps,), np.float32, 'rollout_rewards'),
        ArraySpec((loss_steps,), np.float32, 'rollout_values'),
    )
    return tensor_spec

def random_struct(tensor_spec):
    struct = []
    for spec in tensor_spec:
        if spec.dtype == np.float32:
            struct.append(np.random.uniform(size=spec.shape))
        elif spec.dtype == np.int32:
            struct.append(np.random.randint(0, 1 << 24, size=spec.shape))
        elif spec.dtype == np.bool:
            struct.append(np.random.choice(a=[False, True], size=spec.shape))
        else:
            raise NotImplementedError(f"tensor spec dtype {spec.dtype} not supported")
    return tuple(struct)

def random_batch(tensor_spec, batch_size):
    structs = []
    batch_size = 4
    for _ in range(batch_size):
        structs.append(random_struct(tensor_spec))
    batch = []
    for i in range(len(tensor_spec)):
        batch_tensor = []
        for j in range(len(structs)):
            batch_tensor.append(structs[j][i])
        batch.append(np.array(batch_tensor))
    return tuple(batch)

def test_atari_tensor_spec(atari_tensor_spec):
    buffer = NPStructureList(10, atari_tensor_spec)
    batch_size = 4
    batch = random_batch(atari_tensor_spec, batch_size)
    buffer.add_batch(batch, list(range(batch_size)))
    batch2 = buffer.select([1, 3])
    for tensor, tensor2 in zip(batch, batch2):
        assert np.allclose(tensor[1], tensor2[0])
        assert np.allclose(tensor[3], tensor2[1])

def test_zero_size_bytes(atari_tensor_spec):
    buffer = NPStructureList(0, atari_tensor_spec)
    size, bytes = buffer.size_bytes()
    assert size == 0
    assert bytes == 0

def test_nonzero_size_bytes(atari_tensor_spec):
    buffer = NPStructureList(10, atari_tensor_spec)
    size, bytes = buffer.size_bytes()
    assert size > 0
    assert bytes > 0
