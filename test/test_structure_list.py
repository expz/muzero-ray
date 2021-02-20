import numpy as np

from .conftest import random_batch
from muzero.structure_list import NPStructureList

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
