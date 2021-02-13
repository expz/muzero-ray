import numpy as np
import tensorflow as tf
import tree


class StructureList:
    def __init__(self, length):
        self.length = length

    def add_batch(self, values, rows, slots=None):
        raise NotImplementedError()
    
    def select(self, indices, slots=None):
        raise NotImplementedError()
    
    def size_bytes(self):
        raise NotImplementedError()


class TFStructureList(tf.Module, StructureList):
    """
    Some code copied from
    https://github.com/tensorflow/agents/blob/master/tf_agents/replay_buffers/table.py
    """
    def __init__(self, length, tensor_spec, scope='structure_list'):
        tf.Module.__init__(self, name='StructureList')
        StructureList.__init__(self, length)
        self._tensor_spec = tensor_spec

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
            # Not sure if this is necessary
            del update
        # Not sure if this is necessary
        del flattened_values

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


class ArraySpec:
    def __init__(self, shape, dtype, name=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name
    
    @property
    def size(self):
        return self.dtype.itemsize * np.prod(self.shape)


class NPStructureList(StructureList):
    """
    Some code copied from
    https://github.com/tensorflow/agents/blob/master/tf_agents/replay_buffers/table.py
    """
    def __init__(self, length, tensor_spec, scope='structure_list'):
        StructureList.__init__(self, length)
        self._tensor_spec = tensor_spec
        self.__i = 0

        def _create_unique_slot_name(spec):
            name = spec.name
            if name is None:
                name = f'slot{self.__i}'
                self.__i += 1
            return name

        self._slots = tree.map_structure(_create_unique_slot_name, self._tensor_spec)
        self._flattened_slots = tree.flatten(self._slots)

        def _create_storage(spec):
            """Create storage for a slot, track it."""
            shape = (self.length,) + spec.shape
            new_storage = np.zeros(shape, dtype=spec.dtype)
            return new_storage

        self._storage = tree.map_structure(_create_storage, self._tensor_spec)

        self._slot2storage_map = dict(
            zip(self._flattened_slots, tree.flatten(self._storage)))

    def add_batch(self, values, rows, slots=None):
        flattened_slots = tree.flatten(slots) if slots is not None else self._flattened_slots
        flattened_values = tree.flatten(values)
        for slot, value in zip(flattened_slots, flattened_values):
            rows_expanded = np.reshape(rows, (-1,) + tuple(1 for _ in value.shape[1:]))
            indices = np.broadcast_to(rows_expanded, (len(rows),) + value.shape[1:])
            np.put_along_axis(self._slot2storage_map[slot], indices, value, axis=0)
            # Not sure if this is necessary
            del rows_expanded
            del indices
        # Not sure if this is necessary
        del flattened_slots
        del flattened_values

    def select(self, rows):
        def _gather(arr):
            indices = np.reshape(rows, (-1,) + tuple(1 for _ in arr.shape[1:]))
            return np.take_along_axis(arr, indices, axis=0)

        return tree.map_structure(_gather, self._storage)

    def size_bytes(self):
        storage_size = 0
        storage_bytes = 0
        for arr in tree.flatten(self._storage):
            arr_size = np.prod(arr.shape)
            storage_size += arr_size
            storage_bytes += arr_size * arr.dtype.itemsize
        return storage_size, storage_bytes
