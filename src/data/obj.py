import queue
import time
from multiprocessing import shared_memory
from multiprocessing.managers import SyncManager

import numpy as np
import webdataset as wds


class ShardWritingManager(SyncManager):
    pass


class SharedNDArray:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.shm = self.create_shared_memory()

    def calc_ndarray_size(self):
        return np.prod(self.shape) * np.dtype(self.dtype).itemsize

    def create_shared_memory(self):
        size = self.calc_ndarray_size()
        return shared_memory.SharedMemory(name=self.name, create=True, size=size)

    def get_shared_memory(self):
        shm = shared_memory.SharedMemory(name=self.name)
        return shm

    def ndarray(self):
        shm = self.get_shared_memory()
        return np.ndarray(self.shape, self.dtype, buffer=shm.buf), shm

    def unlink(self):
        self.shm.close()
        self.shm.unlink()


class SharedShardWriter(wds.ShardWriter):
    def __init__(self, shard_pattern, maxcount, verbose=0, post=None):
        super().__init__(shard_pattern, maxcount, verbose=verbose, post=post)
        self.write_que = queue.Queue()
        self.finished = False

    def add_write_que(self, data):
        self.write_que.put(data)

    def set_finish_writing(self):
        self.finished = True

    def completed(self):
        return self.finished and self.write_que.empty()

    def write_async(self):
        while True:
            while self.write_que.empty():
                time.sleep(0.001)
                if self.completed():
                    return  # complete writing all data (avoid infinite loop)

            self.write(self.write_que.get_nowait())

            if self.completed():
                return  # complete writing all data
