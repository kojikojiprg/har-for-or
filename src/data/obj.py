from collections import deque
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
    def __init__(self, shard_pattern, maxcount, verbose=0, post=None, max_que_size=1000):
        super().__init__(shard_pattern, maxcount, verbose=verbose, post=post)
        self.write_que = deque()
        self.finished = False
        self.watch_dog_count = 0
        self.verbose = bool(verbose)

    def add_write_que(self, data):
        self.write_que.append(data)
        if self.verbose:
            print(f"Put, qsize:{self.write_que_size()}")

    def set_finish_writing(self):
        self.write_que.append(None)
        self.finished = True

    def write_que_size(self):
        return len(self.write_que)

    def is_write_que_empty(self):
        return len(self.write_que) == 0

    def write_async(self):
        while True:
            while self.is_write_que_empty():
                time.sleep(0.001)

                self.watch_dog_count += 1
                if self.watch_dog_count > 60 * 10 / 0.001:  # wait for 10 min
                    raise RuntimeError("The dog barked in SharedShardWriter.write_async process!")
            self.watch_dog_count = 0

            data = self.write_que.popleft()
            if data is None:
                return  # complete writing all data
            self.write(data)
            del data

            if self.finished:
                print("Remaining write queue size:", self.write_que_size() - 1)
            else:
                if self.verbose:
                    print(f"Pop and written, qsize:{self.write_que_size()}")
