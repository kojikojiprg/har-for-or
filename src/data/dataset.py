import functools
import gc
import os
import time
import warnings
from glob import glob
from multiprocessing import shared_memory
from multiprocessing.managers import SyncManager
from types import SimpleNamespace

warnings.filterwarnings("ignore")

import numpy as np
import webdataset as wds
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm

from src.model import HumanTracking
from src.utils import json_handler, video

from .transform import (
    FlowToTensor,
    FrameToTensor,
    NormalizeBbox,
    NormalizeKeypoints,
    group_pkl_to_tensor,
    images_to_tensor,
    individual_pkl_to_tensor,
)

set_start_method("spawn", force=True)


class ShardWritingManager(SyncManager):
    @staticmethod
    def calc_ndarray_size(shape, dtype):
        return np.prod(shape) * np.dtype(dtype).itemsize

    @classmethod
    def create_shared_ndarray(cls, name, shape, dtype):
        size = cls.calc_ndarray_size(shape, dtype)
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        return np.ndarray(shape, dtype, shm.buf), shm


def _check_full(tail_frame, tail_ind, head, que_len):
    is_frame_que_full = (tail_frame.value + 1) % que_len == head.value
    is_ind_que_full = (tail_ind.value + 1) % que_len == head.value
    is_eq = tail_frame.value == tail_ind.value
    return is_frame_que_full and is_ind_que_full and is_eq


def _optical_flow_async(lock, cap, frame_que, flow_que, tail_frame, head, pbar):
    que_len = frame_que.shape[0]

    with lock:
        prev_frame = cap.read(0)[1]
        cap.set_pos_frame_count(0)  # reset read position
        frame_count = cap.get_frame_count()

    for n_frame in range(frame_count):
        with lock:
            frame = cap.read(n_frame)[1]
        flow = video.optical_flow(prev_frame, frame)
        prev_frame = frame

        with lock:
            frame_que[tail_frame.value] = frame
            flow_que[tail_frame.value] = flow
            pbar.update()

        if n_frame + 1 == frame_count:
            break  # finish

        next_tail = (tail_frame.value + 1) % que_len
        while next_tail == head.value:
            time.sleep(0.01)
        tail_frame.value = next_tail


def _human_tracking_async(
    lock, cap, ind_que, tail_ind, head, pbar, json_path, model=None
):
    que_len = len(ind_que)

    do_human_tracking = not os.path.exists(json_path)
    if not do_human_tracking:
        json_data = json_handler.load(json_path)

    with lock:
        frame_count = cap.get_frame_count()

    for n_frame in range(frame_count):
        if do_human_tracking:
            with lock:
                frame = cap.read(n_frame)[1]
            inds_tmp = model.predict(frame, n_frame)
        else:
            inds_tmp = [ind for ind in json_data if ind["n_frame"] == n_frame]

        with lock:
            ind_que[tail_ind.value] = inds_tmp
            pbar.update()

        if n_frame + 1 == frame_count:
            break  # finish

        next_tail = (tail_ind.value + 1) % que_len
        while next_tail == head.value:
            time.sleep(0.01)
        tail_ind.value = next_tail

    if model is not None:
        del model


def _write_shard_async(
    n_frame, head_val, lock, sink, frame_que, flow_que, ht_que, pbar, video_name
):
    with lock:
        # copy queue
        copy_frame_que = frame_que.copy()
        copy_flow_que = flow_que.copy()
        copy_ht_que = list(ht_que)

    que_len = len(copy_ht_que)
    assert n_frame % que_len == head_val

    sorted_idxs = list(range(head_val, que_len)) + list(range(0, head_val))
    sorted_ht = []
    for idx in sorted_idxs:
        sorted_ht.append(copy_ht_que[idx])
    img_size = copy_frame_que.shape[1:3]

    data = {
        "__key__": f"{video_name}_{n_frame}",
        "frame.npy": copy_frame_que[sorted_idxs].astype(np.uint8),
        "flow.npy": copy_flow_que[sorted_idxs].astype(np.float32),
        "pickle": (sorted_ht, img_size),
    }
    sink.write(data)

    # pbar.write(f"Complete writing n_frame:{n_frame}")
    pbar.update()
    del data
    gc.collect()


def write_shards(
    video_path: str,
    config: SimpleNamespace,
    config_human_tracking: SimpleNamespace,
    device: str,
    n_processes: int = None,
):
    if n_processes is None:
        n_processes = os.cpu_count()

    data_root = os.path.dirname(video_path)
    video_name = os.path.basename(video_path).split(".")[0]
    dir_path = os.path.join(data_root, video_name)

    json_path = os.path.join(dir_path, "json", "pose.json")
    if not os.path.exists(json_path):
        model_ht = HumanTracking(config_human_tracking, device)
    else:
        model_ht = None

    shard_maxsize = float(config.max_shard_size)
    seq_len = int(config.seq_len)
    stride = int(config.stride)
    shard_pattern = f"seq_len{seq_len}-stride{stride}" + "-%06d.tar"

    shard_pattern = os.path.join(dir_path, "shards", shard_pattern)
    os.makedirs(os.path.dirname(shard_pattern), exist_ok=True)

    ShardWritingManager.register("Tqdm", tqdm)
    ShardWritingManager.register("Capture", video.Capture)
    ShardWritingManager.register("ShardWriter", wds.ShardWriter)
    with Pool(n_processes) as pool, ShardWritingManager() as swm:
        async_results = []

        lock = swm.Lock()
        cap = swm.Capture(video_path)
        frame_count, img_size = cap.get_frame_count(), cap.get_size()
        head = swm.Value("i", 0)

        # create progress bars
        pbar_of = swm.Tqdm(
            total=frame_count, ncols=100, desc="optical flow", position=1, leave=False
        )
        pbar_ht = swm.Tqdm(
            total=frame_count, ncols=100, desc="human tracking", position=2, leave=False
        )
        total = (frame_count - seq_len) // stride
        pbar_w = swm.Tqdm(
            total=total, ncols=100, desc="writing", position=3, leave=False
        )

        # create shared ndarray and start optical flow
        shape = (seq_len, img_size[1], img_size[0], 3)
        frame_que, frame_shm = swm.create_shared_ndarray("frame", shape, np.uint8)
        shape = (seq_len, img_size[1], img_size[0], 2)
        flow_que, flow_shm = swm.create_shared_ndarray("flow", shape, np.float32)
        tail_frame = swm.Value("i", 0)
        pool.apply_async(
            _optical_flow_async,
            (lock, cap, frame_que, flow_que, tail_frame, head, pbar_of),
        )

        # create shared list of indiciduals and start human tracking
        ht_que = swm.list([[] for _ in range(seq_len)])
        tail_ind = swm.Value("i", 0)
        pool.apply_async(
            _human_tracking_async,
            (lock, cap, ht_que, tail_ind, head, pbar_ht, json_path, model_ht),
        )

        # create shard writer and start writing
        sink = swm.ShardWriter(shard_pattern, maxsize=shard_maxsize, verbose=0)
        write_shard_async_f = functools.partial(
            _write_shard_async,
            lock=lock,
            sink=sink,
            frame_que=frame_que,
            flow_que=flow_que,
            ht_que=ht_que,
            pbar=pbar_w,
            video_name=video_name,
        )
        check_full_f = functools.partial(
            _check_full,
            tail_frame=tail_frame,
            tail_ind=tail_ind,
            head=head,
            que_len=seq_len,
        )
        for n_frame in range(seq_len, frame_count, stride):
            while not check_full_f():
                time.sleep(0.01)

            # start writing
            result = pool.apply_async(
                write_shard_async_f,
                (n_frame, head.value),
            )
            async_results.append(result)
            head.value = (head.value + stride) % seq_len

        while [r.wait() for r in async_results].count(True) > 0:
            time.sleep(0.01)
        pbar_of.close()
        pbar_ht.close()
        pbar_w.close()
        frame_shm.unlink()
        frame_shm.close()
        flow_shm.unlink()
        flow_shm.close()
        sink.close()
        del frame_que, flow_que


def load_dataset(data_root: str, dataset_type: str, config: SimpleNamespace):
    shard_paths = []
    data_dirs = sorted(glob(os.path.join(data_root, "*/")))

    seq_len = int(config.seq_len)
    stride = int(config.stride)
    resize_ratio = float(config.resize_ratio)
    shard_pattern = f"seq_len{seq_len}-stride{stride}" + "-%06d.tar"
    for dir_path in data_dirs:
        shard_paths += sorted(glob(os.path.join(dir_path, "shards", shard_pattern)))

    frame_to_tensor = functools.partial(
        images_to_tensor, transform=FrameToTensor(resize_ratio)
    )
    flow_to_tensor = functools.partial(
        images_to_tensor, transform=FlowToTensor(resize_ratio)
    )
    idv_pkl_to_tensor = functools.partial(
        individual_pkl_to_tensor,
        bbox_transform=NormalizeBbox(),
        kps_transform=NormalizeKeypoints(),
    )
    grp_pkl_to_tensor = functools.partial(
        group_pkl_to_tensor,
        bbox_transform=NormalizeBbox(),
        kps_transform=NormalizeKeypoints(),
    )

    if dataset_type == "individual":
        return wds.DataPipeline(
            wds.SimpleShardList(shard_paths),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.to_tuple("frame.npy", "flow.npy", "pickle"),
            wds.map_tuple(frame_to_tensor, flow_to_tensor, idv_pkl_to_tensor),
        )
    elif dataset_type == "group":
        return wds.DataPipeline(
            wds.SimpleShardList(shard_paths),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.to_tuple("frame.npy", "flow.npy", "pickle"),
            wds.map_tuple(frame_to_tensor, flow_to_tensor, grp_pkl_to_tensor),
        )
    else:
        raise ValueError
