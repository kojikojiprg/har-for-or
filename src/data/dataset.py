import functools
import gc
import itertools
import os
import sys
import time
import warnings
from glob import glob
from multiprocessing import shared_memory
from multiprocessing.managers import SyncManager
from types import SimpleNamespace

from torch.utils.data import IterableDataset

warnings.filterwarnings("ignore")

import numpy as np
import torch
import webdataset as wbs
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm

from .functional import (
    calc_bbox_center,
    gen_edge_attr_s,
    gen_edge_attr_t,
    gen_edge_index,
)
from .transform import FlowToTensor, FrameToTensor, NormalizeKeypoints, TimeSeriesResize

set_start_method("spawn", force=True)
sys.path.append("src")
from model import HumanTracking
from utils import json_handler, video


class ShardWritingManager(SyncManager):
    @staticmethod
    def calc_ndarray_size(shape, dtype):
        return np.prod(shape) * np.dtype(dtype).itemsize

    @classmethod
    def create_shared_ndarray(cls, name, shape, dtype):
        size = cls.calc_ndarray_size(shape, dtype)
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        return np.ndarray(shape, dtype, shm.buf), shm


def check_full(tail_frame, tail_ind, head, que_len):
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


def _transform_images(frames, flows, frame_trans, flow_trans):
    frames = frame_trans(frames).numpy()
    flows = flow_trans(flows).numpy()

    return np.concatenate([frames, flows], axis=1)  # (seq_len, 5, h, w)


_partial_transform_images = functools.partial(
    _transform_images, frame_trans=FrameToTensor(), flow_trans=FlowToTensor()
)


def _create_ind_data(copy_ind_que, sorted_idxs, unique_ids, img_size, kps_norm_func):
    inds_dict = {_id: {"bbox": [], "keypoints": []} for _id in unique_ids}
    for idx in sorted_idxs:
        inds_tmp = copy_ind_que[idx]
        ids_tmp = [ind["id"] for ind in inds_tmp]
        for key_id in unique_ids:
            if key_id in ids_tmp:
                ind = [ind for ind in inds_tmp if ind["id"] == key_id][0]
                inds_dict[key_id]["bbox"].append(
                    np.array(ind["bbox"], dtype=np.float32)[:4]
                )
                inds_dict[key_id]["keypoints"].append(
                    np.array(ind["keypoints"], dtype=np.float32)[:, :2]
                )
            else:
                # append dmy
                inds_dict[key_id]["bbox"].append(np.full((4,), -1, dtype=np.float32))
                inds_dict[key_id]["keypoints"].append(
                    np.full((17, 2), -1, dtype=np.float32)
                )

    ids = list(inds_dict.keys())  # id

    bbox = [ind["bbox"] for ind in inds_dict.values()]
    bbox = np.array(bbox)  # (-1, seq_len, 4)
    n_id, seq_len = bbox.shape[:2]
    bbox = bbox.reshape(n_id, seq_len, 2, 2)  # (-1, seq_len, 2, 2)

    kps = [ind["keypoints"] for ind in inds_dict.values()]
    kps = np.array(kps)  # (-1, seq_len, 17, 2)
    kps = kps_norm_func(kps, bbox, img_size)  # (-1, seq_len, 34, 2)

    return ids, bbox, kps


_partial_create_ind_data = functools.partial(
    _create_ind_data, kps_norm_func=NormalizeKeypoints()
)


def _create_group_data(copy_ind_que, sorted_idxs, unique_ids, img_size, kps_norm_func):
    node_dict = {}
    node_idxs_s = []
    node_idxs_t = {_id: [] for _id in unique_ids}
    node_idx = 0
    for t, idx in enumerate(sorted_idxs):
        inds_tmp = copy_ind_que[idx]
        node_idxs_s.append([])
        for ind in inds_tmp:
            _id = ind["id"]
            node_dict[node_idx] = (
                t,
                _id,
                np.array(ind["bbox"], dtype=np.float32)[:4],
                np.array(ind["keypoints"], dtype=np.float32)[:, :2],
            )
            node_idxs_s[t].append(node_idx)
            node_idxs_t[_id].append(node_idx)
            node_idx += 1

    bbox = [ind[2] for ind in node_dict.values()]
    bbox = np.array(bbox).reshape(-1, 2, 2)  # (-1, 2, 2)

    kps = [ind[3] for ind in node_dict.values()]
    kps = kps_norm_func(np.array(kps), bbox, img_size)  # (-1, 34, 2)

    y = [ind[1] for ind in node_dict.values()]  # id
    pos = [calc_bbox_center(ind[2]) / img_size for ind in node_dict.values()]
    time = [ind[0] for ind in node_dict.values()]  # t
    edge_index_s = gen_edge_index(node_idxs_s)
    edge_index_t = gen_edge_index(list(node_idxs_t.values()))
    edge_attr_s = gen_edge_attr_s(pos, edge_index_s)
    edge_attr_t = gen_edge_attr_t(pos, time, edge_index_t)

    return bbox, kps, y, pos, time, edge_index_s, edge_attr_s, edge_index_t, edge_attr_t


_partial_create_group_data = functools.partial(
    _create_group_data, kps_norm_func=NormalizeKeypoints()
)


def _write_async(
    n_frame, head_val, lock, sink, frame_que, flow_que, ind_que, pbar, video_num
):
    with lock:
        # copy queue
        copy_frame_que = frame_que.copy()
        copy_flow_que = flow_que.copy()
        copy_ind_que = list(ind_que)

    que_len = len(copy_ind_que)
    assert n_frame % que_len == head_val

    # transform images
    img_size = copy_frame_que.shape[1:3]
    images = _partial_transform_images(copy_frame_que, copy_flow_que)

    sorted_idxs = list(range(head_val, que_len)) + list(range(0, head_val))
    unique_ids = set(
        itertools.chain.from_iterable(
            [[ind["id"] for ind in inds] for inds in copy_ind_que]
        )
    )

    # create individual data
    ind_data = _partial_create_ind_data(copy_ind_que, sorted_idxs, unique_ids, img_size)

    # create group data
    group_data = _partial_create_group_data(
        copy_ind_que, sorted_idxs, unique_ids, img_size
    )

    data = {
        "__key__": f"{video_num}_{n_frame}",
        "npz": {"imgs": images},
        "individuals.pickle": ind_data,
        "group.pickle": group_data,
    }
    sink.write(data)

    # pbar.write(f"Complete writing n_frame:{n_frame}")
    pbar.update()
    del data
    gc.collect()


def create_shards(
    video_path: str,
    config: SimpleNamespace,
    config_human_tracking: SimpleNamespace,
    device: str,
    n_processes: int = None,
):
    if n_processes is None:
        n_processes = os.cpu_count()

    data_root = os.path.dirname(video_path)
    video_num = os.path.basename(video_path).split(".")[0]
    dir_path = os.path.join(data_root, video_num)

    json_path = os.path.join(dir_path, "json", "pose.json")
    if not os.path.exists(json_path):
        model_ht = HumanTracking(config_human_tracking, device)
    else:
        model_ht = None

    shard_maxsize = float(config.max_shard_size)
    seq_len = int(config.seq_len)
    stride = int(config.stride)
    shard_pattern = f"seq_len{seq_len}-stride{stride}" + "-%05d.tar"

    shard_pattern = os.path.join(dir_path, "shards", shard_pattern)
    os.makedirs(os.path.dirname(shard_pattern), exist_ok=True)

    ShardWritingManager.register("Tqdm", tqdm)
    ShardWritingManager.register("Capture", video.Capture)
    ShardWritingManager.register("ShardWriter", wbs.ShardWriter)
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
        ind_que = swm.list([[] for _ in range(seq_len)])
        tail_ind = swm.Value("i", 0)
        pool.apply_async(
            _human_tracking_async,
            (lock, cap, ind_que, tail_ind, head, pbar_ht, json_path, model_ht),
        )

        # create shard writer and start writing
        sink = swm.ShardWriter(shard_pattern, maxsize=shard_maxsize, verbose=0)
        write_async_partial = functools.partial(
            _write_async,
            lock=lock,
            sink=sink,
            frame_que=frame_que,
            flow_que=flow_que,
            ind_que=ind_que,
            pbar=pbar_w,
            video_num=video_num,
        )
        check_full_partial = functools.partial(
            check_full,
            tail_frame=tail_frame,
            tail_ind=tail_ind,
            head=head,
            que_len=seq_len,
        )
        for n_frame in range(seq_len, frame_count, stride):
            while not check_full_partial():
                time.sleep(0.01)

            # start writing
            result = pool.apply_async(
                write_async_partial,
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


def _npz_to_tensor(npz, img_transform):
    imgs = torch.tensor(list(npz.values())[0], dtype=torch.float32).contiguous()
    return img_transform(imgs)


def _ind_pkl_to_tensor(ind_pkl):
    ids, bbox, kps = ind_pkl
    return (
        torch.tensor(ids, dtype=torch.long).contiguous(),
        torch.tensor(bbox, dtype=torch.float32).contiguous(),
        torch.tensor(kps, dtype=torch.float32).contiguous(),
    )


def _grp_pkl_to_tensor(grp_pkl):
    bbox, kps, y, pos, time, edge_index_s, edge_attr_s, edge_index_t, edge_attr_t = (
        grp_pkl
    )
    return (
        torch.tensor(bbox, dtype=torch.long).contiguous(),
        torch.tensor(kps, dtype=torch.long).contiguous(),
        torch.tensor(y, dtype=torch.long).contiguous(),
        torch.tensor(pos, dtype=torch.float32).contiguous(),
        torch.tensor(time, dtype=torch.long).contiguous(),
        torch.tensor(edge_index_s, dtype=torch.long).contiguous(),
        torch.tensor(edge_attr_s, dtype=torch.float32).contiguous(),
        torch.tensor(edge_index_t, dtype=torch.long).contiguous(),
        torch.tensor(edge_attr_t, dtype=torch.float32).contiguous(),
    )


def load_dataset(
    data_root: str, dataset_type: str, resize_ratio: float = 1.0
) -> IterableDataset:
    shard_paths = []
    data_dirs = sorted(glob(os.path.join(data_root, "*/")))
    for dir_path in data_dirs:
        shard_paths += sorted(glob(os.path.join(dir_path, "shards", "*.tar")))

    partial_npz_to_tensor = functools.partial(
        _npz_to_tensor, img_transform=TimeSeriesResize(resize_ratio)
    )

    dataset = wbs.WebDataset(shard_paths).decode()

    if dataset_type == "individual":
        dataset = dataset.to_tuple("npz", "individuals.pickle")
        dataset = dataset.map_tuple(partial_npz_to_tensor, _ind_pkl_to_tensor)
    elif dataset_type == "group":
        dataset = dataset.to_tuple("npz", "group.pickle")
        dataset = dataset.map_tuple(partial_npz_to_tensor, _grp_pkl_to_tensor)
    else:
        raise ValueError

    return dataset
