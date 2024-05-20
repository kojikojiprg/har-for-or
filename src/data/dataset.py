import functools
import gc
import itertools
import os
import sys
import time
import warnings
from multiprocessing import shared_memory
from multiprocessing.managers import SyncManager
from types import SimpleNamespace
from typing import List

warnings.filterwarnings("ignore")

import numpy as np
import torch
import webdataset as wbs
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm

from .data import SpatialTemporalData
from .functional import (
    calc_bbox_center,
    gen_edge_attr_s,
    gen_edge_attr_t,
    gen_edge_index,
)
from .transform import FlowToTensor, FrameToTensor, NormalizeX

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

    @staticmethod
    def check_que_len(head_frame, head_ind, tail, seq_len):
        th = tail.value + seq_len
        return head_frame.value >= th and head_ind.value >= th


def _optical_flow_async(lock, cap, frame_que, flow_que, head_frame, tail, pbar):
    que_len = frame_que.shape[0]
    with lock:
        prev_frame = cap.read(0)[1]
        cap.set_pos_frame_count(0)  # reset read position

    for n_frame in range(cap.get_frame_count()):
        with lock:
            frame = cap.read(n_frame)[1]
        flow = video.optical_flow(prev_frame, frame)
        prev_frame = frame

        while head_frame.value - tail.value >= que_len:
            time.sleep(0.01)  # waiting for updating tail

        with lock:
            if n_frame < que_len:
                frame_que[n_frame] = frame
                flow_que[n_frame] = flow
            else:
                frame_que[:-1] = frame_que[:1]
                flow_que[:-1] = flow_que[:1]
                frame_que[-1] = frame
                flow_que[-1] = flow
            head_frame.value += 1

        pbar.update()


def _human_tracking_async(
    lock,
    cap,
    individuals_que,
    head_ind,
    tail,
    que_len,
    pbar,
    json_path: str,
    model=None,
):
    do_human_tracking = not os.path.exists(json_path)
    if not do_human_tracking:
        json_data = json_handler.load(json_path)

    for n_frame in range(cap.get_frame_count()):
        if do_human_tracking:
            with lock:
                frame = cap.read(n_frame)[1]
            inds_tmp = model.predict(frame, n_frame)
        else:
            inds_tmp = [ind for ind in json_data if ind["n_frame"] == n_frame]

        while head_ind.value - tail.value >= que_len:
            time.sleep(0.01)  # waiting for updating tail

        with lock:
            if n_frame < que_len:
                individuals_que.append(inds_tmp)
            else:
                individuals_que = individuals_que[1:]
                individuals_que.append(inds_tmp)
            head_ind.value += 1
        pbar.update()


def _write_async(n_frame, lock, sink, frame_que, flow_que, ind_que, pbar):
    pbar.write(f"Start writing n_frame:{n_frame}")
    with lock:
        unique_ids = set(
            itertools.chain.from_iterable(
                [[ind["id"] for ind in inds] for inds in ind_que]
            )
        )
        inds_dict = {}
        node_idxs_s = []
        node_idxs_t = {_id: [] for _id in unique_ids}
        node_idx = 0
        for t, inds in enumerate(ind_que):
            node_idxs_s.append([])
            for ind in inds:
                _id = ind["id"]
                inds_dict[node_idx] = (
                    t,
                    _id,
                    ind["bbox"][:4],
                    np.array(ind["keypoints"])[:, :2].tolist(),
                )
                node_idxs_s[t].append(node_idx)
                node_idxs_t[_id].append(node_idx)
                node_idx += 1

        data = {
            "__key__": str(n_frame),
            "npz": {"frame": frame_que, "optical_flow": flow_que},
            "pickle": (inds_dict, node_idxs_s, node_idxs_t),
        }

    sink.write(data)
    pbar.write(f"Complete writing n_frame:{n_frame}")
    del data
    gc.collect()


def create_shards(
    video_path: str,
    config: SimpleNamespace,
    config_human_tracking: SimpleNamespace,
    device: str,
    n_processes: int = 16,
):
    data_root = os.path.dirname(video_path)
    video_num = os.path.basename(video_path).split(".")[0]
    dir_path = os.path.join(data_root, video_num)

    json_path = os.path.join(dir_path, "json", "pose.json")
    if not os.path.exists(json_path):
        model = HumanTracking(config_human_tracking, device)
    else:
        model = None

    shard_maxsize = float(config.max_shard_size)
    seq_len = int(config.seq_len)
    stride = int(config.stride)
    shard_pattern = f"dstg-w{seq_len}-s{stride}" + "-%05d.tar"

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
        tail = swm.Value("i", 0)

        # create shared ndarray and start optical flow
        shape = (seq_len, img_size[1], img_size[0], 3)
        frame_que, frame_shm = swm.create_shared_ndarray("frame", shape, np.uint8)
        shape = (seq_len, img_size[1], img_size[0], 2)
        flow_que, flow_shm = swm.create_shared_ndarray("flow", shape, np.float16)
        head_frame = swm.Value("i", 0)
        pbar_of = swm.Tqdm(
            total=frame_count, ncols=100, desc="optical flow", position=1
        )
        result = pool.apply_async(
            _optical_flow_async,
            (lock, cap, frame_que, flow_que, head_frame, tail, pbar_of),
        )
        async_results.append(result)

        # create shared list of indiciduals and start human tracking
        ind_que = swm.list()
        head_ind = swm.Value("i", 0)
        pbar_ht = swm.Tqdm(
            total=frame_count, ncols=100, desc="human tracking", position=2
        )
        result = pool.apply_async(
            _human_tracking_async,
            (lock, cap, ind_que, head_ind, tail, seq_len, pbar_ht, json_path, model),
        )
        async_results.append(result)

        pbar_w = swm.Tqdm(total=frame_count, ncols=100, desc="writing", position=3)
        sink = swm.ShardWriter(shard_pattern, shard_maxsize, verbose=0)

        write_async_partial = functools.partial(
            _write_async,
            lock=lock,
            sink=sink,
            frame_que=frame_que,
            flow_que=flow_que,
            ind_que=ind_que,
            pbar=pbar_w,
        )
        check_que_len_partial = functools.partial(
            swm.check_que_len,
            head_frame=head_frame,
            head_ind=head_ind,
            tail=tail,
            seq_len=seq_len,
        )
        for n_frame in range(0, frame_count, stride):
            while not check_que_len_partial():
                # waiting for optical flow and human tracking
                time.sleep(0.01)

            # start writing
            result = pool.apply_async(write_async_partial, args=(n_frame,))
            async_results.append(result)
            tail.value += stride
            pbar_w.update(stride)

        print("Waiting for writing shards.")
        [result.wait() for result in async_results]
        frame_shm.close()
        frame_shm.unlink()
        flow_shm.close()
        flow_shm.unlink()
        pbar_of.close()
        pbar_ht.close()
        pbar_w.close()
        sink.close()
        del frame_que, flow_que
    print("Complete!")


def _npz_to_tensor(npz, frame_trans, flow_trans):
    frames, flows = list(npz.values())
    frames = frame_trans(frames)
    flows = flow_trans(flows)

    return torch.cat([frames, flows], dim=1).contiguous()


_partial_npz_to_tensor = functools.partial(
    _npz_to_tensor, frame_trans=FrameToTensor(), flow_trans=FlowToTensor()
)


def _extract_individual_features(pkl, kps_norm_func):
    inds_dict, _, _ = pkl
    bboxs = [ind[2] for ind in inds_dict.values()]  # bbox
    kps = [kps_norm_func(ind[3]) for ind in inds_dict.values()]  # keypoints
    return bboxs, kps


def _create_graph(pkl, norm_func, has_edge_attr):
    inds_dict, node_idxs_s, node_idxs_t = pkl
    x = [norm_func(ind[3]) for ind in inds_dict.values()]  # keypoints
    y = [ind[1] for ind in inds_dict.values()]  # id
    pos = [calc_bbox_center(ind[2]) for ind in inds_dict.values()]
    time = [ind[0] for ind in inds_dict.values()]  # t
    edge_index_s = gen_edge_index(node_idxs_s)
    edge_index_t = gen_edge_index(list(node_idxs_t.values()))
    if has_edge_attr:
        edge_attr_s = gen_edge_attr_s(pos, edge_index_s)
        edge_attr_t = gen_edge_attr_t(pos, time, edge_index_t)
        edge_attr_s = torch.tensor(edge_attr_s, dtype=torch.float32)
        edge_attr_t = torch.tensor(edge_attr_t, dtype=torch.float32)
    else:
        edge_attr_s = None
        edge_attr_t = None

    return SpatialTemporalData(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(pos, dtype=torch.float32),
        torch.tensor(time, dtype=torch.long),
        torch.tensor(edge_index_s, dtype=torch.long),
        edge_attr_s,
        torch.tensor(edge_index_t, dtype=torch.long),
        edge_attr_t,
    )


def load_dataset(
    shard_paths: List[str], dataset_type: str, kps_norm_type: str, has_edge_attr: bool
):
    dataset = wbs.WebDataset(shard_paths).decode().to_tuple("npz", "pickle")
    if dataset_type == "individual":
        partial_extract_individual_features = functools.partial(
            _extract_individual_features,
            kps_norm_func=NormalizeX(kps_norm_type),
        )
        dataset = dataset.map_tuple(
            _partial_npz_to_tensor, partial_extract_individual_features
        )
    elif dataset_type == "group":
        partial_create_graph = functools.partial(
            _create_graph,
            norm_func=NormalizeX(kps_norm_type),
            has_edge_attr=has_edge_attr,
        )
        dataset = dataset.map_tuple(_partial_npz_to_tensor, partial_create_graph)
    else:
        raise ValueError

    return dataset
