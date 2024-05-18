import functools
import itertools
import os
import sys
from multiprocessing import Pool
from multiprocessing.managers import SyncManager
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch
import webdataset as wbs
from numpy.typing import NDArray
from tqdm import tqdm

from .data import SpatialTemporalData
from .functional import (
    calc_bbox_center,
    gen_edge_attr_s,
    gen_edge_attr_t,
    gen_edge_index,
)
from .transform import FlowToTensor, FrameToTensor, NormalizeX

sys.path.append("src")
from utils import json_handler, video


class ShardManager(SyncManager):
    pass


def _write_async(
    n_frame: int,
    seq_frames: List[NDArray],
    seq_optflows: List[NDArray],
    seq_inds: List[Dict[str, Any]],
    seq_ids: List[int],
    lock: Any,
    sink: wbs.ShardWriter,
    pbar: tqdm = None,
):
    with lock:
        unique_ids = set(itertools.chain.from_iterable(seq_ids))
        inds_dict = {}
        node_idxs_s = []
        node_idxs_t = {_id: [] for _id in unique_ids}
        node_idx = 0
        for t, inds in enumerate(seq_inds):
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
            "npz": {
                "frame": np.array(seq_frames),
                "optical_flow": np.array(seq_optflows),
            },
            "pickle": (inds_dict, node_idxs_s, node_idxs_t),
        }
        sink.write(data)
        if pbar is not None:
            pbar.write(f"Complete writing n_frame:{n_frame}", end="", nolock=True)


def create_shards(video_path: str, config: SimpleNamespace):
    data_root = os.path.dirname(video_path)
    video_num = os.path.basename(video_path).split(".")[0]
    dir_path = os.path.join(data_root, video_num)

    maxsize = float(config.max_shard_size)
    seq_len = int(config.seq_len)
    stride = int(config.stride)
    shard_pattern = f"dstg-w{seq_len}-s{stride}" + "-%05d.tar"

    shard_pattern = os.path.join(dir_path, "shards", shard_pattern)
    os.makedirs(os.path.dirname(shard_pattern), exist_ok=True)

    json_path = os.path.join(dir_path, "json", "pose.json")
    indvisuals = json_handler.load(json_path)

    cap = video.Capture(video_path)
    prev_frame = cap.read(0)[1]
    cap.set_pos_frame_count(0)  # reset read position

    ShardManager.register("Tqdm", tqdm)
    ShardManager.register("ShardWriter", wbs.ShardWriter)
    with Pool() as pool, ShardManager() as manager:
        lock = manager.Lock()
        pbar = manager.Tqdm(total=cap.frame_count, ncols=100)
        sink = manager.ShardWriter(shard_pattern, maxsize)
        write_async_partial = functools.partial(
            _write_async, lock=lock, sink=sink, pbar=pbar
        )
        async_results = []

        seq_frames = []
        seq_optflows = []
        seq_inds = []
        seq_ids = []
        for n_frame in range(cap.frame_count):
            frame = cap.read()[1]
            seq_frames.append(frame)
            seq_optflows.append(video.optical_flow(prev_frame, frame))
            prev_frame = frame

            inds_tmp = [i for i in indvisuals if i["n_frame"] == n_frame]
            seq_inds.append(inds_tmp)

            seq_ids.append([i["id"] for i in inds_tmp])

            if n_frame < seq_len - 1:
                pbar.update()
                continue

            if (n_frame - (seq_len - 1)) % stride == 0:
                result = pool.apply_async(
                    write_async_partial,
                    args=(n_frame, seq_frames, seq_optflows, seq_inds, seq_ids),
                )
                async_results.append(result)

            # update data
            seq_frames = seq_frames[1:]
            seq_optflows = seq_optflows[1:]
            seq_inds = seq_inds[1:]
            seq_ids = seq_ids[1:]
            pbar.update()

        print("Waiting for writing shards.")
        [result.wait() for result in async_results]
        sink.close()
        pbar.close()
        del cap, seq_frames, seq_optflows, seq_inds, seq_ids
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
