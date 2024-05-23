import itertools
import pickle

import numpy as np
import torch
from numpy.typing import NDArray

from src.data.graph import DynamicSpatialTemporalGraph


def _calc_bbox_center(bbox) -> NDArray:
    return (bbox[1] - bbox[0]) / 2 + bbox[0]


def _gen_edge_index(node_idxs) -> list:
    e_i = [[idxs[i] for i in range(len(idxs))] for idxs in node_idxs]
    e_j = [[idxs[(i + 1) % len(idxs)] for i in range(len(idxs))] for idxs in node_idxs]
    e_i = list(itertools.chain.from_iterable(e_i))
    e_j = list(itertools.chain.from_iterable(e_j))
    return [e_i, e_j]


def _gen_edge_attr_s(pos, edge_indexs_s) -> NDArray:
    pos = np.array(pos)
    diffs = np.array(
        [np.abs(pos[i] - pos[j]) for i, j in zip(*edge_indexs_s)], dtype=np.float32
    )
    return diffs


def _gen_edge_attr_t(pos, time, edge_indexs_t) -> NDArray:
    pos = np.array(pos)
    pos_diffs = np.array([np.abs(pos[i] - pos[j]) for i, j in zip(*edge_indexs_t)])
    tm_diffs = np.array([abs(time[i] - time[j]) for i, j in zip(*edge_indexs_t)])
    return (pos_diffs * (1 / tm_diffs.reshape(-1, 1))).astype(np.float32)


def group_pkl_to_tensor(pkl, bbox_transform, kps_transform):
    human_tracking_data, img_size = pickle.loads(pkl)

    unique_ids = set(
        itertools.chain.from_iterable(
            [[idv["id"] for idv in idvs] for idvs in human_tracking_data]
        )
    )
    unique_ids = list(unique_ids)

    # collect data
    node_idxs_s = []
    node_idxs_t = {_id: [] for _id in unique_ids}
    node_idx = 0
    time = []
    y = []
    bbox = []
    kps = []
    for t, idvs in enumerate(human_tracking_data):
        node_idxs_s.append([])
        for idv in idvs:
            _id = idv["id"]
            time.append(t)
            y.append(_id)
            bbox.append(np.array(idv["bbox"], dtype=np.float32)[:4])
            kps.append(np.array(idv["keypoints"], dtype=np.float32)[:, :2])
            node_idxs_s[t].append(node_idx)
            node_idxs_t[_id].append(node_idx)
            node_idx += 1

    bbox = np.array(bbox).reshape(-1, 2, 2)
    bbox = bbox_transform(bbox, img_size)  # (-1, 2, 2)

    kps = np.array(kps)
    kps = kps_transform(kps, bbox, img_size)  # (-1, 34, 2)

    pos = [_calc_bbox_center(b) for b in bbox]

    edge_index_s = _gen_edge_index(node_idxs_s)
    edge_index_t = _gen_edge_index(list(node_idxs_t.values()))
    edge_attr_s = _gen_edge_attr_s(pos, edge_index_s)
    edge_attr_t = _gen_edge_attr_t(pos, time, edge_index_t)

    return DynamicSpatialTemporalGraph(
        None,
        torch.tensor(y, dtype=torch.long).contiguous(),
        torch.tensor(pos, dtype=torch.float32).contiguous(),
        torch.tensor(time, dtype=torch.long).contiguous(),
        torch.tensor(edge_index_s, dtype=torch.long).contiguous(),
        torch.tensor(edge_attr_s, dtype=torch.float32).contiguous(),
        torch.tensor(edge_index_t, dtype=torch.long).contiguous(),
        torch.tensor(edge_attr_t, dtype=torch.float32).contiguous(),
    )
