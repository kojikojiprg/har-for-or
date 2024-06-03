import io
import itertools

import numpy as np
import torch
from numpy.typing import NDArray

from src.data.graph import DynamicSpatialTemporalGraph


def _calc_bbox_center(bbox) -> NDArray:
    return (bbox[1] - bbox[0]) / 2 + bbox[0]


def _gen_edge_index(node_idxs) -> list:
    e = [list(itertools.permutations(idxs, 2)) for idxs in node_idxs]
    e = list(itertools.chain.from_iterable(e))
    e = np.array(e, dtype=np.uint16).T.tolist()
    return e


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


def group_npz_to_tensor(
    sample, data_type, frame_transform, flow_transform, bbox_transform, kps_transform
):
    npz = sample["npz"]
    npz = np.load(io.BytesIO(npz))

    meta, ids, frames, flows, bboxs, kps, img_size = list(npz.values())

    # collect data
    time = torch.tensor(meta[:, 0], dtype=torch.long).contiguous()
    y = torch.tensor(ids, dtype=torch.long).contiguous()

    if data_type == "keypoints":
        kps = kps_transform(kps, bboxs)
        x = torch.tensor(kps, dtype=torch.float32).contiguous()
    elif data_type == "images":
        frames = frame_transform(frames)
        flows = flow_transform(flows)
        x = torch.cat([frames, flows], dim=1).contiguous()

    bboxs = bbox_transform(bboxs, img_size)
    bbox_centers = [_calc_bbox_center(b) for b in bboxs]
    bboxs = torch.tensor(bboxs, dtype=torch.float32).contiguous()

    # create edges
    seq_len, n = np.max(meta, axis=0) + 1
    node_idxs_s = [[] for i in range(seq_len)]
    node_idxs_t = [[] for i in range(n)]
    node_idx = 0
    for t, i in meta:
        node_idxs_s[t].append(node_idx)
        node_idxs_t[i].append(node_idx)
        node_idx += 1
    edge_index_s = _gen_edge_index(node_idxs_s)
    edge_index_s = torch.tensor(edge_index_s, dtype=torch.long).contiguous()
    edge_index_t = _gen_edge_index(node_idxs_t)
    edge_index_t = torch.tensor(edge_index_t, dtype=torch.long).contiguous()
    edge_attr_s = _gen_edge_attr_s(bbox_centers, edge_index_s)
    edge_attr_s = torch.tensor(edge_attr_s, dtype=torch.float32).contiguous()
    edge_attr_t = _gen_edge_attr_t(bbox_centers, time, edge_index_t)
    edge_attr_t = torch.tensor(edge_attr_t, dtype=torch.float32).contiguous()

    del sample, npz, frames, flows  # release memory

    graph = DynamicSpatialTemporalGraph(
        x, y, bboxs, time, edge_index_s, edge_attr_s, edge_index_t, edge_attr_t
    )
    return graph
