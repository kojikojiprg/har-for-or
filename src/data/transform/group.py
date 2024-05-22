import numpy as np
import torch

from src.data.functional import (
    calc_bbox_center,
    gen_edge_attr_s,
    gen_edge_attr_t,
    gen_edge_index,
)
from src.data.graph import DynamicSpatialTemporalGraph


def human_tracking_to_graoup_shard(
    human_tracking_data, sorted_idxs, unique_ids, img_size, kps_norm_func
):
    node_dict = {}
    node_idxs_s = []
    node_idxs_t = {_id: [] for _id in unique_ids}
    node_idx = 0
    for t, idx in enumerate(sorted_idxs):
        inds_tmp = human_tracking_data[idx]
        node_idxs_s.append([])
        for idv in inds_tmp:
            _id = idv["id"]
            node_dict[node_idx] = (
                t,
                _id,
                np.array(idv["bbox"], dtype=np.float32)[:4],
                np.array(idv["keypoints"], dtype=np.float32)[:, :2],
            )
            node_idxs_s[t].append(node_idx)
            node_idxs_t[_id].append(node_idx)
            node_idx += 1

    bbox = [idv[2] for idv in node_dict.values()]
    bbox = np.array(bbox).reshape(-1, 2, 2)  # (-1, 2, 2)

    kps = [idv[3] for idv in node_dict.values()]
    kps = kps_norm_func(np.array(kps), bbox, img_size)  # (-1, 34, 2)

    y = [idv[1] for idv in node_dict.values()]  # id
    pos = [calc_bbox_center(idv[2]) / img_size for idv in node_dict.values()]
    time = [idv[0] for idv in node_dict.values()]  # t
    edge_index_s = gen_edge_index(node_idxs_s)
    edge_index_t = gen_edge_index(list(node_idxs_t.values()))
    edge_attr_s = gen_edge_attr_s(pos, edge_index_s)
    edge_attr_t = gen_edge_attr_t(pos, time, edge_index_t)

    return bbox, kps, y, pos, time, edge_index_s, edge_attr_s, edge_index_t, edge_attr_t


def group_pkl_to_tensor(grp_pkl, feature_type):
    bbox, kps, y, pos, time, edge_index_s, edge_attr_s, edge_index_t, edge_attr_t = (
        grp_pkl
    )
    if feature_type == "keypoints":
        x = kps
    elif feature_type == "visual":
        x = bbox
    else:
        raise ValueError

    return DynamicSpatialTemporalGraph(
        torch.tensor(x, dtype=torch.float32).contiguous(),
        torch.tensor(y, dtype=torch.long).contiguous(),
        torch.tensor(pos, dtype=torch.float32).contiguous(),
        torch.tensor(time, dtype=torch.long).contiguous(),
        torch.tensor(edge_index_s, dtype=torch.long).contiguous(),
        torch.tensor(edge_attr_s, dtype=torch.float32).contiguous(),
        torch.tensor(edge_index_t, dtype=torch.long).contiguous(),
        torch.tensor(edge_attr_t, dtype=torch.float32).contiguous(),
    )
