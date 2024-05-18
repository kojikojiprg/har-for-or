import itertools

import numpy as np
from numpy.typing import NDArray


def normalize(x, axis=0) -> NDArray:
    l2 = np.linalg.norm(x, ord=2, axis=axis)
    if l2.ndim > 0:
        l2[l2 == 0] = 1
    else:
        if l2 == 0:
            return x
    return x / np.expand_dims(l2, axis=axis)


def calc_bbox_center(bbox) -> NDArray:
    bbox = np.array(bbox).reshape(2, 2)
    return (bbox[1] - bbox[0]) / 2 + bbox[0]


def gen_edge_index(node_idxs) -> list:
    e_i = [[idxs[i] for i in range(len(idxs))] for idxs in node_idxs]
    e_j = [[idxs[(i + 1) % len(idxs)] for i in range(len(idxs))] for idxs in node_idxs]
    e_i = list(itertools.chain.from_iterable(e_i))
    e_j = list(itertools.chain.from_iterable(e_j))
    return [e_i, e_j]


def gen_edge_attr_s(pos, edge_indexs_s) -> list:
    pos = np.array(pos)
    diffs = np.array([np.abs(pos[i] - pos[j]) for i, j in zip(*edge_indexs_s)])
    print(diffs.shape)
    # diffs /= np.array((1280, 940))
    return normalize(1 / diffs).tolist()


def gen_edge_attr_t(pos, time, edge_indexs_t) -> list:
    pos = np.array(pos)
    pos_diffs = np.array([np.abs(pos[i] - pos[j]) for i, j in zip(*edge_indexs_t)])
    tm_diffs = np.array([abs(time[i] - time[j]) for i, j in zip(*edge_indexs_t)])
    return normalize(1 / (pos_diffs * tm_diffs.reshape(-1, 1))).tolist()
