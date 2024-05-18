import os
import sys
from glob import glob
from types import SimpleNamespace

import numpy as np
import torch
from torch import tensor
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from .data import SpatialTemporalData

sys.path.append("src")
from utils import json_handler


class DynamicSpatialTemporalGraphDataset(InMemoryDataset):
    def __init__(
        self,
        data_root,
        config: SimpleNamespace,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        self._feature_type = config.feature_type
        self._img_size = (config.img_size.w, config.img_size.h)
        self._config = config
        super().__init__(
            data_root, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        self.load(self.processed_paths[0], SpatialTemporalData)

    @property
    def processed_file_names(self):
        return [f"dataset_{self._feature_type}.pt"]

    def _get_feature_shape(self):
        if self._feature_type == "keypoints":
            return 17 * 2
        elif self._feature_type == "bbox":
            return 2 * 2
        elif self._feature_type == "i3d":
            raise NotImplementedError
        else:
            raise ValueError

    @staticmethod
    def _calc_bbox_center(bbox):
        bbox = np.array(bbox).reshape(2, 2)
        return (bbox[1] - bbox[0]) / 2 + bbox[0]

    def process(self):
        # get config
        seq_len = self._config.seq_len
        stride = self._config.stride
        th_cut_interval = self._config.th_cut_interval

        dirs = glob(os.path.join(self.root, "*/"))
        dirs = dirs[:1]

        data_list = []
        for dir_path in tqdm(dirs, ncols=100):
            json_path = os.path.join(dir_path, "json", "pose.json")
            pose_data_lst = json_handler.load(json_path)

            # sort data by frame and id
            pose_data_lst = sorted(pose_data_lst, key=lambda x: x["n_frame"])
            max_frame = pose_data_lst[-1]["n_frame"]
            pose_data_lst = sorted(pose_data_lst, key=lambda x: x["id"])

            # initialize sequential queue
            seq_features = []
            seq_bbox_centers = []
            seq_ids = []
            seq_times = []
            seq_n_frames = []
            n_vertax_offsets = []
            edge_index_s = []
            edge_index_t = []
            for n_frame in tqdm(range(1, max_frame + 1), leave=False, ncols=100):
                pose_data_frame = [
                    data for data in pose_data_lst if data["n_frame"] == n_frame
                ]
                if len(pose_data_frame) == 0:
                    seq_n_frames.append(n_frame)
                    if len(n_vertax_offsets) > 0:
                        n_vertax_offsets.append(n_vertax_offsets[-1])
                    else:
                        n_vertax_offsets.append(0)
                    continue

                # append data
                if self._feature_type == "keypoints":
                    features = [
                        np.array(data[self._feature_type])[:, :2].tolist()
                        for data in pose_data_frame
                    ]
                elif self._feature_type == "bbox":
                    features = [data[self._feature_type] for data in pose_data_frame]
                seq_features += features

                bbox_centers = [
                    self._calc_bbox_center(data["bbox"]).tolist()
                    for data in pose_data_frame
                ]
                seq_bbox_centers += bbox_centers

                cur_ids = [data["id"] for data in pose_data_frame]
                min_n_frame = min(seq_n_frames) if len(seq_n_frames) > 0 else 0
                cur_times = [n_frame - min_n_frame for _ in range(len(pose_data_frame))]

                # append n_vertax_offsets
                nv = len(seq_ids)
                n_vertax_offsets.append(nv)

                # add spatial edge
                adj = np.full((len(cur_ids), len(cur_ids)), 1) - np.eye(len(cur_ids))
                cur_edge_index = (np.array(np.where(adj == 1)).T + nv).tolist()
                edge_index_s += cur_edge_index

                # add temporal edge
                if n_frame > 1:
                    rev_nvos = list(reversed(n_vertax_offsets))
                    for cur_idx, cur_id in enumerate(cur_ids):
                        cur_idx = nv + cur_idx
                        cut_interval_idx = min(n_vertax_offsets[-th_cut_interval:])
                        if cur_id not in set(seq_ids[cut_interval_idx:]):
                            continue  # n_frame is the start frame of this _id
                        for j, pre_n_frame in enumerate(
                            reversed(seq_n_frames[-th_cut_interval:])
                        ):
                            pre_nvo = rev_nvos[j + 1]
                            nxt_nvo = rev_nvos[j]
                            pre_ids = seq_ids[pre_nvo:nxt_nvo]
                            if cur_id in pre_ids:
                                pre_idx = pre_ids.index(cur_id)
                                break  # get previous frame num
                        else:
                            print(seq_ids[pre_nvo:], pre_ids, cur_id)
                            print(j, pre_n_frame, pre_nvo, nxt_nvo)
                            raise ValueError  # debug

                        edge_index_t.append([pre_idx, cur_idx])

                # append sequential data
                seq_ids += cur_ids
                seq_times += cur_times
                seq_n_frames.append(n_frame)

                if n_frame < seq_len:
                    continue  # wait for saving seauential data

                if (n_frame - 1) % stride == 0:
                    # create graph
                    fs = self._get_feature_shape()
                    graph = SpatialTemporalData(
                        tensor(seq_features, dtype=torch.float).view(-1, fs),
                        tensor(seq_ids, dtype=torch.int),
                        tensor(seq_bbox_centers, dtype=torch.float).view(-1, 2),
                        tensor(seq_times, dtype=torch.int),
                        tensor(n_vertax_offsets, dtype=torch.int),
                        tensor(edge_index_s, dtype=torch.long).t().contiguous(),
                        None,
                        tensor(edge_index_t, dtype=torch.long).t().contiguous(),
                        None,
                    )
                    data_list.append(graph)

                    # get next tail index and next vertax offset
                    tail_idx = np.where(np.array(seq_n_frames) % stride == 0)[0][0] + 1
                    tail_nvo = n_vertax_offsets[tail_idx]

                    # update vertax offsets
                    n_vertax_offsets = n_vertax_offsets[tail_idx:]
                    n_vertax_offsets = list(
                        map(lambda x: x - tail_nvo, n_vertax_offsets)
                    )

                    # update edges
                    rm_idxs_s = np.any(np.array(edge_index_s) < tail_nvo, axis=1)
                    edge_index_s = np.array(edge_index_s)[~rm_idxs_s]
                    edge_index_s = (edge_index_s - tail_nvo).tolist()

                    rm_idxs_t = np.any(np.array(edge_index_t) < tail_nvo, axis=1)
                    edge_index_t = np.array(edge_index_t)[~rm_idxs_t]
                    edge_index_t = (edge_index_t - tail_nvo).tolist()

                    # update vertax feature, pos, id and n_frames in sequential data
                    seq_features = seq_features[tail_nvo:]
                    seq_bbox_centers = seq_bbox_centers[tail_nvo:]
                    seq_ids = seq_ids[tail_nvo:]
                    seq_times = (
                        np.array(seq_times[tail_nvo:]) - seq_n_frames[tail_idx - 1]
                    ).tolist()
                    seq_n_frames = seq_n_frames[tail_idx:]

            del pose_data_lst
            del seq_features, seq_bbox_centers, seq_ids, seq_n_frames
            del n_vertax_offsets, edge_index_s, edge_index_t

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if os.path.exists(self.processed_paths[0]):
            os.remove(self.processed_paths[0])
        self.save(data_list, self.processed_paths[0])
