import sys

import numpy as np
import torch
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import SelectOutput

sys.path.append("src")
from data import SpatialTemporalData


class TemporalPooling(torch.nn.Module):
    def __init__(self, seq_len: int, stride: int):
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride

    def forward(self, data: SpatialTemporalData):
        time = data.time.cpu().numpy()
        max_n_frames = np.max(time)

        data_list = []
        for start_n_frame in range(1, max_n_frames, self.stride):
            end_n_frame = start_n_frame + self.seq_len
            node_index = np.where((start_n_frame <= time) & (time < end_n_frame))[0]
            select_out = SelectOutput(node_index, data.x.size(0), node_index, None)
            if hasattr(data, "edge_attr_t"):
                connect_out = FilterEdges(select_out, data.edge_index_t, data.edge_attr_t)
            else:
                connect_out = FilterEdges(select_out, data.edge_index_t, None)

            pooled_data = SpatialTemporalData(
                data.x[node_index],
                data.y[node_index],
                data.pos[node_index],
                data.time[node_index],
                edge_index_t=connect_out.edge_index,
                edge_attr_t=connect_out.edge_attr,
            )
            data_list.append(pooled_data)

        return data_list
