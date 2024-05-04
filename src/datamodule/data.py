from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import Data


class SpatialTemporalData(Data):
    def __init__(
        self,
        x: Optional[Tensor] = None,
        edge_index_s: Optional[Tensor] = None,
        edge_attr_s: Optional[Tensor] = None,
        edge_index_t: Optional[Tensor] = None,
        edge_attr_t: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ):
        if edge_index_s is not None and edge_index_t is not None:
            edge_index = torch.cat((edge_index_s, edge_index_t), dim=1).contiguous()
            edge_attr = torch.cat((edge_attr_s, edge_attr_t), dim=0).contiguous()
        super().__init__(x, edge_index, edge_attr, y, pos)

        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
