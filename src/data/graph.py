from typing import Optional

from torch import Tensor
from torch_geometric.data import Data


class DynamicSpatialTemporalGraph(Data):
    def __init__(
        self,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
        edge_index_s: Optional[Tensor] = None,
        edge_attr_s: Optional[Tensor] = None,
        edge_index_t: Optional[Tensor] = None,
        edge_attr_t: Optional[Tensor] = None,
    ):
        super().__init__(x=x, y=y, pos=pos, time=time)
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
