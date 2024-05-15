from types import SimpleNamespace
from typing import List, Tuple, Union

import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform("global_norm")
class GlobalNorm(BaseTransform):
    def __init__(
        self,
        img_size: Union[Tuple[int, int], SimpleNamespace],
        attrs: List[str] = ["x", "edge_attr_s", "edge_attr_t"],
    ):
        if isinstance(img_size, tuple):
            self.img_size = torch.tensor(img_size, dtype=torch.int)
        else:
            self.img_size = torch.tensor((img_size.w, img_size.h), dtype=torch.int)
        self.attrs = attrs

    def forward(self, data: Data) -> Data:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                store[key] = value / self.img_size

        return data


@functional_transform("local_norm")
class LocalNorm(BaseTransform):
    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def forward(self, data: Data) -> Data:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if value.numel() > 0:
                    value = value - value.min()
                    value.div_(value.sum(dim=-1, keepdim=True).clamp_(min=1.0))
                    store[key] = value
        return data


@functional_transform("spatial_edge_distance")
class SpatialEdgeDistance(BaseTransform):
    def __init__(self, attrs: List[str] = ["edge_index_s"]):
        for a in attrs:
            assert a.startswith("edge_index")
        self.attrs = attrs

    def forward(self, data: Data) -> Data:
        for store in data.stores:
            pos = store.pos
            for index_key, value in store.items(*self.attrs):
                pos_i = pos[value[0]]
                pos_j = pos[value[1]]
                attr_key = index_key.replace("index", "attr")
                store[attr_key] = torch.abs(pos_i - pos_j)

        return data
