import numpy as np
import torch
from numpy.typing import NDArray
from torchvision.transforms import Compose, Normalize

from .functional import normalize


class TimeSeriesToTensor:
    def __init__(self):
        self.default_float_dtype = torch.get_default_dtype()

    def __call__(self, imgs: NDArray) -> torch.Tensor:
        imgs = imgs.transpose(0, 3, 1, 2)
        imgs = torch.from_numpy(imgs).contiguous()
        if isinstance(imgs, torch.ByteTensor):
            return imgs.to(dtype=self.default_float_dtype).div(255)
        else:
            return imgs


class NormalizeX:
    def __init__(self, norm_type):
        self.norm_type = norm_type

    def __call__(self, x):
        if self.norm_type == "local":
            return self.local_norm(x)
        elif self.norm_type == "global":
            return self.global_norm(x)

    def local_norm(self, x):
        return normalize(x).tolist()

    def global_norm(self, x):
        return (np.array(x) / self.img_size).tolist()


class FrameToTensor(Compose):
    def __init__(self):
        super().__init__(
            [
                TimeSeriesToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


class FlowToTensor(Compose):
    def __init__(self):
        super().__init__(
            [
                TimeSeriesToTensor(),
            ]
        )
