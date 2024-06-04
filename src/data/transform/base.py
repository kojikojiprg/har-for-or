import numpy as np
import torch
import torchvision.transforms.functional as F
from numpy.typing import NDArray
from torchvision.transforms import Compose, Normalize


class NormalizeBbox:
    def __call__(self, bbox, img_size):
        return bbox / img_size


class NormalizeKeypoints:
    def __call__(self, kps, bbox):
        ndim = kps.ndim
        if ndim == 4:
            n_ids, seq_len = kps.shape[:2]
            bbox = bbox.reshape(n_ids, seq_len, 2, 2)
            kps = kps - bbox[:, :, 0].reshape(n_ids, seq_len, 1, 2)
            kps /= (bbox[:, :, 1] - bbox[:, :, 0]).reshape(n_ids, seq_len, 1, 2)
            kps = np.clip(kps, 0, 1)
            return kps
        elif ndim == 3:
            n = kps.shape[0]
            bbox = bbox.reshape(n, 2, 2)
            kps = kps - bbox[:, 0].reshape(n, 1, 2)
            kps /= (bbox[:, 1] - bbox[:, 0]).reshape(n, 1, 2)
            kps = np.clip(kps, 0, 1)
            return kps
        else:
            raise ValueError


class TimeSeriesToTensor:
    def __init__(self, div_by_255: bool):
        self.div_by_255 = div_by_255
        self.default_float_dtype = torch.get_default_dtype()

    def __call__(self, imgs: NDArray) -> torch.Tensor:
        imgs = torch.from_numpy(imgs.transpose(0, 3, 1, 2))
        if self.div_by_255:
            return imgs.to(dtype=self.default_float_dtype).div(255)
        else:
            return imgs


class TimeSeriesTensorResize:
    def __init__(self, resize_ratio=1.0):
        self.resize_ratio = resize_ratio

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.resize_ratio != 1.0:
            seq_len, c, h, w = imgs.shape
            new_size = (int(h * self.resize_ratio), int(w * self.resize_ratio))
            imgs = F.resize(imgs.view(-1, h, w), new_size)
            imgs = imgs.view(seq_len, c, new_size[0], new_size[1])
        return imgs


class FrameToTensor(Compose):
    def __init__(self, resize_ratio=1.0):
        super().__init__(
            [
                TimeSeriesToTensor(div_by_255=True),
                TimeSeriesTensorResize(resize_ratio),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


class FlowToTensor(Compose):
    def __init__(self, resize_ratio=1.0):
        super().__init__(
            [
                TimeSeriesToTensor(div_by_255=False),
                TimeSeriesTensorResize(resize_ratio),
                Normalize([0.0, 0.0], [3.0, 3.0]),
            ]
        )
