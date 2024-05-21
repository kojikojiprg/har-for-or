import numpy as np
import torch
import torchvision.transforms.functional as F
from numpy.typing import NDArray
from torchvision.transforms import Compose, Normalize


class NormalizeKeypoints:
    def __call__(self, kps, bbox, img_size):
        global_kps = self.global_norm(kps, img_size)
        local_kps = self.local_norm(kps, bbox)
        return np.concatenate([global_kps, local_kps], axis=2)

    def global_norm(self, kps, img_size):
        return kps / img_size

    def local_norm(self, kps, bbox):
        bbox = bbox.reshape(6, 300, 2, 2)
        kps = kps - bbox[:, :, 0].reshape(6, 300, 1, 2)
        kps /= (bbox[:, :, 1] - bbox[:, :, 0]).reshape(6, 300, 1, 2)
        return kps


class TimeSeriesToTensor:
    def __init__(self, resize_ratio=1.0):
        self.resize_ratio = resize_ratio
        self.default_float_dtype = torch.get_default_dtype()

    def __call__(self, imgs: NDArray) -> torch.Tensor:
        imgs = imgs.transpose(0, 3, 1, 2)
        imgs = torch.from_numpy(imgs).contiguous()
        if self.resize_ratio != 1.0:
            seq_len, c, h, w = imgs.shape
            new_size = (int(h * self.resize_ratio), int(w * self.resize_ratio))
            imgs = F.resize(imgs.view(-1, h, w), new_size)
            imgs = imgs.view(seq_len, c, new_size[0], new_size[1])
        if isinstance(imgs, torch.ByteTensor):
            return imgs.to(dtype=self.default_float_dtype).div(255)
        else:
            return imgs


class FrameToTensor(Compose):
    def __init__(self, resize_ratio):
        super().__init__(
            [
                TimeSeriesToTensor(resize_ratio),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


class FlowToTensor(Compose):
    def __init__(self, resize_ratio):
        super().__init__(
            [
                TimeSeriesToTensor(resize_ratio),
            ]
        )
