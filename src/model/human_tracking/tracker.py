from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray

from .ext import BoostTrack


class Tracker:
    def __init__(self, cfg: SimpleNamespace, device: str):
        self.tracker = BoostTrack(**cfg.tracking.__dict__)

    def __del__(self):
        self.tracker.dump_cache()
        del self.tracker

    def update(self, bboxs: NDArray, img: NDArray):
        img_tensor = np.array([img]).transpose(0, 3, 1, 2)
        tracks = self.tracker.update(bboxs, img_tensor, img, None)
        return tracks
