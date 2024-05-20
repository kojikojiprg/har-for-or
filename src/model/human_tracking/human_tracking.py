from types import SimpleNamespace

import numpy as np
import scipy
import scipy.stats
from numpy.typing import NDArray

from .detector import Detector
from .tracker import Tracker


class HumanTracking:
    def __init__(self, config: SimpleNamespace, device: str):
        self._cfg = config
        self._device = device
        self._detector = Detector(self._cfg, self._device)
        self._tracker = Tracker(self._cfg, self._device)

    def __del__(self):
        del self._detector, self._tracker

    def reset_tracker(self):
        del self._tracker
        self._tracker = Tracker(self._cfg, self._device)

    def predict(self, frame: NDArray, frame_num: int):
        # keypoints detection
        bboxs, kps = self._detector.predict(frame)

        # tracking
        tracks = self._tracker.update(bboxs, frame)
        tracks = tracks[np.argsort(tracks[:, 4])]  # sort by track id

        # append result
        results = []
        for t in tracks:
            # get id is closed kps
            near_idxs = np.where(np.isclose(t[:4], bboxs[:, :4], atol=10.0))[0]
            if len(near_idxs) < 2:
                continue
            i = scipy.stats.mode(near_idxs).mode

            # create result
            result = {
                "n_frame": int(frame_num),
                "id": int(t[4]),
                "bbox": bboxs[i],
                "keypoints": kps[i],
            }
            results.append(result)

        return results
