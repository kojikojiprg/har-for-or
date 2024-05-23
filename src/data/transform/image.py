import io

import cv2
import numpy as np


def clip_images_by_bbox(frames, flows, human_tracking_data, resize):
    idv_frames = []
    idv_flows = []
    for t, idvs in enumerate(human_tracking_data):
        for idv in idvs:
            x1, y1, x2, y2 = list(map(int, idv["bbox"][:4]))
            idv_frames.append(cv2.resize(frames[t, y1:y2, x1:x2], resize))
            idv_flows.append(cv2.resize(flows[t, y1:y2, x1:x2], resize))

    return np.array(idv_frames), np.array(idv_flows)


def images_to_tensor(npy, transform):
    npy = np.lib.format.read_array(io.BytesIO(npy))
    return transform(npy)
