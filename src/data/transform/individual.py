import io

import numpy as np
import torch


def collect_human_tracking(human_tracking_data, unique_ids):
    meta = []
    ids = []
    bboxs = []
    kps = []
    for t, idvs in enumerate(human_tracking_data):
        for idv in idvs:
            i = unique_ids.index(idv["id"])
            meta.append([t, i])
            ids.append(idv["id"])
            bboxs.append(np.array(idv["bbox"], dtype=np.float32)[:4].reshape(2, 2))
            kps.append(np.array(idv["keypoints"], dtype=np.float32)[:, :2])

    return (
        np.array(meta, np.uint16),
        np.array(ids, np.uint16),
        np.array(bboxs, np.float32),
        np.array(kps, np.float32),
    )


def individual_to_npz(meta, unique_ids, frames, flows, bboxs, kps, img_size):
    h, w = frames.shape[1:3]
    seq_len, n = np.max(meta, axis=0) + 1
    frames_idvs = np.full((n, seq_len, h, w, 3), -1, dtype=np.uint8)
    flows_idvs = np.full((n, seq_len, h, w, 2), -1, dtype=np.float32)
    bboxs_idvs = np.full((n, seq_len, 2, 2), -1, dtype=np.float32)
    kps_idvs = np.full((n, seq_len, 17, 2), -1, dtype=np.float32)

    # collect data
    for (t, i), frames_i, flows_i, bboxs_i, kps_i in zip(
        meta, frames, flows, bboxs, kps
    ):
        frames_idvs[i, t] = frames_i
        flows_idvs[i, t] = flows_i
        bboxs_idvs[i, t] = bboxs_i
        kps_idvs[i, t] = kps_i

    idvs = []
    for i, _id in enumerate(unique_ids):
        data = {
            "id": _id,
            "frame": frames_idvs[i],
            "flow": flows_idvs[i],
            "bbox": bboxs_idvs[i],
            "keypoints": kps_idvs[i],
            "img_size": img_size,
        }
        idvs.append(data)
    return idvs


def individual_npz_to_tensor(
    npz, frame_transform, flow_transform, bbox_transform, kps_transform
):
    npz = np.load(io.BytesIO(npz))
    _id, frames, flows, bboxs, kps, img_size = list(npz.values())

    seq_len, h, w = frames.shape[:3]
    frames = frame_transform(frames)
    flows = flow_transform(flows)

    # NOTE: keypoints normalization is depend on raw bboxs.
    #       So that normalize keypoints first.
    kps = kps_transform(kps, bboxs)
    bboxs = bbox_transform(bboxs, img_size)

    # collect data
    return (
        torch.tensor(_id, dtype=torch.long),
        frames,
        flows,
        torch.tensor(bboxs, dtype=torch.float32).contiguous(),
        torch.tensor(kps, dtype=torch.float32).contiguous(),
    )
