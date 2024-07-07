import io

import numpy as np
import torch
from scipy import interpolate


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


def individual_to_npz(
    meta, unique_ids, frames, flows, bboxs, kps, frame_size, th_nan_ratio=0.3
):
    h, w = frames.shape[1:3]
    seq_len, n = np.max(meta, axis=0) + 1
    frames_idvs = np.full((n, seq_len, h, w, 3), 0, dtype=np.uint8)
    flows_idvs = np.full((n, seq_len, h, w, 2), -1e10, dtype=np.float32)
    bboxs_idvs = np.full((n, seq_len, 2, 2), -1e10, dtype=np.float32)
    kps_idvs = np.full((n, seq_len, 17, 2), -1e10, dtype=np.float32)

    # collect data
    for (t, i), frames_i, flows_i, bboxs_i, kps_i in zip(
        meta, frames, flows, bboxs, kps
    ):
        frames_idvs[i, t] = frames_i
        flows_idvs[i, t] = flows_i
        bboxs_idvs[i, t] = bboxs_i
        kps_idvs[i, t] = kps_i

    # cleansing
    unique_ids, frames_idvs, flows_idvs, bboxs_idvs, kps_idvs = cleansing_individual(
        unique_ids, frames_idvs, flows_idvs, bboxs_idvs, kps_idvs, th_nan_ratio
    )

    idvs = []
    for i, _id in enumerate(unique_ids):
        data = {
            "id": _id,
            "frame": frames_idvs[i],
            "flow": flows_idvs[i],
            "bbox": bboxs_idvs[i],
            "keypoints": kps_idvs[i],
            "frame_size": frame_size,  # (h, w)
        }
        idvs.append(data)
    return idvs, unique_ids


def cleansing_individual(
    unique_ids, frames_idvs, flows_idvs, bboxs_idvs, kps_idvs, th_nan_ratio=0.3
):
    unique_ids = np.array(unique_ids)
    n, seq_len = bboxs_idvs.shape[:2]
    # delete items whose first and last are nan
    mask_not_nan = np.all(bboxs_idvs[:, 0] >= 0, axis=(1, 2)) & np.all(
        bboxs_idvs[:, -1] >= 0, axis=(1, 2)
    )
    if np.count_nonzero(mask_not_nan) < n:
        unique_ids = unique_ids[mask_not_nan]
        frames_idvs = frames_idvs[mask_not_nan]
        flows_idvs = flows_idvs[mask_not_nan]
        bboxs_idvs = bboxs_idvs[mask_not_nan]
        kps_idvs = kps_idvs[mask_not_nan]

    if len(bboxs_idvs) == 0:
        return unique_ids, frames_idvs, flows_idvs, bboxs_idvs, kps_idvs

    # delete sample with high proportion of nan
    n, seq_len = bboxs_idvs.shape[:2]
    nan = bboxs_idvs < 0
    nan_ratio = np.count_nonzero(nan.reshape(n, -1), axis=1) / (seq_len * 2 * 2)
    mask_lower_nan_ratio = nan_ratio < th_nan_ratio
    unique_ids = unique_ids[mask_lower_nan_ratio]
    frames_idvs = frames_idvs[mask_lower_nan_ratio]
    flows_idvs = flows_idvs[mask_lower_nan_ratio]
    bboxs_idvs = bboxs_idvs[mask_lower_nan_ratio]
    kps_idvs = kps_idvs[mask_lower_nan_ratio]

    return unique_ids, frames_idvs, flows_idvs, bboxs_idvs, kps_idvs


def individual_npz_to_tensor(
    sample,
    seq_len,
    frame_transform,
    flow_transform,
    bbox_transform,
    kps_transform,
    interpolate=False,
):
    key = sample["__key__"]
    npz = list(np.load(io.BytesIO(sample["npz"])).values())
    _id, frames, flows, bboxs, kps, frame_size = npz

    if len(bboxs) < seq_len:
        # padding
        pad_shape = ((0, seq_len - len(bboxs)), (0, 0), (0, 0), (0, 0))
        # frames = np.pad(frames, pad_shape, constant_values=-1)
        # flows = np.pad(flows, pad_shape, constant_values=-1e10)
        pad_shape = ((0, seq_len - len(bboxs)), (0, 0), (0, 0))
        bboxs = np.pad(bboxs, pad_shape, constant_values=-1e10)
        kps = np.pad(kps, pad_shape, constant_values=-1e10)

    # interpolate
    if interpolate:
        frames, flows, bboxs, kps = interpolate_individual(frames, flows, bboxs, kps)

    # frames = frame_transform(frames)
    # flows = flow_transform(flows)
    # pixcels = torch.cat([frames, flows], dim=1).to(torch.float32)

    mask = torch.from_numpy(np.any(bboxs < 0, axis=(1, 2))).to(torch.bool)

    kps[~mask] = kps_transform(kps[~mask], bboxs[~mask])
    kps = torch.from_numpy(kps).to(torch.float32)

    bboxs[~mask] = bbox_transform(bboxs[~mask], frame_size[::-1])  # frame_size: (h, w)
    bboxs = torch.from_numpy(bboxs).to(torch.float32)

    del sample, npz, frames, flows  # release memory

    # return key, _id, pixcels, bboxs, mask
    return key, _id, kps, bboxs, mask


def interpolate_individual(frames, flows, bboxs, kps):
    n, seq_len, h, w = frames.shape[:4]

    mask_not_nan = np.all(bboxs > 0, axis=(2, 3))
    frames = _interpolate(frames, mask_not_nan).reshape(n, seq_len, h, w, 3)
    flows = _interpolate(flows, mask_not_nan).reshape(n, seq_len, h, w, 2)
    bboxs = _interpolate(bboxs, mask_not_nan).reshape(n, seq_len, 2, 2)
    kps = _interpolate(kps, mask_not_nan).reshape(n, seq_len, 17, 2)

    return frames, flows, bboxs, kps


def _interpolate(y, mask):
    n, seq_len = y.shape[:2]
    new_y = y.copy().reshape(n, seq_len, -1)
    for i in range(len(y)):
        x = np.arange(seq_len)[mask[i]]
        vals = y[i].reshape(seq_len, -1)
        curve = interpolate.interp1d(x, vals[mask[i]], kind="cubic", axis=0)
        new_y[i, ~mask[i]] = curve(np.arange(seq_len))[~mask[i]].astype(y.dtype)

    return new_y
