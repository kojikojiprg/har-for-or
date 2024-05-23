import io
import itertools

import numpy as np
import torch


def collect_human_tracking(human_tracking_data):
    unique_ids = set(
        itertools.chain.from_iterable(
            [[idv["id"] for idv in idvs] for idvs in human_tracking_data]
        )
    )
    unique_ids = sorted(list(unique_ids))
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


def individual_npz_to_tensor(
    npz, frame_transform, flow_transform, bbox_transform, kps_transform
):
    npz = np.load(io.BytesIO(npz))
    meta, ids, frames, flows, bboxs, kps, img_size = list(npz.values())

    h, w = frames.shape[1:3]
    seq_len, n = np.max(meta, axis=0) + 1
    unique_ids = sorted(list(set(ids)))
    t_frames = torch.full((n, seq_len, 3, h, w), -1, dtype=torch.float32)
    t_flows = torch.full((n, seq_len, 2, h, w), -1, dtype=torch.float32)
    t_bboxs = torch.full((n, seq_len, 2, 2), -1, dtype=torch.float32)
    t_kps = torch.full((n, seq_len, 17, 2), -1, dtype=torch.float32)

    frames = frame_transform(frames)
    flows = flow_transform(flows)

    # NOTE: keypoints normalization is depend on raw bboxs.
    #       So that normalize keypoints first.
    kps = kps_transform(kps, bboxs)
    bboxs = bbox_transform(bboxs, img_size)

    # collect data
    for (t, i), frames_i, flows_i, bboxs_i, kps_i in zip(
        meta, frames, flows, bboxs, kps
    ):
        t_frames[i, t] = frames_i
        t_flows[i, t] = flows_i
        t_bboxs[i, t] = torch.tensor(bboxs_i).contiguous()
        t_kps[i, t] = torch.tensor(kps_i).contiguous()

    return (
        torch.tensor(unique_ids, dtype=torch.long).contiguous(),
        t_frames,
        t_flows,
        t_bboxs,
        t_kps,
    )
