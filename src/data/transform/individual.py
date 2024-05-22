import itertools
import pickle

import numpy as np
import torch


def individual_pkl_to_tensor(pkl, bbox_transform, kps_transform):
    human_tracking_data, img_size = pickle.loads(pkl)

    unique_ids = set(
        itertools.chain.from_iterable(
            [[ind["id"] for ind in inds] for inds in human_tracking_data]
        )
    )
    unique_ids = list(unique_ids)

    # collect data
    inds_dict = {_id: {"bbox": [], "keypoints": []} for _id in unique_ids}
    for idvs in human_tracking_data:
        ids_tmp = [idv["id"] for idv in idvs]
        for key_id in unique_ids:
            if key_id in ids_tmp:
                idv = [idv for idv in idvs if idv["id"] == key_id][0]
                inds_dict[key_id]["bbox"].append(
                    np.array(idv["bbox"], dtype=np.float32)[:4]
                )
                inds_dict[key_id]["keypoints"].append(
                    np.array(idv["keypoints"], dtype=np.float32)[:, :2]
                )
            else:
                # append dmy
                inds_dict[key_id]["bbox"].append(np.full((4,), -1, dtype=np.float32))
                inds_dict[key_id]["keypoints"].append(
                    np.full((17, 2), -1, dtype=np.float32)
                )

    bbox = [idv["bbox"] for idv in inds_dict.values()]
    bbox = np.array(bbox)  # (-1, seq_len, 4)
    n_id, seq_len = bbox.shape[:2]
    bbox = bbox.reshape(n_id, seq_len, 2, 2)  # (-1, seq_len, 2, 2)
    bbox = bbox_transform(bbox, img_size)

    kps = [idv["keypoints"] for idv in inds_dict.values()]
    kps = np.array(kps)  # (-1, seq_len, 17, 2)
    kps = kps_transform(kps, bbox, img_size)

    return (
        torch.tensor(unique_ids, dtype=torch.long).contiguous(),
        torch.tensor(bbox, dtype=torch.float32).contiguous(),
        torch.tensor(kps, dtype=torch.float32).contiguous(),
    )
