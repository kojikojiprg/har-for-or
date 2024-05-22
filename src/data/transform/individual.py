import numpy as np
import torch


def human_tracking_to_individual_shard(
    human_tracking_data, sorted_idxs, unique_ids, img_size, kps_norm_func
):
    inds_dict = {_id: {"bbox": [], "keypoints": []} for _id in unique_ids}
    for idx in sorted_idxs:
        inds_tmp = human_tracking_data[idx]
        ids_tmp = [idv["id"] for idv in inds_tmp]
        for key_id in unique_ids:
            if key_id in ids_tmp:
                idv = [idv for idv in inds_tmp if idv["id"] == key_id][0]
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

    ids = list(inds_dict.keys())  # id

    bbox = [idv["bbox"] for idv in inds_dict.values()]
    bbox = np.array(bbox)  # (-1, seq_len, 4)
    n_id, seq_len = bbox.shape[:2]
    bbox = bbox.reshape(n_id, seq_len, 2, 2)  # (-1, seq_len, 2, 2)

    kps = [idv["keypoints"] for idv in inds_dict.values()]
    kps = np.array(kps)  # (-1, seq_len, 17, 2)
    kps = kps_norm_func(kps, bbox, img_size)  # (-1, seq_len, 34, 2)

    return ids, bbox, kps


def individual_pkl_to_tensor(idv_pkl):
    ids, bbox, kps = idv_pkl
    return (
        torch.tensor(ids, dtype=torch.long).contiguous(),
        torch.tensor(bbox, dtype=torch.float32).contiguous(),
        torch.tensor(kps, dtype=torch.float32).contiguous(),
    )
