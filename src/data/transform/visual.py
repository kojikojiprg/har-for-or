import numpy as np
import torch


def frame_flow_to_visual_shard(frames, flows, frame_trans, flow_trans):
    frames = frame_trans(frames).numpy()
    flows = flow_trans(flows).numpy()

    return np.concatenate([frames, flows], axis=1)  # (seq_len, 5, h, w)


def visual_npz_to_tensor(npz):
    return torch.tensor(list(npz.values())[0], dtype=torch.float32).contiguous()
