import argparse
import os
import pickle
import sys
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(".")
from src.utils import video, vis, yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("-v", "--version", type=int, default=0)
    args = parser.parse_args()
    data_root = args.data_root
    v = args.version

    data_dirs = sorted(glob(os.path.join(data_root, "*/")))

    # load config
    checkpoint_dir = f"models/individual/sqvae/version_{v}"
    checkpoint_path = sorted(glob(f"{checkpoint_dir}/*.ckpt"))[-1]
    config_path = f"{checkpoint_dir}/individual-sqvae.yaml"
    config = yaml_handler.load(config_path)
    seq_len = config.seq_len
    stride = config.stride

    # load model
    for data_dir in tqdm(data_dirs, ncols=100):
        if data_dir[-1] == "/":
            data_dir = data_dir[:-1]

        # load results
        paths = glob(os.path.join(data_dir, "pred", "*"))
        results = []
        for path in paths:
            with open(path, "rb") as f:
                results.append(pickle.load(f))

        # load video
        video_path = f"{data_dir}.mp4"
        cap = video.Capture(video_path)
        max_n_frame = cap.frame_count
        frame_size = cap.size

        size_heatmaps = (config.nlayers * 200, cap.size[1])
        attn_frame_size = (cap.size[0] + size_heatmaps[0], cap.size[1])

        # create writers
        wrt_kps = video.Writer(f"{data_dir}/pred_kps.mp4", cap.fps, cap.size)
        wrt_bbox = video.Writer(f"{data_dir}/pred_bbox.mp4", cap.fps, cap.size)
        wrt_cluster = video.Writer(f"{data_dir}/pred_cluster.mp4", cap.fps, cap.size)
        wrt_attn = video.Writer(
            f"{data_dir}/pred_attention.mp4", cap.fps, attn_frame_size
        )

        for n_frame in tqdm(range(cap.frame_count), ncols=100):
            _, frame = cap.read()
            if n_frame < config.seq_len:
                n_frame_result = config.seq_len
                idx_data = n_frame
            else:
                n_frame_result = seq_len + ((n_frame - seq_len) // stride + 1) * stride
                idx_data = seq_len - (n_frame_result - n_frame)

            result_tmp = [
                r for r in results if int(r["key"].split("_")[1]) == n_frame_result
            ]

            # plot kps
            frame_kps = vis.plot_kps_on_frame(
                frame.copy(), result_tmp, idx_data, frame_size
            )
            wrt_kps.write(frame_kps)

            # plot bbox
            frame_bbox = vis.plot_bbox_on_frame(
                frame.copy(), result_tmp, idx_data, frame_size
            )
            wrt_bbox.write(frame_bbox)

            # plot clustering
            frame_cluster = vis.plot_cluster_on_frame(
                frame.copy(), result_tmp, idx_data, frame_size
            )
            wrt_cluster.write(frame_cluster)

            # plot attention
            frame_attention = vis.plot_attention_on_frame(
                frame.copy(), result_tmp, idx_data, frame_size
            )
            if idx_data == seq_len - stride or n_frame == 0:
                img_heatmaps = vis.arange_attention_heatmaps(
                    result_tmp, config.n_clusters, config.nlayers, size_heatmaps
                )
                img_heatmaps = cv2.cvtColor(img_heatmaps, cv2.COLOR_RGBA2BGR)
            frame_attention = np.concatenate([frame_attention, img_heatmaps], axis=1)
            wrt_attn.write(frame_attention)

        del cap, wrt_kps, wrt_bbox, wrt_cluster, wrt_attn
