import argparse
import os
import pickle
import sys
from glob import glob

import cv2
from tqdm import tqdm

sys.path.append(".")
from src.utils import video, vis, yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument(
        "-mt", "--model_type", required=False, type=str, default="sqvae"
    )
    args = parser.parse_args()
    data_root = args.data_root
    model_type = args.model_type

    config = yaml_handler.load("configs/individual-sqvae.yaml")
    seq_len = config.seq_len
    stride = config.stride

    # load results
    paths = glob(os.path.join(data_root, f"pred_{model_type}", "*"))
    results = []
    for path in paths:
        with open(path, "rb") as f:
            results.append(pickle.load(f))

    # load video
    video_path = f"{data_root}.mp4"
    cap = video.Capture(video_path)
    max_n_frame = cap.frame_count
    frame_size = cap.size

    # create writers
    wrt = video.Writer(f"{data_root}/vis_pose.mp4", cap.fps, cap.size)

    for n_frame in tqdm(range(cap.frame_count), desc=f"{data_root[-2:]}", ncols=100):
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

        # plot
        frame = cv2.putText(
            frame,
            f"frame:{n_frame}",
            (10, 40),
            cv2.FONT_HERSHEY_COMPLEX,
            1.0,
            (255, 255, 255),
            1,
        )
        frame = vis.plot_true_bbox_kps_on_frame(
            frame, result_tmp, idx_data, frame_size, config.range_points
        )

        wrt.write(frame)

    del cap, wrt
