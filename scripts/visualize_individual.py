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
    parser.add_argument(
        "-mt", "--model_type", required=False, type=str, default="sqvae"
    )
    parser.add_argument("-v", "--version", type=int, default=0)
    args = parser.parse_args()
    data_root = args.data_root
    model_type = args.model_type
    v = args.version

    data_dirs = sorted(glob(os.path.join(data_root, "*/")))

    # load config
    checkpoint_dir = f"models/individual/{model_type}/version_{v}"
    checkpoint_path = sorted(glob(f"{checkpoint_dir}/*.ckpt"))[-1]
    config_path = f"{checkpoint_dir}/individual-{model_type}.yaml"
    config = yaml_handler.load(config_path)
    seq_len = config.seq_len
    stride = config.stride
    range_points = config.range_points

    # load model
    for data_dir in tqdm(data_dirs, ncols=100):
        if data_dir[-1] == "/":
            data_dir = data_dir[:-1]

        # load results
        paths = glob(os.path.join(data_dir, f"pred_{model_type}", "*"))
        results = []
        for path in paths:
            with open(path, "rb") as f:
                results.append(pickle.load(f))

        # load video
        video_path = f"{data_dir}.mp4"
        cap = video.Capture(video_path)
        max_n_frame = cap.frame_count
        frame_size = cap.size

        size_heatmap_attn = (config.nlayers * 200, cap.size[1])
        attn_frame_size = (cap.size[0] + size_heatmap_attn[0], cap.size[1])
        size_heatmap_attn_cls = (200, cap.size[1])
        attn_cls_frame_size = (cap.size[0] + size_heatmap_attn_cls[0], cap.size[1])
        size_heatmap_book = (400, cap.size[1])
        book_frame_size = (cap.size[0] + size_heatmap_book[0], cap.size[1])

        # create writers
        wrt_kps = video.Writer(f"{data_dir}/pred_kps.mp4", cap.fps, cap.size)
        wrt_bbox = video.Writer(f"{data_dir}/pred_bbox.mp4", cap.fps, cap.size)
        wrt_cluster = video.Writer(f"{data_dir}/pred_cluster.mp4", cap.fps, cap.size)
        wrt_attn = video.Writer(
            f"{data_dir}/pred_attention.mp4", cap.fps, attn_frame_size
        )
        wrt_attn_cls = video.Writer(
            f"{data_dir}/pred_attention_clustering.mp4", cap.fps, attn_cls_frame_size
        )
        wrt_book = video.Writer(
            f"{data_dir}/pred_book_indices.mp4", cap.fps, book_frame_size
        )

        for n_frame in tqdm(range(cap.frame_count), desc=f"{data_dir[-2:]}", ncols=100):
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

            # put frame number
            frame = cv2.putText(
                frame,
                f"frame:{n_frame}",
                (10, 40),
                cv2.FONT_HERSHEY_COMPLEX,
                1.0,
                (255, 255, 255),
                1,
            )

            if len(result_tmp) != 0:
                # plot kps
                frame_kps = vis.plot_kps_on_frame(
                    frame.copy(), result_tmp, idx_data, frame_size, range_points
                )

                # plot bbo
                frame_bbox = vis.plot_bbox_on_frame(
                    frame.copy(), result_tmp, idx_data, frame_size, range_points
                )

                # plot cluster
                frame_cluster = vis.plot_cluster_on_frame(
                    frame.copy(), result_tmp, idx_data, frame_size, range_points
                )

                # plot attention
                frame_attn = vis.plot_attention_on_frame(
                    frame.copy(), result_tmp, idx_data, frame_size, range_points
                )
                if idx_data == seq_len - stride or n_frame == 0:
                    img_heatmaps_attn = vis.arange_attention_heatmaps(
                        result_tmp, config.n_clusters, config.nlayers, size_heatmap_attn
                    )
                    img_heatmaps_attn = cv2.cvtColor(
                        img_heatmaps_attn, cv2.COLOR_RGBA2BGR
                    )
                frame_attn = np.concatenate([frame_attn, img_heatmaps_attn], axis=1)

                # plot attention
                frame_attn_cls = vis.plot_attention_clustering_on_frame(
                    frame.copy(), result_tmp, idx_data, frame_size, range_points
                )
                if idx_data == seq_len - stride or n_frame == 0:
                    img_heatmaps_attn_cls = vis.arange_attention_clustering_heatmaps(
                        result_tmp, config.n_clusters, size_heatmap_attn_cls
                    )
                    img_heatmaps_attn_cls = cv2.cvtColor(
                        img_heatmaps_attn_cls, cv2.COLOR_RGBA2BGR
                    )
                frame_attn_cls = np.concatenate(
                    [frame_attn_cls, img_heatmaps_attn_cls], axis=1
                )

                # plot book indices
                frame_book = vis.plot_book_idx_on_frame(
                    frame.copy(),
                    result_tmp,
                    idx_data,
                    frame_size,
                    config.book_size,
                    range_points,
                )
                if idx_data == seq_len - stride or n_frame == 0:
                    img_heatmaps_book = vis.arange_book_idx_heatmaps(
                        result_tmp,
                        config.n_clusters,
                        size_heatmap_book,
                        config.book_size,
                    )
                    img_heatmaps_book = cv2.cvtColor(
                        img_heatmaps_book, cv2.COLOR_RGBA2BGR
                    )
                frame_book = np.concatenate([frame_book, img_heatmaps_book], axis=1)
            else:
                frame_kps = frame
                frame_bbox = frame
                frame_cluster = frame
                frame_attn = frame
                frame_attn_cls = frame
                frame_book = frame

            wrt_kps.write(frame_kps)
            wrt_bbox.write(frame_bbox)
            wrt_cluster.write(frame_cluster)
            wrt_attn.write(frame_attn)
            wrt_attn_cls.write(frame_attn_cls)
            wrt_book.write(frame_book)

        del cap, wrt_kps, wrt_bbox, wrt_cluster, wrt_attn, wrt_attn_cls, wrt_book
