import argparse
import os
import pickle
import sys
from glob import glob

import cv2
import torch
from tqdm import tqdm
from webdataset import WebLoader

sys.path.append(".")
from src.data import load_dataset
from src.model import VAE
from src.utils import video, vis, yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("-v", "--version", type=int, default=0)
    parser.add_argument("-g", "--gpu_id", type=int, default=None)
    args = parser.parse_args()
    data_root = args.data_root
    v = args.version
    ep = args.epoch
    gpu_id = args.gpu_id
    device = f"cuda:{gpu_id}"

    data_dirs = sorted(glob(os.path.join(data_root, "*/")))

    checkpoint_dir = f"models/individual/vae/version_{v}"
    checkpoint_path = sorted(glob(f"{checkpoint_dir}/*.ckpt"))[-1]

    # load config
    config_path = f"{checkpoint_dir}/individual-vae.yaml"
    config = yaml_handler.load(config_path)
    seq_len = config.seq_len
    stride = config.stride

    # load model
    model = VAE(config)
    model.configure_model()
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    for data_dir in tqdm(data_dirs, ncols=100):
        if data_dir[-1] == "/":
            data_dir = data_dir[:-1]

        # load dataset
        dataset, n_samples = load_dataset([data_dir], "individual", config, False)
        dataloader = WebLoader(dataset, num_workers=1, pin_memory=True)

        # load video
        video_path = f"{data_dir}.mp4"
        cap = video.Capture(video_path)
        max_n_frame = cap.frame_count
        frame_size = cap.size

        # create writers
        wrt_kps = video.Writer(f"{data_dir}/pred_kps.mp4", cap.fps, cap.size)
        wrt_bbox = video.Writer(f"{data_dir}/pred_bbox.mp4", cap.fps, cap.size)
        wrt_cluster = video.Writer(f"{data_dir}/pred_cluster.mp4", cap.fps, cap.size)

        # pred
        save_dir = os.path.join(data_dir, "pred")
        os.makedirs(save_dir, exist_ok=True)

        model.eval()
        pre_n_frame = seq_len
        results_tmp = []
        for batch in tqdm(dataloader, total=n_samples):
            results = model.predict_step(batch)
            for result in results:
                n_frame = int(result["key"].split("_")[1])
                key = result["key"]
                _id = result["id"]

                path = os.path.join(save_dir, f"{key}.pkl")
                with open(path, "wb") as f:
                    pickle.dump(result, f)

                # plot bboxs
                if pre_n_frame < n_frame:
                    for i in range(stride):
                        n_frame_tmp = pre_n_frame - stride + i
                        idx_data = seq_len - stride + i

                        frame = cap.read(n_frame_tmp)[1]
                        frame = cv2.putText(
                            frame,
                            f"frame:{n_frame_tmp}",
                            (10, 40),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1.0,
                            (255, 255, 255),
                            1,
                        )

                        frame_kps = vis.plot_kps_on_frame(
                            frame.copy(), results_tmp, idx_data, frame_size
                        )
                        wrt_kps.write(frame_kps)
                        frame_bbox = vis.plot_bbox_on_frame(
                            frame.copy(), results_tmp, idx_data, frame_size
                        )
                        wrt_bbox.write(frame_bbox)
                        frame_cluster = vis.plot_cluster_on_frame(
                            frame.copy(), results_tmp, idx_data, frame_size
                        )
                        wrt_cluster.write(frame_cluster)

                    results_tmp = []
                    pre_n_frame = n_frame

                # add result in temporary result list
                results_tmp.append(result)

    del cap, wrt_kps, wrt_bbox, wrt_cluster
