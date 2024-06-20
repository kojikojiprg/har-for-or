# import os
# from glob import glob
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm
from webdataset import WebLoader

sys.path.append(".")
from src.data import load_dataset
from src.model import IndividualActivityRecognition
from src.utils import video, vis, yaml_handler

if __name__ == "__main__":
    data_root = "../datasets/dataset03/train/01"
    # data_dirs = sorted(glob(os.path.join(data_root, "*/")))
    data_dirs = [data_root]
    data_dir = data_dirs[0]

    checkpoint_path = (
        "models/individual/individual-seq_len90-stride30-256x192-loss-min-v2.ckpt"
    )
    config = yaml_handler.load("configs/individual.yaml")
    seq_len = config.seq_len
    stride = config.stride
    gpu_id = 0
    device = f"cuda:{gpu_id}"

    # load dataset
    dataset, n_samples = load_dataset(data_dirs, "individual", config, False)
    dataloader = WebLoader(dataset, num_workers=config.num_workers, pin_memory=True)

    # load model
    model = IndividualActivityRecognition(config)
    model.configure_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    # load video
    video_path = f"{data_dir}.mp4"
    cap = video.Capture(video_path)
    max_n_frame = cap.frame_count
    img_size = cap.size

    # create writers
    wrt_x_vis = video.Writer(f"{data_dir}/pred_x_vis.mp4", cap.fps, cap.size)
    wrt_x_spc = video.Writer(f"{data_dir}/pred_x_spc.mp4", cap.fps, cap.size)
    wrt_cluster = video.Writer(f"{data_dir}/pred_cluster.mp4", cap.fps, cap.size)

    # pred
    mse_x_vis_dict = {}
    mse_x_spc_dict = {}
    latent_features = {"label": [], "mu": [], "logvar": [], "z": []}

    model.eval()
    with torch.no_grad():
        pre_n_frame = seq_len
        results_tmp = []
        for batch in tqdm(dataloader, total=n_samples):
            result = model.predict_step(batch)
            n_frame = int(result["key"].split("_")[1])
            _id = result["id"]

            # collect mse
            if _id not in mse_x_vis_dict:
                mse_x_vis_dict[_id] = {}
                mse_x_spc_dict[_id] = {}
            mse_x_vis_dict[_id][n_frame] = result["mse_x_vis"]
            mse_x_spc_dict[_id][n_frame] = result["mse_x_spc"]

            # collect latent features
            latent_features["label"].append(np.argmax(result["y"]).item())
            latent_features["mu"].append(result["mu"])
            latent_features["logvar"].append(result["logvar"])

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

                    frame_vis = vis.plot_on_frame(
                        frame.copy(), results_tmp, idx_data, img_size, "x_vis"
                    )
                    wrt_x_vis.write(frame_vis)
                    frame_spc = vis.plot_on_frame(
                        frame.copy(), results_tmp, idx_data, img_size, "x_spc"
                    )
                    wrt_x_spc.write(frame_spc)
                    frame_cluster = vis.plot_on_frame(
                        frame.copy(), results_tmp, idx_data, img_size, "cluster"
                    )
                    wrt_cluster.write(frame_cluster)

                results_tmp = []
                pre_n_frame = n_frame

            # add result in temporary result list
            results_tmp.append(result)

    del cap, wrt_x_vis, wrt_x_spc, wrt_cluster

    # plot mse
    vis.plot_mse(mse_x_vis_dict, max_n_frame, stride, f"{data_dir}/pred_x_vis.jpg")
    vis.plot_mse(mse_x_spc_dict, max_n_frame, stride, f"{data_dir}/pred_x_spc.jpg")

    # plot latent feature
    X = np.array(latent_features["mu"])
    labels = np.array(latent_features["label"])
    vis.plot_tsne(X, labels, f"{data_dir}/pred_mu_tsne.jpg")
