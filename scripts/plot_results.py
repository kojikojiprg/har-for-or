import argparse
import os
import pickle
import sys
from glob import glob

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.append(".")
from src.model import SQVAE
from src.utils import vis, yaml_handler

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

    checkpoint_dir = f"models/individual/{model_type}/version_{v}"
    checkpoint_path = sorted(glob(f"{checkpoint_dir}/*.ckpt"))[-1]

    # load classes
    path = f"{data_root}/annotation/classes.txt"
    classes = np.loadtxt(path, str, usecols=0, delimiter=",")
    classes = [c.title() for c in classes]

    # load config
    config = yaml_handler.load(f"{checkpoint_dir}/individual-{model_type}.yaml")
    seq_len = config.seq_len
    stride = config.stride
    if config.mask_leg:
        n_pts = 13 + 2
    else:
        n_pts = 17 + 2

    # create image dir
    img_dir = f"{data_root}/images"
    os.makedirs(img_dir, exist_ok=True)

    # ====================
    # Test
    # ====================
    print("plotting confusion matrix")
    # load annotation
    path = glob(f"{data_root}/annotation/*test*.txt")[0]
    video_num = int(os.path.basename(path).split(".")[0].split("_")[2])
    annotations = np.loadtxt(path, str, skiprows=1, delimiter=" ")

    # load preds
    data_root_test = f"{data_root}/test/{video_num:02d}"
    paths = glob(os.path.join(data_root_test, f"pred_{model_type}", "*"))
    results = []
    for path in paths:
        with open(path, "rb") as f:
            results.append(pickle.load(f))

    label_preds = []
    label_gts = []
    ann_keys = annotations.T[0]
    for result in results:
        key = result["key"]
        video_num, n_frame, label = key.split("_")
        n_frame = int(n_frame)
        key = f"{video_num}_{label}"
        label_pred = result["label"]
        label_preds.append(label_pred)

        ann_tmp = annotations[ann_keys == key]
        if len(ann_tmp) >= 2:
            # groud truth is chenged sometimes since the tracking model mistakes
            ann_n_frames = ann_tmp.T[2].astype(int)
            ann_tmp = ann_tmp[ann_n_frames <= n_frame][-1]
            label_gt = int(ann_tmp[1])
        else:
            label_gt = int(ann_tmp[0, 1])
        label_gts.append(label_gt)

    cm = confusion_matrix(label_gts, label_preds).T
    figpath = f"{img_dir}/cm_test_{video_num}.png"
    vis.plot_cm(cm, classes, figpath, normalize=True)

    path = f"{img_dir}/cm_test_report.tsv"
    report = classification_report(
        label_gts, label_preds, digits=3, output_dict=True, zero_division=0
    )
    pd.DataFrame.from_dict(report).T.to_csv(path, sep="\t")

    # ====================
    # tSNE of codebooks
    # ====================
    print("plotting scatter of codebooks using tSNE")
    model = SQVAE(config)
    model.configure_model()
    checkpoint_path = sorted(glob(f"{checkpoint_dir}/*.ckpt"))[-1]
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    books = {i: book.detach().numpy() for i, book in enumerate(model.quantizer.books)}
    X = np.array([book for book in books.values()]).reshape(-1, config.latent_ndim)
    labels = np.array(
        [[c for _ in range(config.book_size)] for c in range(config.n_clusters)]
    ).ravel()
    path = f"{img_dir}/t-sen_codebooks.jpg"
    vis.plot_tsne(X, labels, classes, 10, path, True, cmap="tab10")

    # ====================
    # Plot Graphs
    # ====================
    print("plotting graphs")
    data_dirs = glob(f"{data_root}/train/*/") + glob(f"{data_root}/test/*/")
    data_dirs = sorted(
        data_dirs, key=lambda x: int(os.path.basename(os.path.dirname(x)))
    )  # sort by video_num

    # create image dirs
    mse_kps_dir = f"{img_dir}/mse_kps"
    os.makedirs(mse_kps_dir, exist_ok=True)
    mse_bbox_dir = f"{img_dir}/mse_bbox"
    os.makedirs(mse_bbox_dir, exist_ok=True)
    mse_cls_dir = f"{img_dir}/clustering"
    os.makedirs(mse_cls_dir, exist_ok=True)

    for data_dir in tqdm(data_dirs, ncols=100):
        video_num = os.path.basename(os.path.dirname(data_dir))

        # load preds
        paths = glob(os.path.join(data_dir, f"pred_{model_type}", "*"))
        results = []
        for path in paths:
            with open(path, "rb") as f:
                results.append(pickle.load(f))

        # collect preds
        mse_kps_dict = {}
        mse_bbox_dict = {}
        label_counts = {i: {} for i in range(config.n_clusters)}
        max_n_frame = 0
        for result in results:
            key = result["key"]
            n_frame = int(key.split("_")[1])
            if max_n_frame < n_frame:
                max_n_frame = n_frame
            label = result["id"]

            # collect mse
            if label not in mse_kps_dict:
                mse_kps_dict[label] = {}
                mse_bbox_dict[label] = {}
            mse_kps_dict[label][n_frame] = result["mse_kps"]
            mse_bbox_dict[label][n_frame] = result["mse_bbox"]

            # label count
            label_pred = int(result["label"])
            if n_frame not in label_counts[label_pred]:
                label_counts[label_pred][n_frame] = 0
            label_counts[label_pred][n_frame] += 1

        mse_kps_figpath = f"{mse_kps_dir}/mse_kps_{video_num}.png"
        vis.plot_mse(
            mse_kps_dict,
            max_n_frame,
            stride,
            0.015,
            "MSE of Keypoints",
            10,
            mse_kps_figpath,
            False,
            ylim=(0, 0.1),
        )

        mse_bbox_figpath = f"{mse_bbox_dir}/mse_bbox_{video_num}.png"
        vis.plot_mse(
            mse_bbox_dict,
            max_n_frame,
            stride,
            0.01,
            "MSE of Bbox",
            10,
            mse_bbox_figpath,
            False,
            ylim=(0, 0.02),
        )

        mse_cls_figpath = f"{mse_cls_dir}/label_ratio_cumsum_{video_num}.png"
        vis.plot_label_ratio_cumsum(
            label_counts, classes, max_n_frame, stride, 30, mse_cls_figpath, False
        )
