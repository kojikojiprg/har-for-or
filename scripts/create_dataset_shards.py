import argparse
import os
import sys
from glob import glob

sys.path.append("src")
from data import create_shards
from utils import yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument(
        "-c", "--config_path", type=str, required=False, default="configs/dataset.yaml"
    )
    parser.add_argument(
        "-cht",
        "--config_human_tracking_path",
        type=str,
        required=False,
        default="configs/human_tracking.yaml",
    )
    parser.add_argument("-np", "--n_processes", type=int, required=False, default=32)

    parser.add_argument("-g", "--gpu", type=int, required=False, default=1)
    args = parser.parse_args()

    video_paths = sorted(glob(os.path.join(args.data_root, "*.mp4")))

    config = yaml_handler.load(args.config_path)
    config_ht = yaml_handler.load(args.config_human_tracking_path)
    device = f"cuda:{args.gpu}"
    n_processes = args.n_processes

    for video_path in video_paths[:1]:
        create_shards(video_path, config, config_ht, device, n_processes)
