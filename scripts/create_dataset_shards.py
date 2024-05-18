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
    args = parser.parse_args()

    video_paths = sorted(glob(os.path.join(args.data_root, "*.mp4")))

    config = yaml_handler.load(args.config_path)

    for video_path in video_paths:
        create_shards(video_path, config)
