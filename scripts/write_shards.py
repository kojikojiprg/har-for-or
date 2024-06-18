import argparse
import os
import sys
from glob import glob

sys.path.append(".")
from tqdm import tqdm

from src.data import write_shards
from src.model import HumanTracking
from src.utils import yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("dataset_type", type=str, help="'individual' or 'group'")

    # optional
    parser.add_argument(
        "-c", "--config_dir", type=str, required=False, default="configs"
    )
    parser.add_argument(
        "-cht",
        "--config_human_tracking_path",
        type=str,
        required=False,
        default="configs/human_tracking.yaml",
    )
    parser.add_argument("-np", "--n_processes", type=int, required=False, default=None)
    parser.add_argument("-g", "--gpu", type=int, required=False, default=1)
    args = parser.parse_args()

    video_paths = sorted(glob(os.path.join(args.data_root, "*.mp4")))
    dataset_type = args.dataset_type

    cfg_path = os.path.join(args.config_dir, f"{dataset_type}.yaml")
    config = yaml_handler.load(cfg_path)
    config_ht = yaml_handler.load(args.config_human_tracking_path)
    device = f"cuda:{args.gpu}"
    n_processes = args.n_processes

    model_ht = HumanTracking(config_ht, device)
    for video_path in tqdm(video_paths, ncols=100, position=0):
        write_shards(video_path, dataset_type, config, model_ht, n_processes)
        model_ht.reset_tracker()

    del model_ht
