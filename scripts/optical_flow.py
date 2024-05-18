import argparse
import os
import sys
from glob import glob

import numpy as np
from tqdm import tqdm

sys.path.append("src")
from utils import video


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset_dir",
        type=str,
        help="path of input dataset directory",
    )
    parser.add_argument("--th_cutoff", required=False, type=float, default=0.05)
    parser.add_argument(
        "--is_not_half", required=False, default=True, action="store_false"
    )

    return parser.parse_args()


def main():
    args = parser()
    dataset_dir = args.dataset_dir
    th_cutoff = args.th_cutoff
    is_half = args.is_not_half

    clip_paths = sorted(glob(os.path.join(dataset_dir, "*.mp4")))

    for clip_path in tqdm(clip_paths[:1], ncols=100):
        flows = video.optical_flow(clip_path, th_cutoff, is_half)

        clip_dir = clip_paths.split(".")[0]
        os.makedirs(clip_dir, exist_ok=True)
        output_path = os.path.join(clip_dir, "flow.npz")
        np.savez_compressed(output_path, flows)
        tqdm.write(f"saved {output_path}")


if __name__ == "__main__":
    main()
