import argparse
import os
import pickle
import sys
from glob import glob

import torch
from tqdm import tqdm

sys.path.append(".")
from src.data import individual_pred_dataloader
from src.model import CSQVAE
from src.utils import yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("-v", "--version", type=int, default=0)
    parser.add_argument("-g", "--gpu_id", type=int, default=None)
    args = parser.parse_args()
    data_root = args.data_root
    v = args.version
    gpu_id = args.gpu_id
    device = f"cuda:{gpu_id}"

    data_dirs = [
        d
        for d in sorted(glob(os.path.join(data_root, "*/")))
        if os.path.basename(d[:-1]).isnumeric()
    ]

    checkpoint_dir = f"models/individual/csqvae/version_{v}"
    checkpoint_path = sorted(glob(f"{checkpoint_dir}/*.ckpt"))[-1]

    # load config
    config_path = f"{checkpoint_dir}/individual-csqvae.yaml"
    config = yaml_handler.load(config_path)
    seq_len = config.seq_len
    stride = config.stride

    # load model
    model = CSQVAE(config)
    model.configure_model()
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    for data_dir in tqdm(data_dirs, ncols=100):
        if data_dir[-1] == "/":
            data_dir = data_dir[:-1]

        # load dataset
        dataloader, n_samples = individual_pred_dataloader(
            data_dir, "individual", config, [gpu_id], False
        )
        if n_samples == 0:
            print(f"Skip predicting {data_dir}")
            continue

        # pred
        save_dir = os.path.join(data_dir, f"pred_csqvae_v{v}")
        os.makedirs(save_dir, exist_ok=True)

        model.eval()
        for batch in tqdm(dataloader, desc=f"{data_dir[-2:]}", ncols=100):
            results = model.predict_step(batch)
            for result in results:
                key = result["key"]
                path = os.path.join(save_dir, f"{key}.pkl")
                with open(path, "wb") as f:
                    pickle.dump(result, f)

    print("complete")
