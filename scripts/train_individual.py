import argparse
import os
import sys
from glob import glob

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from tqdm import tqdm

sys.path.append(".")
from src.data import individual_train_dataloader, load_dataset
from src.model import GAN, VAE
from src.utils import yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("model_type", type=str)
    parser.add_argument("-g", "--gpu_ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "-p", "--pretrain", required=False, action="store_true", default=False
    )
    args = parser.parse_args()
    data_root = args.data_root
    model_type = args.model_type
    gpu_ids = args.gpu_ids
    pretrain = args.pretrain

    # load config
    config_path = f"configs/individual-{model_type}.yaml"
    config = yaml_handler.load(config_path)

    # model checkpoint callback
    h, w = config.img_size
    checkpoint_dir = "models/individual/"
    filename = f"{model_type}-seq_len{config.seq_len}-stride{config.stride}-{h}x{w}"
    if pretrain:
        filename += "-pre"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename=filename + "-best",
        monitor="loss",
        mode="min",
        save_last=True,
    )
    model_checkpoint.CHECKPOINT_NAME_LAST = filename + "-last"

    # load dataset
    dataloader = individual_train_dataloader(data_root, "individual", config, gpu_ids)

    # create model
    if model_type == "vae":
        model = VAE(config)
        ddp = DDPStrategy(find_unused_parameters=False, process_group_backend="nccl")
        accumulate_grad_batches = config.accumulate_grad_batches
    elif model_type == "gan":
        if pretrain:
            pre_checkpoint_path = None
            clustering_init_batch = None
        else:
            pre_checkpoint_path = os.path.join(
                checkpoint_dir, f"{filename}-pre-last.ckpt"
            )
            if not os.path.exists(pre_checkpoint_path):
                pre_checkpoint_path = None

            # load clutering init batch
            data_dirs = sorted(glob(os.path.join(data_root, "*/")))
            dataset, _ = load_dataset(data_dirs, "individual", config, shuffle=False)
            dataset = iter(dataset)
            x_vis_lst = []
            x_spc_lst = []
            mask_lst = []
            for i in tqdm(
                range(config.n_clustering_init_batch),
                ncols=100,
                desc="load clustering init",
            ):
                _, _, x_vis, x_spc, mask = next(dataset)
                x_vis_lst.append(x_vis.view(1, config.seq_len, 17, 2))
                x_spc_lst.append(x_spc.view(1, config.seq_len, 2, 2))
                mask_lst.append(mask.view(1, config.seq_len))
            clustering_init_batch = (
                torch.cat(x_vis_lst, dim=0).contiguous(),
                torch.cat(x_spc_lst, dim=0).contiguous(),
                torch.cat(mask_lst, dim=0).contiguous(),
            )

        # create model
        model = GAN(config, clustering_init_batch, pretrain)
        ddp = DDPStrategy(find_unused_parameters=True, process_group_backend="nccl")
        accumulate_grad_batches = 1  # manual backward loss
    else:
        raise ValueError

    logger = TensorBoardLogger("logs/individual", name=model_type)
    trainer = Trainer(
        accelerator="cuda",
        strategy=ddp,
        devices=gpu_ids,
        logger=logger,
        callbacks=[model_checkpoint],
        max_epochs=config.epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        benchmark=True,
    )
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=pre_checkpoint_path)
