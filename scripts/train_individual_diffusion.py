import argparse
import os
import shutil
import sys
from glob import glob

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

sys.path.append(".")
from src.data import individual_train_dataloader
from src.model import Diffusion
from src.utils import yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root_lst", type=str, nargs="*")
    parser.add_argument("-ckpt", "--csqvae_checkpoint", required=True, type=str)
    parser.add_argument("-g", "--gpu_ids", type=int, nargs="*", default=None)
    args = parser.parse_args()
    data_root_lst = args.data_root_lst
    gpu_ids = args.gpu_ids
    csqvae_checkpoint_path = args.csqvae_checkpoint

    # load config
    config_path = "configs/individual-diffusion.yaml"
    config = yaml_handler.load(config_path)
    csqvae_config_path = os.path.join(os.path.dirname(csqvae_checkpoint_path), "individual-csqvae.yaml")
    csqvae_config = yaml_handler.load(csqvae_config_path)
    config.csqvae = csqvae_config  # copy csqvae config

    if "WORLD_SIZE" not in os.environ:
        # create checkpoint directory of this version
        checkpoint_dir = "models/individual/diffusion"
        ckpt_dirs = glob(os.path.join(checkpoint_dir, "*/"))
        ckpt_dirs = [d for d in ckpt_dirs if "version_" in d]
        if len(ckpt_dirs) > 0:
            max_v_num = 0
            for d in ckpt_dirs:
                last_ckpt_dir = os.path.dirname(d)
                v_num = int(last_ckpt_dir.split("/")[-1].replace("version_", ""))
                if v_num > max_v_num:
                    max_v_num = v_num
            v_num = max_v_num + 1
        else:
            v_num = 0
        checkpoint_dir = os.path.join(checkpoint_dir, f"version_{v_num}")

        # copy config
        os.makedirs(checkpoint_dir, exist_ok=False)
        copy_config_path = os.path.join(checkpoint_dir, "individual-diffusion.yaml")
        shutil.copyfile(config_path, copy_config_path)
        copy_config_path = os.path.join(checkpoint_dir, "individual-csqvae.yaml")
        shutil.copyfile(csqvae_config_path, copy_config_path)

        # copy csqvae_checkpoint
        copy_ckpt_path = os.path.join(checkpoint_dir, os.path.basename(csqvae_checkpoint_path))
        shutil.copyfile(csqvae_checkpoint_path, copy_ckpt_path)
    else:
        checkpoint_dir = "models/individual/diffusion"
        ckpt_dirs = glob(os.path.join(checkpoint_dir, "*/"))
        ckpt_dirs = [d for d in ckpt_dirs if "version_" in d]
        checkpoint_dir = ckpt_dirs[-1]

    # model checkpoint callback
    filename = f"diffusion-seq_len{config.csqvae.seq_len}-stride{config.csqvae.stride}"
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename=filename + "-best-{epoch}",
        monitor="loss",
        mode="min",
        save_last=True,
    )
    model_checkpoint.CHECKPOINT_NAME_LAST = filename + "-last-{epoch}"

    # load dataset
    train_data_root_lst = [
        os.path.join(data_root, "train") for data_root in data_root_lst
    ]
    dataloader = individual_train_dataloader(
        train_data_root_lst,
        "individual",
        config.csqvae,
        gpu_ids,
        is_mapped=False,
    )

    # create model
    model = Diffusion(config, csqvae_checkpoint_path)
    ddp = DDPStrategy(find_unused_parameters=False, process_group_backend="nccl")
    accumulate_grad_batches = config.accumulate_grad_batches

    logger = TensorBoardLogger("logs/individual", name="diffusion")
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
    trainer.fit(model, train_dataloaders=dataloader)

    print("complete")
