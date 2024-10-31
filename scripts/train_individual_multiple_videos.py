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
from src.data import individual_train_dataloader, load_annotation_train
from src.model import SQVAE, VAE
from src.utils import yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root_lst", type=str, nargs="*")
    parser.add_argument("-g", "--gpu_ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "-mt", "--model_type", required=False, type=str, default="sqvae"
    )
    parser.add_argument(
        "-ut",
        "--unsupervised_training",
        required=False,
        action="store_true",
        default=False,
    )
    parser.add_argument("-ckpt", "--checkpoint", required=False, type=str, default=None)
    args = parser.parse_args()
    data_root_lst = args.data_root_lst
    model_type = args.model_type
    unsupervised_training = args.unsupervised_training
    gpu_ids = args.gpu_ids
    pre_checkpoint_path = args.checkpoint

    # load config
    config_path = f"configs/individual-{model_type}.yaml"
    config = yaml_handler.load(config_path)

    if "WORLD_SIZE" not in os.environ:
        # create checkpoint directory of this version
        checkpoint_dir = f"models/individual/{model_type}"
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
        copy_config_path = os.path.join(checkpoint_dir, f"individual-{model_type}.yaml")
        shutil.copyfile(config_path, copy_config_path)
    else:
        checkpoint_dir = f"models/individual/{model_type}"
        ckpt_dirs = glob(os.path.join(checkpoint_dir, "*/"))
        ckpt_dirs = [d for d in ckpt_dirs if "version_" in d]
        checkpoint_dir = ckpt_dirs[-1]

    # model checkpoint callback
    # h, w = config.img_size
    filename = f"{model_type}-seq_len{config.seq_len}-stride{config.stride}"
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename=filename + "-best-{epoch}",
        monitor="loss",
        mode="min",
        save_last=True,
    )
    model_checkpoint.CHECKPOINT_NAME_LAST = filename + "-last-{epoch}"

    # create annotation
    annotations = load_annotation_train(data_root_lst, checkpoint_dir, config)

    # load dataset
    train_data_root_lst = [
        os.path.join(data_root, "train") for data_root in data_root_lst
    ]
    dataloader = individual_train_dataloader(
        train_data_root_lst,
        "individual",
        config,
        gpu_ids,
        is_mapped=False,
    )

    # create model
    if model_type == "vae":
        # ann_path = f"{data_root}/annotation/role.txt"
        # model = VAE(config, annotation_path=ann_path)
        model = VAE(config)
        ddp = DDPStrategy(find_unused_parameters=True, process_group_backend="nccl")
        raise NotImplementedError  # TODO: delete
    elif model_type == "sqvae":
        if unsupervised_training:
            ann_path = None
        model = SQVAE(config, annotations)
        ddp = DDPStrategy(find_unused_parameters=False, process_group_backend="nccl")
    accumulate_grad_batches = config.accumulate_grad_batches

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
