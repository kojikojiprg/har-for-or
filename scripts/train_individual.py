import argparse
import os
import sys

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

sys.path.append(".")
from src.data import individual_train_dataloader
from src.model import IndividualActivityRecognition
from src.utils import yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("-g", "--gpu_ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "-l", "--load_checkpoint", required=False, action="store_true", default=False
    )
    args = parser.parse_args()
    data_root = args.data_root
    gpu_ids = args.gpu_ids
    load_checkpoint = args.load_checkpoint

    # load config
    config_path = "configs/individual.yaml"
    config = yaml_handler.load(config_path)

    # model checkpoint callback
    h, w = config.img_size
    checkpoint_dir = "models/individual/"
    filename = f"individual-seq_len{config.seq_len}-stride{config.stride}-{h}x{w}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename=filename + "-loss-min",
        monitor="l",
        mode="min",
        save_last=True,
    )
    model_checkpoint.CHECKPOINT_NAME_LAST = filename + "-last"
    if load_checkpoint:
        checkpoint_path = model_checkpoint.CHECKPOINT_NAME_LAST
    else:
        checkpoint_path = None

    # load dataset
    dataloader = individual_train_dataloader(data_root, "individual", config, gpu_ids)

    # create model
    model = IndividualActivityRecognition(config)

    logger = TensorBoardLogger("logs", name="individual")
    trainer = Trainer(
        accelerator="cuda",
        strategy="ddp",
        devices=gpu_ids,
        logger=logger,
        callbacks=[model_checkpoint],
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        benchmark=True,
    )
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=checkpoint_path)
