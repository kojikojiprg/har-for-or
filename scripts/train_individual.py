import argparse
import os
import sys

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

sys.path.append(".")
from src.data import DataModule, load_dataset
from src.model import IndividualActivityRecognition
from src.utils import yaml_handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("data_type", type=str, help="'keypoints' or 'images'")
    parser.add_argument("-g", "--gpu_ids", type=int, nargs="*", default=None)
    args = parser.parse_args()
    data_root = args.data_root
    data_type = args.data_type
    gpu_ids = args.gpu_ids

    # load config
    config_path = f"configs/individual_{data_type}.yaml"
    config = yaml_handler.load(config_path)

    # model checkpoint callback
    h, w = config.img_size
    checkpoint_dir = f"models/individual/{data_type}/"
    filename = f"individual_{data_type}_seq_len{config.seq_len}-stride{config.stride}-{h}x{w}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename=filename + "_loss_min",
        monitor="l",
        mode="min",
        save_last=True,
    )
    model_checkpoint.CHECKPOINT_NAME_LAST = filename + "_last"

    # load dataset
    dataset = load_dataset(data_root, "individual", data_type, config, True)
    datamodule = DataModule(
        dataset, "individual", config.batch_size, config.num_workers
    )

    # create model
    model = IndividualActivityRecognition(config)

    logger = TensorBoardLogger("logs/individual/", name=data_type)
    trainer = Trainer(
        accelerator="cuda",
        strategy="fsdp",
        devices=gpu_ids,
        logger=logger,
        callbacks=[model_checkpoint],
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        benchmark=True,
    )
    trainer.fit(model, datamodule=datamodule)
