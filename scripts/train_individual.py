import argparse
import os
import sys

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from webdataset import WebLoader

sys.path.append(".")
from src.data import load_dataset
from src.model import IndividualActivityRecognition
from src.utils import yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("-g", "--gpu_ids", type=int, nargs="*", default=None)
    args = parser.parse_args()
    data_root = args.data_root
    gpu_ids = args.gpu_ids

    # load config
    config_path = "configs/individual.yaml"
    config = yaml_handler.load(config_path)

    # model checkpoint callback
    h, w = config.img_size
    checkpoint_dir = "models/individual/"
    filename = f"individual-seq_len{config.seq_len}-stride{config.stride}-{h}x{w}.pt"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename=filename + "-loss-min",
        monitor="l",
        mode="min",
        save_weights_only=True,
        save_last=True,
    )
    model_checkpoint.CHECKPOINT_NAME_LAST = filename + "-last.pt"

    # load dataset
    dataset, n_samples = load_dataset(data_root, "individual", config, True)
    dataset = dataset.batched(config.batch_size, partial=False)

    dataloader = WebLoader(dataset, num_workers=config.num_workers, pin_memory=True)
    n_samples = int(n_samples / len(gpu_ids) / config.batch_size)
    dataloader.repeat(2).with_epoch(n_samples).with_length(n_samples - 1)

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
    trainer.fit(model, train_dataloaders=dataloader)
