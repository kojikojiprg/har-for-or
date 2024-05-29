import argparse
import sys

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import FSDPStrategy

sys.path.append(".")
from src.data import DataModule, load_dataset
from src.model import IndividualActivityRecognition
from src.model.layers import (
    IndividualEmbedding,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)
from src.utils import yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("data_type", type=str, help="'individual' or 'images'")
    parser.add_argument("-g", "--gpu_ids", type=int, nargs="*", default=None)
    args = parser.parse_args()
    data_root = args.data_root
    data_type = args.data_type
    gpu_ids = args.gpu_ids

    # load configs
    dataset_cfg_path = "configs/dataset.yaml"
    model_cfg_path = f"configs/individual_{data_type}.yaml"
    dataset_cfg = yaml_handler.load(dataset_cfg_path)
    model_cfg = yaml_handler.load(model_cfg_path)

    # load dataset
    dataset = load_dataset(data_root, "individual", dataset_cfg)
    datamodule = DataModule(dataset, model_cfg.batch_size)

    # create model
    model = IndividualActivityRecognition(model_cfg)

    logger = TensorBoardLogger("logs/individual/", name=data_type)
    fsdp = FSDPStrategy(
        auto_wrap_policy={
            IndividualEmbedding,
            TransformerDecoderBlock,
            TransformerEncoderBlock,
        },
        sharding_strategy="FULL_SHARD",
    )
    trainer = Trainer(
        accelerator="cuda",
        strategy=fsdp,
        devices=gpu_ids,
        logger=logger,
        callbacks=model.callbacks,
        max_epochs=model_cfg.epochs,
        accumulate_grad_batches=model_cfg.accumulate_grad_batches,
    )
    trainer.fit(model, datamodule=datamodule)
