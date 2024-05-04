import argparse
import sys

sys.path.append("src")
from datamodule import DynamicSpatialTemporalGraphDataset
from utils import yaml_handler

parser = argparse.ArgumentParser()
parser.add_argument("data_root", type=str)
parser.add_argument("feature_type", type=str)
parser.add_argument(
    "-c", "--config", type=str, required=False, default="configs/model_config.yaml"
)
args = parser.parse_args()

config = yaml_handler.load(args.config)
DynamicSpatialTemporalGraphDataset(args.data_root, args.feature_type, config)
