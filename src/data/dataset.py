import functools
import itertools
import os
import tarfile
from glob import glob
from math import ceil
from types import SimpleNamespace

import webdataset as wds

from .transform import (
    FlowToTensor,
    FrameToTensor,
    NormalizeBbox,
    NormalizeKeypoints,
    group_npz_to_tensor,
    individual_npz_to_tensor,
)


def load_dataset(
    data_dirs: list, dataset_type: str, config: SimpleNamespace, shuffle: bool
) -> wds.WebLoader:
    shard_paths = []

    seq_len = int(config.seq_len)
    stride = int(config.stride)
    h, w = config.img_size
    shard_pattern = f"{dataset_type}-seq_len{seq_len}-stride{stride}-{h}x{w}" + "-*.tar"
    n_samples = 0
    for dir_path in data_dirs:
        shard_paths_tmp = sorted(glob(os.path.join(dir_path, "shards", shard_pattern)))
        shard_paths += shard_paths_tmp
        n_samples += (len(shard_paths_tmp) - 1) * config.max_shard_count
        with tarfile.open(shard_paths_tmp[-1]) as tar:
            n_samples += len(tar.getnames())

    dataset = wds.WebDataset(
        shard_paths, shardshuffle=shuffle, nodesplitter=_node_splitter
    )
    if shuffle:
        dataset = dataset.shuffle(100)

    if dataset_type == "individual":
        idv_npz_to_tensor = functools.partial(
            individual_npz_to_tensor,
            seq_len=seq_len,
            frame_transform=FrameToTensor(),
            flow_transform=FlowToTensor(),
            bbox_transform=NormalizeBbox(),
            kps_transform=NormalizeKeypoints(),
            mask_leg=config.mask_leg,
        )
        dataset = dataset.map(idv_npz_to_tensor)
    elif dataset_type == "group":
        grp_npz_to_tensor = functools.partial(
            group_npz_to_tensor,
            frame_transform=FrameToTensor(),
            flow_transform=FlowToTensor(),
            point_transform=NormalizeBbox(),
        )
        dataset = dataset.map(grp_npz_to_tensor)
    else:
        raise ValueError

    return dataset, n_samples


def individual_train_dataloader(
    data_root: str, dataset_type: str, config: SimpleNamespace, gpu_ids: list
) -> wds.WebLoader:
    data_dirs = sorted(glob(os.path.join(data_root, "*/")))

    dataset, n_batches = load_dataset(data_dirs, dataset_type, config, True)
    dataset = dataset.batched(config.batch_size, partial=False)

    # create dataloader
    dataloader = wds.WebLoader(
        dataset,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    n_batches = int(n_batches / len(gpu_ids) / config.batch_size)
    if n_batches % config.accumulate_grad_batches != 0:
        n_batches -= n_batches % config.accumulate_grad_batches
    dataloader.repeat(config.epochs, n_batches)

    return dataloader, n_batches


def individual_pred_dataloader(
    data_dir: str, dataset_type: str, config: SimpleNamespace, gpu_ids: list
) -> wds.WebLoader:
    dataset, n_batches = load_dataset([data_dir], dataset_type, config, False)
    dataset = dataset.batched(config.batch_size, partial=True)

    # create dataloader
    dataloader = wds.WebLoader(
        dataset,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    n_batches = ceil(n_batches / len(gpu_ids) / config.batch_size)
    dataloader.repeat(1, n_batches)

    return dataloader, n_batches


def _node_splitter(src):
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    if world_size > 1:
        rank = int(os.environ["LOCAL_RANK"])
        yield from itertools.islice(src, rank, None, world_size)
    else:
        yield from src
