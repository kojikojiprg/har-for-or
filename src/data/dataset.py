import functools
import itertools
import os
import tarfile
from glob import glob
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
    data_dirs: str, dataset_type: str, config: SimpleNamespace, shuffle: bool
) -> wds.WebLoader:
    shard_paths = []

    seq_len = int(config.seq_len)
    stride = int(config.stride)
    h, w = config.img_size
    shard_pattern = f"{dataset_type}-seq_len{seq_len}-stride{stride}-{h}x{w}" + "-*.tar"
    for dir_path in data_dirs:
        shard_paths += sorted(glob(os.path.join(dir_path, "shards", shard_pattern)))

    node_splitter = functools.partial(_node_splitter, length=len(shard_paths))
    dataset = wds.WebDataset(
        shard_paths, shardshuffle=shuffle, nodesplitter=node_splitter
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

    n_samples = (len(shard_paths) - 1) * config.max_shard_count
    with tarfile.open(shard_paths[-1]) as tar:
        n_samples += len(tar.getnames())

    return dataset, n_samples


def individual_train_dataloader(
    data_root: str, dataset_type: str, config: SimpleNamespace, gpu_ids: list
) -> wds.WebLoader:
    data_dirs = sorted(glob(os.path.join(data_root, "*/")))

    dataset, n_samples = load_dataset(data_dirs, dataset_type, config, True)
    dataset = dataset.batched(config.batch_size, partial=False)

    # create dataloader
    dataloader = wds.WebLoader(dataset, num_workers=config.num_workers, pin_memory=True, persistent_workers=True)
    n_samples = int(n_samples / len(gpu_ids) / config.batch_size)
    if n_samples % config.accumulate_grad_batches != 0:
        n_samples -= n_samples % config.accumulate_grad_batches
    dataloader.with_epoch(n_samples).with_length(n_samples)

    return dataloader


def _node_splitter(src, length):
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    if world_size > 1:
        rank = int(os.environ["LOCAL_RANK"])
        yield from itertools.islice(src, rank, length, world_size)
    else:
        yield from src
