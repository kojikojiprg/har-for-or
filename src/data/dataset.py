import functools
import itertools
import os
import tarfile
from glob import glob
from math import ceil
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import webdataset as wds
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .transform import (
    FlowToTensor,
    FrameToTensor,
    NormalizeBbox,
    NormalizeKeypoints,
    group_npz_to_tensor,
    individual_npz_to_tensor,
)


class IndividualDatasetMapped(Dataset):
    def __init__(self, shard_paths, func_to_tensor):
        self.keys = []
        self.ids = []
        self.bboxs = []
        self.kps = []
        self.masks = []

        for path in tqdm(shard_paths, ncols=100, desc="loading shards"):
            with tarfile.open(path, "r") as tar:
                for tarinfo in tar:
                    name = tarinfo.name.split(".")[0]
                    sample = tar.extractfile(tarinfo).read()
                    sample = dict(__key__=name, npz=sample)
                    key, _id, bbox, kps, mask = func_to_tensor(sample)
                    self.keys.append(key)
                    self.ids.append(_id)
                    self.bboxs.append(bbox)
                    self.kps.append(kps)
                    self.masks.append(mask)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        _id = self.ids[index]
        bbox = self.bboxs[index]
        kps = self.kps[index]
        mask = self.masks[index]
        return key, _id, bbox, kps, mask


def load_dataset_mapped(
    data_dirs: list, dataset_type: str, config: SimpleNamespace
) -> Dataset:
    shard_paths = []

    seq_len = int(config.seq_len)
    stride = int(config.stride)
    h, w = config.img_size
    shard_pattern = f"{dataset_type}-seq_len{seq_len}-stride{stride}-{h}x{w}" + "-*.tar"
    for dir_path in data_dirs:
        shard_paths_tmp = sorted(glob(os.path.join(dir_path, "shards", shard_pattern)))
        shard_paths += shard_paths_tmp

    if dataset_type == "individual":
        idv_npz_to_tensor = functools.partial(
            individual_npz_to_tensor,
            seq_len=seq_len,
            frame_transform=FrameToTensor(),
            flow_transform=FlowToTensor(),
            bbox_transform=NormalizeBbox(),
            kps_transform=NormalizeKeypoints(),
            mask_leg=config.mask_leg,
            range_points=config.range_points,
            load_frame_flow=False,  # avoid memory overflow
        )
        dataset = IndividualDatasetMapped(shard_paths, idv_npz_to_tensor)
    elif dataset_type == "group":
        # grp_npz_to_tensor = functools.partial(
        #     group_npz_to_tensor,
        #     frame_transform=FrameToTensor(),
        #     flow_transform=FlowToTensor(),
        #     point_transform=NormalizeBbox(),
        # )
        raise NotImplementedError
    else:
        raise ValueError

    return dataset


def load_dataset_iterable(
    data_dirs: list, dataset_type: str, config: SimpleNamespace, shuffle: bool
) -> Tuple[wds.WebDataset, int]:
    shard_paths = []

    seq_len = int(config.seq_len)
    stride = int(config.stride)
    # h, w = config.img_size
    shard_pattern = f"{dataset_type}-seq_len{seq_len}-stride{stride}-*.tar"
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
            range_points=config.range_points,
            load_frame_flow=False,  # TODO: set True when use frame and flow
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
    data_root_lst: List[str],
    dataset_type: str,
    config: SimpleNamespace,
    gpu_ids: list,
    is_mapped: bool,
) -> Union[DataLoader, wds.WebLoader]:
    data_dirs = []
    for data_root in data_root_lst:
        data_dirs += sorted(glob(os.path.join(data_root, "*/")))

    if is_mapped:
        dataset = load_dataset_mapped(data_dirs, dataset_type, config)
        dataloader = DataLoader(
            dataset,
            config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    else:
        dataset, n_batches = load_dataset_iterable(
            data_dirs, dataset_type, config, True
        )
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
        dataloader.repeat(config.epochs, n_batches).with_length(n_batches)

    return dataloader


def individual_pred_dataloader(
    data_dir: str,
    dataset_type: str,
    config: SimpleNamespace,
    gpu_ids: Optional[list],
    is_mapped: bool,
) -> Union[DataLoader, wds.WebLoader]:
    if is_mapped:
        dataset = load_dataset_mapped([data_dir], dataset_type, config)
        dataloader = DataLoader(
            dataset,
            config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    else:
        dataset, n_batches = load_dataset_iterable(
            [data_dir], dataset_type, config, False
        )
        dataset = dataset.batched(config.batch_size, partial=True)

        # create dataloader
        dataloader = wds.WebLoader(
            dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        ngpus = len(gpu_ids) if gpu_ids is not None else 1
        n_batches = ceil(n_batches / ngpus / config.batch_size)
        dataloader.repeat(1, n_batches).with_length(n_batches)

    return dataloader


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
