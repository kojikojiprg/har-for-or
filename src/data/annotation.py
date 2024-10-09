import os
import sys
from glob import glob
from types import SimpleNamespace

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")
from src.data.dataset import load_dataset_mapped


def load_annotation_train(
    data_root: str,
    checkpoint_dir: str,
    config: SimpleNamespace,
    seed: int = 42,
):
    np.random.seed(seed)

    # load annotation
    path = f"{data_root}/annotation/role_train.txt"
    annotation = np.loadtxt(path, str, skiprows=1)

    counts_samples = count_samples(data_root, config)

    # count labels
    counts_dict = {i: {} for i in range(config.n_clusters)}
    total = 0
    for key, count in counts_samples:
        count = int(count)
        ann = annotation[annotation.T[0] == key]
        total += count
        if len(ann) == 1:
            label = int(ann[0, 1])
            if key not in counts_dict[label]:
                counts_dict[label][key] = 0
            counts_dict[label][key] += count
        elif len(ann) == 0:
            pass
        else:
            print("warning", key, len(ann))

    # sum label counts until min_n_samples
    used_annotation = []
    summary_annotation = []
    count_non_labeled = total
    for label, counts in counts_dict.items():
        # sort counts by keys
        counts = sorted([(key, c) for key, c in counts.items()], key=lambda x: x[0])

        # shuffle
        counts = np.array(counts)
        indices = np.random.choice(len(counts), len(counts), replace=False)
        counts = counts[indices]

        # sum counts
        count_sum = 0
        for k, c in counts:
            used_annotation.append((k, label))
            count_sum += int(c)
            count_non_labeled -= int(c)
            if count_sum >= config.min_n_labeled_samples:
                summary_annotation.append((label, count_sum))
                break

    summary_annotation.append(("non-labeld", count_non_labeled))
    summary_annotation.append(("total", total))

    # save sammary into checkpoint directory
    path = f"{checkpoint_dir}/annotation_train_summary.tsv"
    if not os.path.exists(path):
        np.savetxt(path, summary_annotation, "%s", delimiter="\t")

    # save used annotation into checkpoint directory
    path = f"{checkpoint_dir}/annotation_train.tsv"
    if not os.path.exists(path):
        np.savetxt(path, used_annotation, "%s", delimiter="\t")

    return np.array(used_annotation)


def count_samples(data_root: str, config: SimpleNamespace):
    path_counts = f"{data_root}/annotation/counts_train.txt"

    if os.path.exists(path_counts):
        counts = np.loadtxt(path_counts, str, delimiter=" ")
    else:
        # load dataset
        data_dirs = glob(f"{data_root}/train/**/")
        dataset = load_dataset_mapped(data_dirs, "individual", config)
        dataloader = DataLoader(dataset, num_workers=16, pin_memory=True)

        # count labels
        count_keys = {}
        for batch in tqdm(iter(dataloader), ncols=100, desc="annot"):
            keys = np.array(batch[0]).ravel()
            for key in keys:
                video_num, n_frame, _id = key.split("_")
                key = f"{video_num}_{_id}"

                if key not in count_keys:
                    count_keys[key] = 0
                count_keys[key] += 1
        del dataset, dataloader

        counts = [(key, count) for key, count in count_keys.items()]
        counts = sorted(counts, key=lambda x: x[0])
        counts = np.array(counts)

        np.savetxt(path_counts, counts, "%s", delimiter=" ")

    return counts
