import functools
import gc
import itertools
import os
import sys
import time
import warnings
from glob import glob
from multiprocessing import shared_memory
from multiprocessing.managers import SyncManager
from types import SimpleNamespace

warnings.filterwarnings("ignore")
import numpy as np
import webdataset as wds
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm

from src.model import HumanTracking
from src.utils import json_handler, video

from .transform import (
    FlowToTensor,
    FrameToTensor,
    NormalizeBbox,
    NormalizeKeypoints,
    clip_images_by_bbox,
    collect_human_tracking,
    group_npz_to_tensor,
    individual_npz_to_tensor,
    individual_to_npz,
)

set_start_method("spawn", force=True)


def write_shards(
    video_path: str,
    dataset_type: str,
    config: SimpleNamespace,
    model_ht: HumanTracking = None,
    n_processes: int = None,
):
    if n_processes is None:
        n_processes = os.cpu_count()

    data_root = os.path.dirname(video_path)
    video_name = os.path.basename(video_path).split(".")[0]
    dir_path = os.path.join(data_root, video_name)

    json_path = os.path.join(dir_path, "json", "pose.json")

    shard_maxcount = float(config.max_shard_count)
    seq_len = int(config.seq_len)
    stride = int(config.stride)
    w = int(config.resize_shape.w)
    h = int(config.resize_shape.h)
    shard_pattern = (
        f"{dataset_type}-seq_len{seq_len}-stride{stride}-{h}x{w}" + "-%06d.tar"
    )

    shard_pattern = os.path.join(dir_path, "shards", shard_pattern)
    os.makedirs(os.path.dirname(shard_pattern), exist_ok=True)

    ShardWritingManager.register("Tqdm", tqdm)
    ShardWritingManager.register("Capture", video.Capture)
    ShardWritingManager.register("SharedShardWriter", SharedShardWriter)
    with Pool(n_processes) as pool, ShardWritingManager() as swm:
        async_results = []

        lock = swm.Lock()
        cap = swm.Capture(video_path)
        frame_count, img_size = cap.get_frame_count(), cap.get_size()
        head = swm.Value("i", 0)

        # create progress bars
        pbar_of = swm.Tqdm(
            total=frame_count, desc="opticalflow", position=1, leave=False, ncols=100
        )
        pbar_ht = swm.Tqdm(
            total=frame_count, desc="tracking", position=2, leave=False, ncols=100
        )
        total = (frame_count - seq_len) // stride + 1
        pbar_w = swm.Tqdm(
            total=total, desc="writing", position=3, leave=False, ncols=100
        )

        # create shared ndarray and start optical flow
        shape = (seq_len, img_size[1], img_size[0], 3)
        frame_sna = SharedNDArray(f"frame_{dataset_type}", shape, np.uint8)
        shape = (seq_len, img_size[1], img_size[0], 2)
        flow_sna = SharedNDArray(f"flow_{dataset_type}", shape, np.float32)
        tail_frame = swm.Value("i", 0)
        pool.apply_async(
            _optical_flow_async,
            (cap, frame_sna, flow_sna, tail_frame, head, lock, pbar_of),
        )

        # create shared list of indiciduals and start human tracking
        ht_que = swm.list([[] for _ in range(seq_len)])
        tail_ht = swm.Value("i", 0)
        pool.apply_async(
            _human_tracking_async,
            (cap, json_path, model_ht, ht_que, tail_ht, head, lock, pbar_ht),
        )

        # create shard writer and start writing
        sink = swm.SharedShardWriter(
            shard_pattern, maxcount=shard_maxcount, verbose=0, post=tqdm.write
        )
        write_shard_async_f = functools.partial(
            _write_shard_async,
            frame_sna=frame_sna,
            flow_sna=flow_sna,
            ht_que=ht_que,
            head=head,
            sink=sink,
            lock=lock,
            pbar=pbar_w,
            video_name=video_name,
            dataset_type=dataset_type,
            seq_len=seq_len,
            stride=stride,
            resize=(w, h),
        )
        check_full_f = functools.partial(
            _check_full,
            tail_frame=tail_frame,
            tail_ht=tail_ht,
            head=head,
            que_len=seq_len,
        )
        for n_frame in range(seq_len, frame_count + 1, stride):
            while not check_full_f():
                time.sleep(0.001)
            time.sleep(0.5)  # after delay

            # start writing
            result = pool.apply_async(
                write_shard_async_f, (n_frame,), error_callback=_error_callback
            )
            async_results.append(result)

            sleep_count = 0
            while check_full_f():
                time.sleep(0.5)  # waiting for coping queue in wirte_async
                sleep_count += 1
                if sleep_count > 60 * 3 / 0.5:
                    break  # avoid infinite loop after 3 min

        while [r.wait() for r in async_results].count(True) > 0:
            time.sleep(0.5)
        frame_sna.unlink()
        flow_sna.unlink()
        pbar_of.close()
        pbar_ht.close()
        pbar_w.close()
        sink.close()
    gc.collect()


class SharedNDArray:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.shm = self.create_shared_memory()

    def calc_ndarray_size(self):
        return np.prod(self.shape) * np.dtype(self.dtype).itemsize

    def create_shared_memory(self):
        size = self.calc_ndarray_size()
        return shared_memory.SharedMemory(name=self.name, create=True, size=size)

    def get_shared_memory(self):
        shm = shared_memory.SharedMemory(name=self.name)
        return shm

    def ndarray(self):
        shm = self.get_shared_memory()
        return np.ndarray(self.shape, self.dtype, buffer=shm.buf), shm

    def unlink(self):
        self.shm.close()
        self.shm.unlink()


class SharedShardWriter(wds.ShardWriter):
    def __init__(self, shard_pattern, maxcount, verbose=0, post=None):
        super().__init__(shard_pattern, maxcount, verbose=verbose, post=post)

    def tar_is_closed(self):
        return self.tarstream is None or self.tarstream.tarstream.closed


class ShardWritingManager(SyncManager):
    pass


def _check_full(tail_frame, tail_ht, head, que_len):
    is_frame_que_full = (tail_frame.value + 1) % que_len == head.value
    is_idv_que_full = (tail_ht.value + 1) % que_len == head.value
    is_eq = tail_frame.value == tail_ht.value
    return is_frame_que_full and is_idv_que_full and is_eq


def _optical_flow_async(cap, frame_sna, flow_sna, tail_frame, head, lock, pbar):
    frame_que, frame_shm = frame_sna.ndarray()
    flow_que, flow_shm = flow_sna.ndarray()
    que_len = frame_que.shape[0]

    with lock:
        frame_count = cap.get_frame_count()
        prev_frame = cap.read(0)[1]
        cap.set_pos_frame_count(0)  # reset read position

        frame_que[tail_frame.value] = prev_frame
        tail_frame.value = 1
    pbar.update()

    for n_frame in range(1, frame_count):
        with lock:
            frame = cap.read(n_frame)[1]
        flow = video.optical_flow(prev_frame, frame)
        prev_frame = frame

        with lock:
            frame_que[tail_frame.value] = frame
            flow_que[tail_frame.value] = flow
        pbar.update()

        if n_frame + 1 == frame_count:
            break  # finish

        next_tail = (tail_frame.value + 1) % que_len
        while next_tail == head.value:
            time.sleep(0.001)
        tail_frame.value = next_tail

    frame_shm.close()
    flow_shm.close()
    del frame_que, flow_que


def _human_tracking_async(cap, json_path, model, ht_que, tail_ht, head, lock, pbar):
    que_len = len(ht_que)

    do_human_tracking = not os.path.exists(json_path)
    if not do_human_tracking:
        json_data = json_handler.load(json_path)

    with lock:
        frame_count = cap.get_frame_count()

    for n_frame in range(frame_count):
        if do_human_tracking:
            with lock:
                frame = cap.read(n_frame)[1]
            idvs_tmp = model.predict(frame, n_frame)
        else:
            idvs_tmp = [idv for idv in json_data if idv["n_frame"] == n_frame]

        with lock:
            ht_que[tail_ht.value] = idvs_tmp
            pbar.update()

        if n_frame + 1 == frame_count:
            break  # finish

        next_tail = (tail_ht.value + 1) % que_len
        while next_tail == head.value:
            time.sleep(0.001)
        tail_ht.value = next_tail


def _write_shard_async(
    n_frame,
    frame_sna,
    flow_sna,
    ht_que,
    head,
    sink,
    lock,
    pbar,
    video_name,
    dataset_type,
    seq_len,
    stride,
    resize,
):
    assert (
        n_frame % seq_len == head.value
    ), f"n_frame % seq_len:{n_frame % seq_len}, head:{head.value}"
    with lock:
        # copy queue
        frame_que, frame_shm = frame_sna.ndarray()
        flow_que, flow_shm = flow_sna.ndarray()
        copy_frame_que = frame_que.copy()
        copy_flow_que = flow_que.copy()
        frame_shm.close()
        flow_shm.close()
        del frame_que, flow_que
        copy_ht_que = list(ht_que)
        head_val_copy = head.value
        head.value = (head.value + stride) % seq_len

    # sort to start from head
    sorted_idxs = list(range(head_val_copy, seq_len)) + list(range(0, head_val_copy))
    copy_frame_que = copy_frame_que[sorted_idxs].astype(np.uint8)
    copy_flow_que = copy_flow_que[sorted_idxs].astype(np.float32)
    copy_ht_que = [copy_ht_que[idx] for idx in sorted_idxs]

    # clip frames and flows by bboxs
    idv_frames, idv_flows = clip_images_by_bbox(
        copy_frame_que, copy_flow_que, copy_ht_que, resize
    )

    # collect human tracking data
    unique_ids = set(
        itertools.chain.from_iterable(
            [[idv["id"] for idv in idvs] for idvs in copy_ht_que]
        )
    )
    unique_ids = sorted(list(unique_ids))
    meta, ids, bboxs, kps = collect_human_tracking(copy_ht_que, unique_ids)
    img_size = np.array(copy_frame_que.shape[1:3])

    if dataset_type == "individual":
        idv_npzs = individual_to_npz(
            meta, unique_ids, idv_frames, idv_flows, bboxs, kps, img_size
        )
        for i, _id in enumerate(unique_ids):
            data = {"__key__": f"{video_name}_{n_frame}_{_id}", "npz": idv_npzs[i]}
            if sink.tar_is_closed():
                time.sleep(0.5)
            sink.write(data)
    elif dataset_type == "group":
        data = {
            "__key__": f"{video_name}_{n_frame}",
            "npz": {
                "meta": meta,
                "id": ids,
                "frame": idv_frames,
                "flow": idv_flows,
                "bbox": bboxs,
                "keypoints": kps,
                "img_size": img_size,
            },
        }
        if sink.tar_is_closed():
            time.sleep(0.5)
        sink.write(data)
    else:
        raise ValueError

    # pbar.write(f"Complete writing n_frame:{n_frame}")
    pbar.update()
    del data
    del copy_frame_que, copy_flow_que, copy_ht_que
    gc.collect()


def _error_callback(*args):
    print(f"Error occurred in write_shard_async process:\n{args}")
    sys.exit()


def load_dataset(
    data_root: str,
    dataset_type: str,
    feature_type: str,
    config: SimpleNamespace,
    shuffle: bool = False,
):
    shard_paths = []
    data_dirs = sorted(glob(os.path.join(data_root, "*/")))

    seq_len = int(config.seq_len)
    stride = int(config.stride)
    h, w = config.img_size
    shard_pattern = f"{dataset_type}-seq_len{seq_len}-stride{stride}-{h}x{w}" + "-*.tar"
    for dir_path in data_dirs:
        shard_paths += sorted(glob(os.path.join(dir_path, "shards", shard_pattern)))

    node_splitter = functools.partial(_node_splitter, length=len(shard_paths))
    idv_npz_to_tensor = functools.partial(
        individual_npz_to_tensor,
        feature_type=feature_type,
        seq_len=seq_len,
        frame_transform=FrameToTensor(),
        flow_transform=FlowToTensor(),
        bbox_transform=NormalizeBbox(),
        kps_transform=NormalizeKeypoints(),
    )
    grp_npz_to_tensor = functools.partial(
        group_npz_to_tensor,
        feature_typeype=feature_type,
        frame_transform=FrameToTensor(),
        flow_transform=FlowToTensor(),
        bbox_transform=NormalizeBbox(),
        kps_transform=NormalizeKeypoints(),
    )

    dataset = wds.WebDataset(
        shard_paths, shardshuffle=shuffle, nodesplitter=node_splitter
    )

    if shuffle:
        dataset = dataset.shuffle(100)

    if dataset_type == "individual":
        dataset = dataset.map(idv_npz_to_tensor)
    elif dataset_type == "group":
        dataset = dataset.map(grp_npz_to_tensor)
    else:
        raise ValueError

    return dataset


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
