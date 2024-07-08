import copy
import functools
import gc
import itertools
import os
import time
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")
import numpy as np
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm

from src.model import HumanTracking
from src.utils import json_handler, video

from .obj import ShardWritingManager, SharedNDArray, SharedShardWriter
from .transform import clip_images_by_bbox, collect_human_tracking, individual_to_npz

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
    h, w = config.img_size
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
        cap_of = swm.Capture(video_path)
        cap_ht = swm.Capture(video_path)
        frame_count, img_size = cap_of.get_frame_count(), cap_of.get_size()
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
        tail_of = swm.Value("i", 0)
        ec = functools.partial(_error_callback, *("_optical_flow_async",))
        result = pool.apply_async(
            _optical_flow_async,
            (cap_of, frame_sna, flow_sna, tail_of, head, lock, pbar_of),
            error_callback=ec,
        )
        async_results.append(result)

        # create shared list of indiciduals and start human tracking
        ht_que = swm.list([[] for _ in range(seq_len)])
        n_frames_que = swm.list([-1 for _ in range(seq_len)])
        tail_ht = swm.Value("i", 0)
        ec = functools.partial(_error_callback, *("_human_tracking_async",))
        result = pool.apply_async(
            _human_tracking_async,
            (
                cap_ht,
                json_path,
                model_ht,
                ht_que,
                n_frames_que,
                tail_ht,
                head,
                lock,
                pbar_ht,
            ),
            error_callback=ec,
        )
        async_results.append(result)

        # create shard writer and start writing
        sink = swm.SharedShardWriter(shard_pattern, maxcount=shard_maxcount, verbose=0)
        ec = functools.partial(_error_callback, *("SharedShardWriter.write_async",))
        write_async_result = pool.apply_async(sink.write_async, error_callback=ec)
        arr_write_que_async_f = functools.partial(
            _add_write_que_async,
            n_frames_que=n_frames_que,
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
            tail_of=tail_of,
            tail_ht=tail_ht,
            head=head,
            que_len=seq_len,
        )
        ec = functools.partial(_error_callback, *("_add_write_que_async",))

        for n_frame in range(seq_len, frame_count + 1, stride):
            while not check_full_f():
                async_results = _monitoring_async_tasks(async_results)
                time.sleep(0.01)

            while n_frame != n_frames_que[tail_ht.value] + 1:
                async_results = _monitoring_async_tasks(async_results)
                time.sleep(0.1)  # waiting for shared memory has been updated
            time.sleep(0.1)  # after delay

            # create and add data in write que
            result = pool.apply_async(
                arr_write_que_async_f, (n_frame,), error_callback=ec
            )
            async_results.append(result)

            sleep_count = 0
            while check_full_f():
                async_results = _monitoring_async_tasks(async_results)
                time.sleep(0.01)  # waiting for coping queue in _add_write_que_async
                sleep_count += 1
                if sleep_count > 60 * 3 / 0.001:
                    break  # exit infinite loop after 3 min

        while [r.ready() for r in async_results].count(False) > 0:
            async_results = _monitoring_async_tasks(async_results)
            time.sleep(0.01)  # waiting for adding write queue

        # finish and waiting for complete writing
        sink.set_finish_writing()
        while not write_async_result.ready():
            async_results = _monitoring_async_tasks(async_results)
            time.sleep(0.01)
        sink.close()

        # close and unlink shared memories
        frame_sna.unlink()
        flow_sna.unlink()

        pbar_of.close()
        pbar_ht.close()
        pbar_w.close()
    gc.collect()


def _monitoring_async_tasks(async_results):
    proceeding_thred_idxs = []
    for i, result in enumerate(async_results):
        if result.ready():
            result.get()
        else:
            proceeding_thred_idxs.append(i)

    async_results = [async_results[i] for i in proceeding_thred_idxs]

    return async_results


def _error_callback(*args):
    print(f"Error occurred in {args[0]}:\n{args[1:]}")


def _check_full(tail_of, tail_ht, head, que_len):
    is_frame_que_full = (tail_of.value + 1) % que_len == head.value
    is_idv_que_full = (tail_ht.value + 1) % que_len == head.value
    is_eq = tail_of.value == tail_ht.value
    return is_frame_que_full and is_idv_que_full and is_eq


def _optical_flow_async(cap, frame_sna, flow_sna, tail_of, head, lock, pbar):
    frame_que, frame_shm = frame_sna.ndarray()
    flow_que, flow_shm = flow_sna.ndarray()
    que_len = frame_que.shape[0]

    frame_count = cap.get_frame_count()
    prev_frame = cap.read(0)[1]
    cap.set_pos_frame_count(0)  # reset read position

    frame_que[tail_of.value] = prev_frame
    y, x = prev_frame.shape[:2]
    flow_que[tail_of.value] = np.zeros((y, x, 2), np.float32)
    tail_of.value = 1
    pbar.update()

    for n_frame in range(1, frame_count):
        frame = cap.read()[1]
        flow = video.optical_flow(prev_frame, frame)
        prev_frame = frame

        with lock:
            frame_que[tail_of.value] = frame
            flow_que[tail_of.value] = flow
        pbar.update()

        if n_frame + 1 == frame_count:
            break  # finish

        next_tail = (tail_of.value + 1) % que_len
        while next_tail == head.value:
            time.sleep(0.001)
        tail_of.value = next_tail

    frame_shm.close()
    flow_shm.close()
    del cap, frame_que, flow_que


def _human_tracking_async(
    cap, json_path, model, ht_que, n_frames_que, tail_ht, head, lock, pbar
):
    que_len = len(ht_que)

    do_human_tracking = not os.path.exists(json_path)
    if not do_human_tracking:
        json_data = json_handler.load(json_path)

    frame_count = cap.get_frame_count()
    for n_frame in range(frame_count):
        if do_human_tracking:
            frame = cap.read()[1]
            idvs_tmp = model.predict(frame, n_frame)
        else:
            idvs_tmp = [idv for idv in json_data if idv["n_frame"] == n_frame]

        with lock:
            ht_que[tail_ht.value] = idvs_tmp
            n_frames_que[tail_ht.value] = n_frame
            pbar.update()

        if n_frame + 1 == frame_count:
            break  # finish

        next_tail = (tail_ht.value + 1) % que_len
        while next_tail == head.value:
            time.sleep(0.001)
        tail_ht.value = next_tail

    del cap


def _add_write_que_async(
    n_frame,
    n_frames_que,
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
    with lock:
        # copy queue
        frame_que, frame_shm = frame_sna.ndarray()
        flow_que, flow_shm = flow_sna.ndarray()
        copy_frame_que = copy.deepcopy(frame_que)
        copy_flow_que = copy.deepcopy(flow_que)
        frame_shm.close()
        flow_shm.close()
        del frame_que, flow_que
        copy_n_frames_que = copy.deepcopy(list(n_frames_que))
        copy_ht_que = copy.deepcopy(list(ht_que))

        # update head
        head_val_copy = int(head.value)
        head.value = (head.value + stride) % seq_len

    # sort to start from head
    sorted_idxs = list(range(head_val_copy, seq_len)) + list(range(0, head_val_copy))
    copy_n_frames_que = [copy_n_frames_que[idx] for idx in sorted_idxs]
    copy_frame_que = copy_frame_que[sorted_idxs].astype(np.uint8)
    copy_flow_que = copy_flow_que[sorted_idxs].astype(np.float32)
    copy_ht_que = [copy_ht_que[idx] for idx in sorted_idxs]

    # check data
    assert (
        n_frame % seq_len == head_val_copy
    ), f"n_frame % seq_len:{n_frame % seq_len}, head:{head_val_copy}"
    assert copy_n_frames_que == list(
        range(n_frame - seq_len, n_frame)
    ), f"copy_n_frames_que:{copy_n_frames_que}"
    assert (
        n_frame == copy_n_frames_que[-1] + 1
    ), f"n_frame:{n_frame}, copy_n_frames_que[-1] + 1:{copy_n_frames_que[-1] + 1}"

    # clip frames and flows by bboxs
    idv_frames, idv_flows = clip_images_by_bbox(
        copy_frame_que, copy_flow_que, copy_ht_que, resize
    )
    if len(idv_frames) == 0:
        # There are no individuals within frames for seq_len (not error)
        pbar.update()
        del copy_frame_que, copy_flow_que, copy_ht_que
        gc.collect()
        return

    # collect human tracking data
    unique_ids = set(
        itertools.chain.from_iterable(
            [[idv["id"] for idv in idvs] for idvs in copy_ht_que]
        )
    )
    unique_ids = sorted(list(unique_ids))
    meta, ids, bboxs, kps = collect_human_tracking(copy_ht_que, unique_ids)
    frame_size = np.array(copy_frame_que.shape[1:3])  # (h, w)

    if dataset_type == "individual":
        idv_npzs, unique_ids = individual_to_npz(
            meta, unique_ids, idv_frames, idv_flows, bboxs, kps, frame_size
        )
        for i, _id in enumerate(unique_ids):
            data = {"__key__": f"{video_name}_{n_frame}_{_id}", "npz": idv_npzs[i]}
            sink.add_write_que(data)
            del data
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
                "frame_size": frame_size,
            },
        }
        sink.add_write_que(data)
        del data
    else:
        raise ValueError

    pbar.update()
    del copy_frame_que, copy_flow_que, copy_ht_que
    gc.collect()
