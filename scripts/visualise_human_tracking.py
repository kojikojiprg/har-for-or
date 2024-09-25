import argparse
import sys

import cv2
import numpy as np
from tqdm import tqdm
from webdataset import WebLoader

sys.path.append(".")
from src.data import load_dataset
from src.utils import video, vis, yaml_handler


def plot_on_frame(frame, _id, bbox, kps, frame_size):
    bbox = (bbox.copy() + 1) / 2 * frame_size

    # id
    pt = tuple(np.mean(bbox, axis=0).astype(int))
    frame = cv2.putText(
        frame, str(_id), pt, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2
    )

    kps = (kps.copy() + 1) / 2 * (bbox[1] - bbox[0]) + bbox[0]
    frame = vis.draw_skeleton(frame, kps, color=(0, 255, 0))

    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    args = parser.parse_args()
    data_root = args.data_root

    # data_dirs = sorted(glob(os.path.join(data_root, "*/")))
    data_dirs = [data_root]
    data_dir = data_dirs[0]

    config = yaml_handler.load("configs/individual-sqvae.yaml")
    seq_len = config.seq_len
    stride = config.stride

    # load dataset
    dataset, n_samples = load_dataset(data_dirs, "individual", config, False)
    dataloader = WebLoader(dataset, num_workers=1, pin_memory=True)

    # load video
    video_path = f"{data_dir}.mp4"
    cap = video.Capture(video_path)
    max_n_frame = cap.frame_count
    frame_size = cap.size

    # create writers
    wrt = video.Writer(f"{data_dir}/vis_pose.mp4", cap.fps, cap.size)

    pre_n_frame = seq_len
    data_tmp = []
    for batch in tqdm(dataloader, total=n_samples, ncols=100):
        keys, ids, kps, bbox, mask = batch
        if kps.ndim == 5:
            ids = ids[0]
            kps = kps[0]
            bbox = bbox[0]

        n_frame = int(keys[0].split("_")[1])
        _id = ids.cpu().numpy()[0]
        kps = kps.cpu().numpy()[0]
        bbox = bbox.cpu().numpy()[0]

        # plot
        if pre_n_frame < n_frame:
            for i in range(stride):
                n_frame_tmp = pre_n_frame - stride + i
                idx_data = seq_len - stride + i

                frame = cap.read(n_frame_tmp)[1]
                frame = cv2.putText(
                    frame,
                    f"frame:{n_frame_tmp}",
                    (10, 40),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1.0,
                    (255, 255, 255),
                    1,
                )
                for _id_tmp, kps_tmp, bbox_tmp in data_tmp:
                    frame = plot_on_frame(
                        frame, _id_tmp, bbox_tmp[idx_data], kps_tmp[idx_data], frame_size
                    )

                wrt.write(frame)

            data_tmp = []
            pre_n_frame = n_frame

        # add result in temporary result list
        data_tmp.append((_id, kps, bbox))

    del cap, wrt
