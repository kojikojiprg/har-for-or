import os
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
from mmpose.apis import inference_topdown, init_model
from mmpose.evaluation.functional import nms
from mmpose.structures import PoseDataSample
from numpy.typing import NDArray
from ultralytics import YOLO


class Detector:
    def __init__(
        self, cfg: SimpleNamespace, device: str, model_cache_dir: str = "models"
    ):
        self._cfg = cfg

        yolo_model = cfg.yolo.model
        model_path = os.path.join(model_cache_dir, "yolo", yolo_model)
        if os.path.exists(model_path):
            self._yolo = YOLO(model_path)
        else:
            self._yolo = YOLO(yolo_model)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.rename(yolo_model, model_path)
        self._yolo = self._yolo.to(device)

        self._pose_model = init_model(cfg.pose.config, cfg.pose.weights, device=device)

    def __del__(self):
        del self._yolo, self._pose_model

    def predict(self, img: np.array) -> Tuple[NDArray, NDArray]:
        bboxs = self._yolo.predict(img, verbose=False)[0].boxes.data.cpu().numpy()
        bboxs = self._process_yolo_results(bboxs)

        pose_results = inference_topdown(
            self._pose_model,
            img,
            bboxs[:, :4],
            bbox_format="xyxy",
        )
        kps = self._collect_kps(pose_results)

        # extract unique result
        if len(bboxs) > 0:
            remain_indices = self._del_leaky(kps)
            remain_indices = self._get_unique(kps, remain_indices)
            bboxs = bboxs[remain_indices]
            kps = kps[remain_indices]

        return bboxs, kps

    def _process_yolo_results(self, bboxs):
        bboxs = bboxs[
            np.logical_and(bboxs[:, 4] > self._cfg.yolo.th_conf, bboxs[:, 5] == 0)
        ]
        bboxs = bboxs[nms(bboxs, self._cfg.yolo.th_iou), :5]
        # areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        # bboxes = bboxes[
        #     (self._cfg.yolo.min_area < areas) & (areas < self._cfg.yolo.max_area)
        # ]
        return bboxs

    # @staticmethod
    # def _collect_bboxs_from_pose(det_results: List[PoseDataSample]):
    #     return np.array([result.pred_instances.bboxes[0] for result in det_results])

    @staticmethod
    def _collect_kps(det_results: List[PoseDataSample]):
        return np.array(
            [
                np.hstack(
                    [
                        result.pred_instances.keypoints[0],
                        result.pred_instances.keypoint_scores.T,
                    ]
                )
                for result in det_results
            ]
        )

    def _del_leaky(self, kps: NDArray):
        return np.where(np.mean(kps[:, :, 2], axis=1) >= self._cfg.pose.th_delete)[0]

    def _get_unique(self, kps: NDArray, indices: NDArray):
        remain_indices = np.empty((0,), dtype=np.int32)

        for idx in indices:
            for ridx in remain_indices:
                # calc diff of all points
                diff = np.linalg.norm(kps[idx, :, :2] - kps[ridx, :, :2], axis=1)

                if (
                    len(np.where(diff < self._cfg.pose.th_diff)[0])
                    >= self._cfg.pose.th_count
                ):
                    # found overlap
                    if np.mean(kps[idx, :, 2]) > np.mean(kps[ridx, :, 2]):
                        # select one which is more confidence
                        remain_indices[remain_indices == ridx] = idx

                    break
            else:
                # if there aren't overlapped
                remain_indices = np.append(remain_indices, idx)

        # return unique_kps
        return remain_indices
