"""ViTPose-L finetuned on SwimXYZ swimmers, with the same interface as rtmlib's RTMPose.

Weights: https://huggingface.co/willalexm/vitpose-l-swimxyz (Apache-2.0). The
model was trained on synthetic swimmers only (SwimXYZ), so check it on real
footage before trusting it (compare_on_track.py).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rtmlib.tools.base import BaseTool
from rtmlib.tools.pose_estimation.pre_processings import (
    bbox_xyxy2cs,
    top_down_affine,
)


VITPOSE_SWIM_MODEL = Path(__file__).resolve().parent.parent / (
    "models/vitpose-l-swimxyz-mv.onnx"
)
VITPOSE_SWIM_URL = (
    "https://huggingface.co/willalexm/vitpose-l-swimxyz/resolve/main/"
    "vitpose-l-swimxyz-mv.onnx"
)

# ImageNet statistics on 0-255 RGB.
IMAGENET_MEAN = np.array((123.675, 116.28, 103.53))
IMAGENET_STD = np.array((58.395, 57.12, 57.375))


def decode_heatmaps(heatmaps: np.ndarray):
    """Return (K, 2) heatmap-pixel locations and (K,) peak values.

    Each peak moves a quarter pixel toward its higher neighbor, the usual
    ViTPose/SimpleBaseline refinement.
    """
    count, height, width = heatmaps.shape
    flat = heatmaps.reshape(count, -1)
    index = flat.argmax(axis=1)
    scores = flat.max(axis=1)
    x = (index % width).astype(float)
    y = (index // width).astype(float)
    for k in range(count):
        px, py = int(x[k]), int(y[k])
        if 0 < px < width - 1:
            x[k] += 0.25 * np.sign(heatmaps[k, py, px + 1] - heatmaps[k, py, px - 1])
        if 0 < py < height - 1:
            y[k] += 0.25 * np.sign(heatmaps[k, py + 1, px] - heatmaps[k, py - 1, px])
    return np.stack([x, y], axis=1), scores


class ViTPoseSwim(BaseTool):
    """Top-down swimmer pose: model(image, bboxes=[xyxy]) -> keypoints, scores."""

    def __init__(self, onnx_model: str | Path = VITPOSE_SWIM_MODEL, device: str = "cpu"):
        if not Path(onnx_model).is_file():
            raise FileNotFoundError(
                f"{onnx_model} not found; download it from {VITPOSE_SWIM_URL}"
            )
        super().__init__(
            str(onnx_model),
            model_input_size=(192, 256),
            backend="onnxruntime",
            device=device,
        )

    def __call__(self, image: np.ndarray, bboxes: list = []):
        if len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]
        keypoints, scores = [], []
        for bbox in bboxes:
            center, scale = bbox_xyxy2cs(np.array(bbox), padding=1.25)
            crop, scale = top_down_affine(self.model_input_size, scale, center, image)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = (crop - IMAGENET_MEAN) / IMAGENET_STD
            heatmaps = self.inference(crop)[0][0]
            locs, score = decode_heatmaps(heatmaps)
            # Heatmap pixels -> model input pixels -> image pixels.
            locs = locs * np.array(self.model_input_size) / heatmaps.shape[:0:-1]
            locs = locs / self.model_input_size * scale + center - scale / 2
            keypoints.append(locs[None])
            scores.append(score[None])
        return np.concatenate(keypoints), np.concatenate(scores)
