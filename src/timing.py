"""Find the start, wall touches and finish from how the swimmer moves.

The camera usually pans to follow the swimmer, so the swimmer's own motion
is their motion in the picture minus the camera's (the median optical flow
of everything outside the swimmer's box). Its main direction is up and down
the pool; along it, the speed is signed:

- start: the first stretch of steady motion (the dive or push-off),
- wall touch: the sign flips and stays flipped (the swimmer turned),
- finish: motion stops for good before the video ends.

Pixels are not meters (the camera zooms and the pool is seen at an angle),
so only the times are used; the pool length turns them into speeds. If the
swimmer never stops before the video ends, there is no finish and the clip
is treated as part of a swim (stroke rhythm only).
"""

from __future__ import annotations

import cv2
import numpy as np


FLOW_WIDTH = 320
# Smoothing of the speed along the pool, in seconds.
SMOOTH_S = 1.0
# Moving: speed above this share of the typical (75th percentile) speed.
MOVING_SHARE = 0.3
MIN_MOVE_S = 1.0
MAX_PAUSE_S = 0.7
# A turn: the direction holds for this long on both sides.
TURN_HOLD_S = 2.0
# A finish must stop at least this long before the video ends.
FINISH_MARGIN_S = 0.5


def camera_flow(video_path: str, boxes: list[list[float] | None]) -> np.ndarray:
    """Per frame, the median optical flow (x, y) outside the swimmer's box,
    in full-size pixels: how the picture moves as the camera pans."""
    capture = cv2.VideoCapture(video_path)
    scale = FLOW_WIDTH / capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    flow = np.zeros((len(boxes), 2))
    previous = None
    index = 0
    while index < len(boxes):
        success, frame = capture.read()
        if not success:
            break
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        if previous is not None:
            field = cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 3, 15, 2, 5, 1.1, 0)
            outside = np.ones(gray.shape, bool)
            if boxes[index] is not None:
                x1, y1, x2, y2 = (int(v * scale) for v in boxes[index])
                outside[max(0, y1 - 5) : y2 + 5, max(0, x1 - 5) : x2 + 5] = False
            flow[index] = np.median(field[outside], axis=0) / scale
        previous = gray
        index += 1
    capture.release()
    return flow


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    known = np.isfinite(values)
    if known.sum() < 2:
        return np.zeros_like(values)
    filled = np.interp(np.arange(len(values)), np.flatnonzero(known), values[known])
    window = max(1, window)
    padded = np.concatenate(([filled[0]] * (window // 2), filled, [filled[-1]] * (window - 1 - window // 2)))
    return np.convolve(padded, np.ones(window) / window, "valid")


def along_pool_speed(centroids: np.ndarray, flow: np.ndarray, fps: float) -> np.ndarray:
    """Signed swimmer speed along their main direction, smoothed, in pixels
    per frame. Where the swimmer is not tracked, the camera is assumed to
    keep them still in the picture."""
    picture = np.vstack([[np.nan, np.nan], np.diff(centroids, axis=0)])
    picture[~np.isfinite(picture)] = 0.0
    own = picture - flow
    own = np.column_stack([smooth(own[:, k], int(0.5 * fps)) for k in range(2)])
    strong = own[np.linalg.norm(own, axis=1) > np.percentile(np.linalg.norm(own, axis=1), 50)]
    # Main direction: the first principal axis of the strong motion, sign-free.
    _, _, axes = np.linalg.svd(strong - 0 if len(strong) else own, full_matrices=False)
    return smooth(own @ axes[0], int(SMOOTH_S * fps))


def stretches(mask: np.ndarray, fps: float) -> list[tuple[int, int]]:
    """(first, last) frames of True runs, short pauses bridged, short runs dropped."""
    runs, start = [], None
    for index, value in enumerate(np.append(mask, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append([start, index - 1])
            start = None
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= MAX_PAUSE_S * fps:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [(a, b) for a, b in merged if b - a >= MIN_MOVE_S * fps]


def find_timing(speed: np.ndarray, fps: float) -> dict:
    """{"start": s, "touches": [s, ...], "finished": bool} or {} if no swim."""
    typical = np.percentile(np.abs(speed), 75)
    if typical <= 0:
        return {}
    moving = stretches(np.abs(speed) > MOVING_SHARE * typical, fps)
    if not moving:
        return {}
    start, end = moving[0][0], moving[-1][1]
    finished = end < len(speed) - 1 - FINISH_MARGIN_S * fps
    hold = int(TURN_HOLD_S * fps)
    touches = []
    for index in range(start + hold, end - hold):
        if np.sign(speed[index]) != np.sign(speed[index + 1]):
            before, after = speed[index - hold : index].mean(), speed[index + 1 : index + 1 + hold].mean()
            threshold = MOVING_SHARE * typical
            if np.sign(before) != np.sign(after) and min(abs(before), abs(after)) > threshold:
                touches.append(index / fps)
    if finished:
        touches.append(end / fps)
    return {"start": start / fps, "touches": touches, "finished": finished}
