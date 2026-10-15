"""Detect one swimmer and export a video with pose landmarks drawn on it.

RTMPose-x finds the keypoints. A person detector picks the swimmer up; after
that, each frame's keypoints place the box for the next frame (see
SwimmerTracker), because detectors rarely see a swimmer in splash.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from lanes import LaneTracker, water_fraction


SWIM_SKELETON = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

SWIM_KEYPOINT_INDICES = tuple(range(5, 17))

# Shoulder, elbow, wrist, hip, knee, ankle for each body side.
SIDE_KEYPOINTS = {
    "left": (5, 7, 9, 11, 13, 15),
    "right": (6, 8, 10, 12, 14, 16),
}

HOLD_FRAMES = 5

# Minimum overlap with the previous swimmer box, as a fraction of the smaller
# box, to keep following the same swimmer (compare_models.py).
MIN_TRACK_OVERLAP = 0.3
# While the pose is unsure, look for the swimmer with the detector this often.
REDETECT_S = 0.5
# ...and only accept a person centered within this many box sides of it.
RECHECK_REACH = 0.75
# Largest move of the box center in one frame, in box sides.
MAX_STEP_X = 0.6
MAX_STEP_Y = 0.25
# While unsure, also try these boxes: (x shift, y shift) in box sides, and scale.
SEARCH_OFFSETS = ((-0.5, 0, 1), (0.5, 0, 1), (0, 0, 1.6))
# With the lane known, unsure frames sweep along it this many half-box steps
# each way instead, since the swimmer can be well ahead of a held box.
LANE_SEARCH_STEPS = 3
# A pose counts as in the lane within this many lane widths beyond each rope
# (from the side, head and arms reach over the far rope).
LANE_MARGIN = 0.5
# Drop the lane after this many seconds with neither rope found (scene cuts).
LANE_LOST_S = 0.5
# Lost this many seconds: sweep the whole frame's water for the swimmer every
# LOST_SWEEP_FRAMES frames, taking a pose at least this sure.
LOST_S = 1.0
LOST_SWEEP_FRAMES = 5
LOST_SWEEP_SCORE = 0.35
# Lock the lane after this many seconds of steady, flat swimming in water.
LANE_LOCK_S = 0.5
# Lock the lane only when at least this share of the swimmer box is water
# (about 0.03 on the block, 0.3 to 0.8 swimming).
LANE_WATER_FRACTION = 0.25
# A shoulder-to-ankle keypoint counts toward tracking at this confidence; the
# box moves only when this many are confident and their mean reaches it too.
TRACK_KEYPOINT_CONFIDENCE = 0.3
TRACK_MIN_KEYPOINTS = 5
# The box size is learned only from frames with this many confident keypoints.
TRACK_SIZE_KEYPOINTS = 8
# Box side as a multiple of the keypoint spread, and its smallest side in pixels.
TRACK_BOX_MARGIN = 1.3
TRACK_MIN_SIDE = 64
# Display box: centered on the torso, smoothed over this many seconds.
DISPLAY_SMOOTHING_S = 0.3
# Display box size covers this percentile of the swimmer's keypoint spread.
DISPLAY_SIZE_PERCENTILE = 90

RTMPOSE_DETECTOR = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_x_8xb8-300e_humanart-a39d44ed.zip"
)
# Smaller detector for the tracker: it only runs to pick up and recheck.
TRACKER_DETECTOR = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
)
RTMPOSE_MODEL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip"
)

COCO_LANDMARK_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def select_person(boxes, previous_box=None) -> int | None:
    """Return the index of the box that continues the tracked swimmer.

    With no track yet, picks the largest box. Otherwise picks the box that
    overlaps the previous swimmer box the most, measured against the smaller
    of the two boxes so a tight box inside a loose one still matches. Returns
    None when nothing overlaps enough, rather than jumping to someone else.
    """
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if previous_box is None:
        return int(areas.argmax())
    x1 = np.maximum(boxes[:, 0], previous_box[0])
    y1 = np.maximum(boxes[:, 1], previous_box[1])
    x2 = np.minimum(boxes[:, 2], previous_box[2])
    y2 = np.minimum(boxes[:, 3], previous_box[3])
    overlap = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    previous_area = (previous_box[2] - previous_box[0]) * (
        previous_box[3] - previous_box[1]
    )
    smaller = np.maximum(np.minimum(areas, previous_area), 1.0)
    match = overlap / smaller
    if match.max() < MIN_TRACK_OVERLAP:
        return None
    return int(match.argmax())


def to_local(box, offset_x: int, offset_y: int):
    """Convert a full-frame box to inference-crop coordinates."""
    if box is None:
        return None
    return (
        box[0] - offset_x,
        box[1] - offset_y,
        box[2] - offset_x,
        box[3] - offset_y,
    )


def get_swimmer_pose(
    result, offset_x: int = 0, offset_y: int = 0, previous_box=None
):
    """Return the tracked swimmer's box and keypoints."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    if result.keypoints is None or len(result.keypoints) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    person_index = select_person(
        boxes, to_local(previous_box, offset_x, offset_y)
    )
    if person_index is None:
        return None
    box = boxes[person_index].astype(float)
    box[0::2] += offset_x
    box[1::2] += offset_y

    keypoints = result.keypoints.xy[person_index].cpu().numpy().astype(float)
    keypoints[:, 0] += offset_x
    keypoints[:, 1] += offset_y
    scores = result.keypoints.conf
    if scores is None:
        scores = [1.0] * len(keypoints)
    else:
        scores = scores[person_index].cpu().numpy().tolist()

    return box.tolist(), keypoints.tolist(), scores


class YoloRunner:
    """Run an Ultralytics pose model and return the tracked swimmer."""

    def __init__(self, model_name: str, image_size: int, confidence: float):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.image_size = image_size
        self.confidence = confidence

    def __call__(self, image, offset_x: int, offset_y: int, previous_box=None):
        result = self.model(
            image,
            imgsz=self.image_size,
            conf=self.confidence,
            verbose=False,
        )[0]
        return get_swimmer_pose(result, offset_x, offset_y, previous_box)


class RtmposeRunner:
    """Run YOLOX person detection followed by RTMPose-x on the swimmer box."""

    def __init__(self, confidence: float):
        from rtmlib import RTMPose, YOLOX

        self.detector = YOLOX(RTMPOSE_DETECTOR, score_thr=confidence)
        self.pose_model = RTMPose(RTMPOSE_MODEL, model_input_size=(288, 384))

    def __call__(self, image, offset_x: int, offset_y: int, previous_box=None):
        boxes = np.asarray(self.detector(image))
        if boxes.size == 0:
            return None
        person_index = select_person(
            boxes, to_local(previous_box, offset_x, offset_y)
        )
        if person_index is None:
            return None
        box = boxes[person_index].astype(float)
        keypoints, scores = self.pose_model(image, bboxes=[box.tolist()])
        keypoints = keypoints[0].astype(float)
        keypoints[:, 0] += offset_x
        keypoints[:, 1] += offset_y
        box[0::2] += offset_x
        box[1::2] += offset_y
        return box.tolist(), keypoints.tolist(), scores[0].tolist()


def square_box(center_x: float, center_y: float, side: float):
    """Return a square box with the given center and side length."""
    half = side / 2
    return [center_x - half, center_y - half, center_x + half, center_y + half]


def box_contains(box, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def is_horizontal(keypoints, scores) -> bool:
    """Whether the torso is confidently found and lies closer to flat than upright."""
    if scores[[5, 6, 11, 12]].min() < TRACK_KEYPOINT_CONFIDENCE:
        return False
    shoulders = (keypoints[5] + keypoints[6]) / 2
    hips = (keypoints[11] + keypoints[12]) / 2
    torso = np.abs(hips - shoulders)
    return bool(torso[0] > torso[1])


class SwimmerTracker:
    """Follow one swimmer by feeding each frame's pose into the next frame.

    Person detectors rarely find a swimmer lying flat in splash, so the
    detector only picks the swimmer up (at the --pick point, or the largest
    person) and rechecks while the pose is unsure. Between those, the box
    follows the swimmer's own keypoints: its center moves to the confident
    keypoints and its size is learned from frames where most of the body is
    visible. Unsure frames hold the box in place instead of shrinking it, and
    the detector may only move it to a person near where the swimmer was.
    pick is the swimmer's (x, y) in image pixels on frame pick_frame.
    """

    def __init__(
        self,
        detection_confidence: float,
        device: str,
        rotate: bool,
        redetect_frames: int,
        pick=None,
        pick_frame: int = 0,
        lane_lost_frames: int = 15,
        lost_frames: int = 30,
        lane_lock_frames: int = 15,
    ):
        from rtmlib import RTMPose, YOLOX

        # CoreML cannot run YOLOX's built-in box filtering, so it stays on CPU.
        self.detector = YOLOX(
            TRACKER_DETECTOR, score_thr=detection_confidence, device="cpu"
        )
        self.pose_model = RTMPose(
            RTMPOSE_MODEL, model_input_size=(288, 384), device=device
        )
        self.rotations = (0, 90, -90) if rotate else (0,)
        self.redetect_frames = redetect_frames
        self.pick = pick
        self.pick_frame = pick_frame
        self.box = None
        self.side = None
        self.unsure_frames = 0
        self.frame_number = -1
        self.lost_frames = lost_frames
        self.lane_lock_frames = lane_lock_frames
        self.lane = LaneTracker(lane_lost_frames)
        self.horizontal = False
        self.swimming_frames = 0
        self.in_pool = False

    def __call__(self, image, offset_x: int, offset_y: int):
        """Return the swimmer's (box, keypoints, scores) in full-frame pixels."""
        self.frame_number += 1
        if self.box is None:
            if self.frame_number < self.pick_frame:
                return None
            self.box = self.pick_up(image)
            if self.box is None:
                return None
        elif (
            self.unsure_frames > self.lost_frames
            and self.frame_number % LOST_SWEEP_FRAMES == 0
        ):
            self.sweep(image)
        elif self.unsure_frames and self.frame_number % self.redetect_frames == 0:
            self.recheck(image)
        self.update_lane(image)

        keypoints, scores = self.best_pose(image, self.box)
        if self.unsure_frames:
            keypoints, scores = self.search(image, keypoints, scores)
        self.follow(image, keypoints, scores)
        box = list(self.box)
        box[0::2] = [value + offset_x for value in box[0::2]]
        box[1::2] = [value + offset_y for value in box[1::2]]
        keypoints[:, 0] += offset_x
        keypoints[:, 1] += offset_y
        return box, keypoints.tolist(), scores.tolist()

    def center(self, box=None):
        box = self.box if box is None else box
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    def update_lane(self, image) -> None:
        """Follow the swimmer's lane, locking on once they swim in it.

        The lane is only locked from a confident, horizontal pose surrounded
        by water, so a swimmer on the block or deck (a start crouch is
        horizontal too) does not lock a wrong lane.
        """
        center_x, center_y = self.center()
        if self.lane.locked():
            self.lane.update(image, center_x)
            return
        # Wait for steady swimming: a dive is horizontal over water briefly.
        if (
            self.unsure_frames == 0
            and self.horizontal
            and water_fraction(image, self.box) >= LANE_WATER_FRACTION
        ):
            self.swimming_frames += 1
        else:
            self.swimming_frames = 0
        if self.swimming_frames >= self.lane_lock_frames:
            self.lane.start(image, center_x, center_y)

    def detect(self, image):
        return np.asarray(self.detector(image), dtype=float).reshape(-1, 4)

    def pick_up(self, image):
        """Return the starting box: the person at the pick point, or the largest."""
        boxes = self.detect(image)
        if self.pick is None:
            if len(boxes) == 0:
                return None
            index = select_person(boxes)
            self.side = self.box_side(boxes[index])
            return self.square(boxes[index])

        x, y = self.pick
        hits = [box for box in boxes if box_contains(box, x, y)]
        if hits:
            box = min(hits, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            self.side = self.box_side(box)
            return self.square(box)
        # No detection there: start around the point and learn the size.
        self.side = 0.25 * min(image.shape[:2])
        return square_box(x, y, self.side)

    def recheck(self, image) -> None:
        """Move the box to a detected person near the swimmer, if there is one.

        Only people centered within RECHECK_REACH box sides of the swimmer, of
        similar size, and lying flat count, so bystanders on the deck (a
        coach next to the block, say) are never taken.
        """
        center = np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])
        best = None
        for box in self.detect(image):
            if not 1 / 3 <= self.box_side(box) / self.side <= 3:
                continue
            if not self.lane.contains(*self.center(box), LANE_MARGIN):
                continue
            distance = np.abs((box[:2] + box[2:]) / 2 - center).max()
            if distance > self.side * RECHECK_REACH or (
                best is not None and distance >= best[0]
            ):
                continue
            if is_horizontal(*self.pose(image, self.square(box), 0)):
                best = (distance, box)
        if best is not None:
            self.box = self.square(best[1])

    def sweep(self, image) -> None:
        """Look over all the water in the frame for the lost swimmer.

        Used after a dive, a long underwater stretch, or a fast camera swing,
        when the swimmer may be anywhere. Only boxes that are mostly water
        (and inside the lane, once known) are tried.
        """
        height, width = image.shape[:2]
        side = float(np.clip(self.side, 0.1 * width, 0.5 * width))
        step = side / 2
        body = list(SWIM_KEYPOINT_INDICES)
        best = None
        for center_y in np.arange(side / 2, height, step):
            for center_x in np.arange(side / 2, width, step):
                box = square_box(center_x, center_y, side)
                if water_fraction(image, box) < LANE_WATER_FRACTION:
                    continue
                if not self.lane.contains(center_x, center_y, LANE_MARGIN):
                    continue
                keypoints, scores = self.pose(image, box, 0)
                score = scores[body].mean()
                # Only a swimmer lies flat; this skips people on the deck.
                if not is_horizontal(keypoints, scores):
                    continue
                if best is None or score > best[0]:
                    best = (score, box)
        if best is not None and best[0] >= LOST_SWEEP_SCORE:
            self.box = best[1]
            self.unsure_frames = 0

    def search(self, image, keypoints, scores):
        """Look beside the held box for the swimmer and keep the surest pose.

        While unsure the box stays put but the swimmer and the camera keep
        moving, mostly sideways, so try shifted and wider boxes too.
        """
        body = list(SWIM_KEYPOINT_INDICES)
        best = (scores[body].mean(), keypoints, scores)
        center_x, center_y = self.center()
        if self.lane.locked():
            return self.search_lane(image, best)
        for shift_x, shift_y, scale in SEARCH_OFFSETS:
            box = square_box(
                center_x + shift_x * self.side,
                center_y + shift_y * self.side,
                self.side * scale,
            )
            keypoints, scores = self.best_pose(image, box)
            if scores[body].mean() > best[0]:
                best = (scores[body].mean(), keypoints, scores)
        return best[1], best[2]

    def search_lane(self, image, best):
        """Sweep boxes along the lane's center line and keep the surest pose.

        The sweep runs the pose model upright only, to keep it cheap; the
        winning box then gets the full pass with sideways crops.
        """
        body = list(SWIM_KEYPOINT_INDICES)
        center_x, _ = self.center()
        best_box = None
        best_score = best[0]
        for step in range(-LANE_SEARCH_STEPS, LANE_SEARCH_STEPS + 1):
            x = center_x + step * 0.5 * self.side
            y = self.lane.center_y(x)
            if step == 0 or y is None or not 0 <= x <= image.shape[1]:
                continue
            box = square_box(x, y, self.side)
            _, scores = self.pose(image, box, 0)
            if scores[body].mean() > best_score:
                best_score, best_box = scores[body].mean(), box
        if best_box is None:
            return best[1], best[2]
        keypoints, scores = self.best_pose(image, best_box)
        if scores[body].mean() > best[0]:
            return keypoints, scores
        return best[1], best[2]

    def best_pose(self, image, box):
        """Run the pose model upright and on sideways crops; keep the surest."""
        best = None
        for rotation in self.rotations:
            keypoints, scores = self.pose(image, box, rotation)
            score = scores[list(SWIM_KEYPOINT_INDICES)].mean()
            if best is None or score > best[0]:
                best = (score, keypoints, scores)
        return best[1], best[2]

    def pose(self, image, box, rotation: int):
        """Run RTMPose on box, turning the crop by rotation degrees first.

        The model mostly saw upright people, so a horizontal swimmer turned on
        end can score better. Keypoints come back in image coordinates.
        """
        if rotation == 0:
            keypoints, scores = self.pose_model(image, bboxes=[list(box)])
            return keypoints[0].astype(float), scores[0].astype(float)
        height, width = image.shape[:2]
        x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
        x2, y2 = min(int(box[2]), width), min(int(box[3]), height)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((17, 2)), np.zeros(17)
        turn = cv2.ROTATE_90_CLOCKWISE if rotation == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE
        turned = cv2.rotate(crop, turn)
        keypoints, scores = self.pose_model(
            turned, bboxes=[[0, 0, turned.shape[1], turned.shape[0]]]
        )
        u, v = keypoints[0][:, 0].astype(float), keypoints[0][:, 1].astype(float)
        crop_height, crop_width = crop.shape[:2]
        if rotation == 90:
            x, y = v, crop_height - 1 - u
        else:
            x, y = crop_width - 1 - v, u
        return np.stack([x + x1, y + y1], axis=1), scores[0].astype(float)

    def follow(self, image, keypoints, scores) -> None:
        """Move the box to the confident keypoints; hold it when unsure."""
        body = list(SWIM_KEYPOINT_INDICES)
        confident = scores[body] >= TRACK_KEYPOINT_CONFIDENCE
        if (
            confident.sum() < TRACK_MIN_KEYPOINTS
            or scores[body].mean() < TRACK_KEYPOINT_CONFIDENCE
        ):
            self.unsure_frames += 1
            # Lost after a start on the block: the swimmer is in the pool now.
            if self.unsure_frames > self.lane_lock_frames:
                self.in_pool = True
            return
        points = keypoints[body][confident]
        center_x, center_y = np.median(points, axis=0)
        # A swimmer stays in their lane: a jump up or down means the pose
        # landed on someone or something else. Sideways, the swimmer keeps
        # moving while the box is held, so allow more the longer it was held.
        # Once the lane is known it replaces the up/down limit.
        step_x = abs(center_x - (self.box[0] + self.box[2]) / 2) / self.side
        step_y = abs(center_y - (self.box[1] + self.box[3]) / 2) / self.side
        reach_x = MAX_STEP_X * min(1 + self.unsure_frames, 4)
        if self.lane.locked():
            off_course = not self.lane.contains(center_x, center_y, LANE_MARGIN)
        else:
            off_course = step_y > MAX_STEP_Y
        horizontal = is_horizontal(keypoints, scores)
        if self.in_pool and not horizontal:
            # In the pool a swimmer lies flat or, at a wall, is in the water;
            # an upright pose out of the water is someone on the deck.
            points_box = square_box(center_x, center_y, self.side)
            off_course |= water_fraction(image, points_box) < LANE_WATER_FRACTION
        if step_x > reach_x or off_course:
            self.unsure_frames += 1
            return
        self.unsure_frames = 0
        self.horizontal = horizontal
        if horizontal and water_fraction(image, self.box) >= LANE_WATER_FRACTION:
            self.in_pool = True
        if confident.sum() >= TRACK_SIZE_KEYPOINTS:
            spread = points.max(axis=0) - points.min(axis=0)
            side = max(spread.max() * TRACK_BOX_MARGIN, TRACK_MIN_SIDE)
            self.side = 0.9 * self.side + 0.1 * side
        self.box = square_box(center_x, center_y, self.side)

    def box_side(self, box) -> float:
        return max(box[2] - box[0], box[3] - box[1], TRACK_MIN_SIDE)

    def square(self, box):
        return square_box(
            (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, self.box_side(box)
        )


class KeypointSmoother:
    """Smooth keypoints across frames and hold them briefly through dropouts."""

    def __init__(self, smoothing: float, confidence_threshold: float):
        self.smoothing = smoothing
        self.confidence_threshold = confidence_threshold
        self.points = [None] * len(COCO_LANDMARK_NAMES)
        self.missed = [0] * len(COCO_LANDMARK_NAMES)
        self.side_scores = None

    def update(self, swimmer_pose):
        """Fold one frame's pose into the smoothed points and return them."""
        keypoints = scores = None
        if swimmer_pose is not None:
            _, keypoints, scores = swimmer_pose

        for index, previous in enumerate(self.points):
            if keypoints is not None and scores[index] >= self.confidence_threshold:
                x, y = keypoints[index]
                if previous is not None:
                    x = self.smoothing * previous[0] + (1 - self.smoothing) * x
                    y = self.smoothing * previous[1] + (1 - self.smoothing) * y
                self.points[index] = (x, y)
                self.missed[index] = 0
            elif previous is not None:
                self.missed[index] += 1
                if self.missed[index] > HOLD_FRAMES:
                    self.points[index] = None

        if scores is not None:
            current = {
                side: sum(scores[index] for index in indices) / len(indices)
                for side, indices in SIDE_KEYPOINTS.items()
            }
            if self.side_scores is None:
                self.side_scores = current
            else:
                self.side_scores = {
                    side: self.smoothing * self.side_scores[side]
                    + (1 - self.smoothing) * current[side]
                    for side in current
                }
        return self.points

    def near_side(self) -> str:
        """Return the body side the camera sees most confidently."""
        if self.side_scores is None:
            return "left"
        return max(self.side_scores, key=self.side_scores.get)


def midpoint(first, second):
    """Return the midpoint of two points, or whichever one exists."""
    if first is None:
        return second
    if second is None:
        return first
    return ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)


def draw_simple_swimmer(frame, points, side: str) -> None:
    """Draw a single-line stick figure using the camera-facing body side."""
    _, elbow, wrist, _, knee, ankle = (
        points[index] for index in SIDE_KEYPOINTS[side]
    )
    neck = midpoint(points[5], points[6])
    pelvis = midpoint(points[11], points[12])
    chains = (
        (points[0], neck, pelvis),
        (neck, elbow, wrist),
        (pelvis, knee, ankle),
    )

    for chain in chains:
        for start, end in zip(chain, chain[1:]):
            if start is not None and end is not None:
                cv2.line(
                    frame,
                    tuple(int(value) for value in start),
                    tuple(int(value) for value in end),
                    (0, 220, 0),
                    3,
                    cv2.LINE_AA,
                )

    for point in (neck, elbow, wrist, pelvis, knee, ankle):
        if point is not None:
            cv2.circle(
                frame,
                tuple(int(value) for value in point),
                5,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )


def draw_swimmer(
    frame,
    swimmer_pose,
    confidence_threshold: float,
    box_padding: float = 0.1,
    box=None,
) -> None:
    """Draw the swimmer's box plus every keypoint and bone inside it.

    box overrides the pose's own detector box, e.g. with a stabilized one.
    """
    box_values, keypoints, keypoint_scores = swimmer_pose
    if box is not None:
        box_values = box
    box = [int(value) for value in box_values]
    x1, y1, x2, y2 = padded_box(box_values, box_padding, *frame.shape[1::-1])
    visible = [
        score >= confidence_threshold
        and x1 <= point[0] <= x2
        and y1 <= point[1] <= y2
        for point, score in zip(keypoints, keypoint_scores)
    ]

    for start, end in SWIM_SKELETON:
        if visible[start] and visible[end]:
            start_point = tuple(int(value) for value in keypoints[start])
            end_point = tuple(int(value) for value in keypoints[end])
            cv2.line(frame, start_point, end_point, (0, 220, 0), 2)

    for index in SWIM_KEYPOINT_INDICES:
        point = keypoints[index]
        if visible[index]:
            cv2.circle(
                frame,
                tuple(int(value) for value in point),
                4,
                (0, 120, 255),
                -1,
            )

    cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (255, 160, 0), 2)
    cv2.putText(
        frame,
        "swimmer",
        (box[0], max(box[1] - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 160, 0),
        2,
        cv2.LINE_AA,
    )


def padded_box(box, padding: float, width: int, height: int):
    """Grow a box by a fraction of its size and clamp it to the frame."""
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        max(int(x1 - pad_x), 0),
        max(int(y1 - pad_y), 0),
        min(int(x2 + pad_x), width),
        min(int(y2 + pad_y), height),
    )


def mask_outside_box(frame, box, mode: str) -> None:
    """Black out or dim everything outside the box, in place."""
    x1, y1, x2, y2 = box
    inside = frame[y1:y2, x1:x2].copy()
    if mode == "black":
        frame[:] = 0
    else:
        frame[:] = (frame * 0.25).astype(frame.dtype)
    frame[y1:y2, x1:x2] = inside


def relative_path(path: Path, start: Path) -> str:
    """Return path relative to start, as a forward-slash string."""
    return Path(os.path.relpath(path.resolve(), start.resolve())).as_posix()


def pose_from_record(record: dict):
    """Convert a landmarks.json frame record to (box, keypoints, scores)."""
    points = record["keypoints"]
    keypoints = [[points[name]["x"], points[name]["y"]] for name in COCO_LANDMARK_NAMES]
    scores = [points[name]["confidence"] for name in COCO_LANDMARK_NAMES]
    return record["box"], keypoints, scores


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average, padding the ends by repeating end values."""
    half = window // 2
    if half < 1:
        return values
    padded = np.concatenate(([values[0]] * half, values, [values[-1]] * half))
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.convolve(padded, kernel, mode="valid")


def display_boxes(
    records: dict[int, dict], total_frames: int, fps: float, confidence: float
) -> dict[int, list[float]]:
    """Return a steady box for every frame with a detected swimmer.

    The detector's boxes change size every frame, so the drawn box uses one
    size for the whole video, big enough to hold the swimmer's keypoints in
    most frames, and follows the torso center smoothed over time. Only
    RTMPose frames set the size and center; YOLO fallback frames, whose
    boxes cover just the upper body, are interpolated.
    """
    torso = [5, 6, 11, 12]
    centers = np.full((total_frames, 2), np.nan)
    half_widths = []
    half_heights = []
    for index, record in records.items():
        if record.get("model", "rtmpose") != "rtmpose":
            continue
        _, keypoints, scores = pose_from_record(record)
        if any(scores[i] < confidence for i in torso):
            continue
        points = np.array(keypoints)
        center = points[torso].mean(axis=0)
        centers[index] = center
        confident = points[[i for i, score in enumerate(scores) if score >= confidence]]
        half_widths.append(np.abs(confident[:, 0] - center[0]).max())
        half_heights.append(np.abs(confident[:, 1] - center[1]).max())

    known = np.flatnonzero(np.isfinite(centers[:, 0]))
    if len(known) == 0:
        return {index: record["box"] for index, record in records.items()}
    box_width = 2 * float(np.percentile(half_widths, DISPLAY_SIZE_PERCENTILE))
    box_height = 2 * float(np.percentile(half_heights, DISPLAY_SIZE_PERCENTILE))
    frames = np.arange(total_frames)
    window = int(round(DISPLAY_SMOOTHING_S * fps))
    center_x, center_y = (
        moving_average(np.interp(frames, known, centers[known, axis]), window)
        for axis in (0, 1)
    )
    return {
        index: [
            float(center_x[index] - box_width / 2),
            float(center_y[index] - box_height / 2),
            float(center_x[index] + box_width / 2),
            float(center_y[index] + box_height / 2),
        ]
        for index in records
    }


def shift_lane(lane: dict, offset_x: int, offset_y: int) -> dict:
    """Move lane ropes (y = slope * x + intercept) from crop to frame pixels."""
    return {
        name: None
        if rope is None
        else [rope[0], rope[1] + offset_y - rope[0] * offset_x]
        for name, rope in lane.items()
    }


def draw_lane(frame, lane: dict) -> None:
    """Draw the swimmer's lane ropes as thin lines across the frame."""
    width = frame.shape[1]
    for rope in lane.values():
        if rope is not None:
            slope, intercept = rope
            cv2.line(
                frame,
                (0, int(intercept)),
                (width, int(slope * width + intercept)),
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )


def run_detection(
    video_path: Path,
    detection_confidence: float,
    roi: tuple[float, float, float, float],
    device: str,
    rotate: bool,
    pick: tuple[float, float] | None,
    pick_time: float,
):
    """Run pose estimation on every frame and return the swimmer records."""
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x_start = int(roi[0] * width)
    y_start = int(roi[1] * height)
    x_end = int(roi[2] * width)
    y_end = int(roi[3] * height)

    if pick is not None:
        # Normalized full-frame point to pixels inside the inference crop.
        pick = (pick[0] * width - x_start, pick[1] * height - y_start)
    tracker = SwimmerTracker(
        detection_confidence,
        device,
        rotate,
        max(int(REDETECT_S * fps), 1),
        pick,
        int(round(pick_time * fps)),
        max(int(LANE_LOST_S * fps), 1),
        int(LOST_S * fps),
        max(int(LANE_LOCK_S * fps), 1),
    )
    records = {}
    frame_number = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            inference_frame = frame[y_start:y_end, x_start:x_end]
            swimmer_pose = tracker(inference_frame, x_start, y_start)
            if swimmer_pose is not None:
                box, keypoints, scores = swimmer_pose
                records[frame_number] = {
                    "frame": frame_number,
                    "timestamp": frame_number / fps,
                    "model": "rtmpose",
                    "box": box,
                    "lane": shift_lane(tracker.lane.as_record(), x_start, y_start),
                    "keypoints": {
                        name: {
                            "x": point[0],
                            "y": point[1],
                            "confidence": scores[index],
                        }
                        for index, (name, point) in enumerate(
                            zip(COCO_LANDMARK_NAMES, keypoints)
                        )
                    },
                }
            frame_number += 1
            if frame_number % 50 == 0:
                print(f"  detected frame {frame_number}", flush=True)
    finally:
        capture.release()
    return records, frame_number


def render_video(
    video_path: Path,
    output_path: Path,
    records: dict[int, dict],
    keypoint_confidence: float,
    style: str,
    smoothing: float,
    background: str,
    box_padding: float,
) -> bool:
    """Draw the swimmer on every frame and write the annotated video."""
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # H.264 plays in browsers (for the dashboard); mp4v is the fallback.
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            break
    if not writer.isOpened():
        capture.release()
        print(f"Error: could not create output video: {output_path}")
        return False

    smoother = KeypointSmoother(smoothing, keypoint_confidence)
    mask_box = None
    frame_number = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            record = records.get(frame_number)
            swimmer_pose = pose_from_record(record) if record else None
            box = record.get("display_box") if record else None

            # Keep the last box through missed frames so the view stays put.
            if box is not None:
                mask_box = padded_box(box, box_padding, width, height)
            if background != "keep" and mask_box is not None:
                mask_outside_box(frame, mask_box, background)

            if record and record.get("lane"):
                draw_lane(frame, record["lane"])
            if style == "simple":
                points = smoother.update(swimmer_pose)
                draw_simple_swimmer(frame, points, smoother.near_side())
            elif swimmer_pose is not None:
                draw_swimmer(
                    frame, swimmer_pose, keypoint_confidence, box_padding, box
                )

            frame_number += 1
            cv2.putText(
                frame,
                f"frame: {frame_number}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
    return True


def detect_pose(
    video_path: Path,
    output_path: Path,
    detection_confidence: float,
    keypoint_confidence: float,
    roi: tuple[float, float, float, float],
    landmarks_output: Path,
    style: str,
    smoothing: float,
    background: str,
    box_padding: float,
    render_only: bool = False,
    device: str = "cpu",
    rotate: bool = True,
    pick: tuple[float, float] | None = None,
    pick_time: float = 0.0,
) -> int:
    """Detect the swimmer, save landmarks, and save a pose-annotated video.

    With render_only, reuse the poses in landmarks_output and only redraw.
    """
    if not video_path.is_file():
        print(f"Error: video file not found: {video_path}")
        return 1

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Error: OpenCV could not open: {video_path}")
        return 1
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        print(f"Error: video has invalid properties: {video_path}")
        return 1

    if render_only:
        if not landmarks_output.is_file():
            print(f"Error: landmarks file not found: {landmarks_output}")
            return 1
        payload = json.loads(landmarks_output.read_text(encoding="utf-8"))
        records = {record["frame"]: record for record in payload["frames"]}
        total_frames = payload["total_frames"]
    else:
        records, total_frames = run_detection(
            video_path,
            detection_confidence,
            roi,
            device,
            rotate,
            pick,
            pick_time,
        )

    boxes = display_boxes(records, total_frames, fps, keypoint_confidence)
    for index, record in records.items():
        record["display_box"] = boxes[index]

    landmarks_output.parent.mkdir(parents=True, exist_ok=True)
    # Paths are relative to the landmarks file so the folder can be moved.
    landmarks_output.write_text(
        json.dumps(
            {
                "video": relative_path(video_path, landmarks_output.parent),
                "annotated_video": relative_path(
                    output_path, landmarks_output.parent
                ),
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
                "frames": [records[index] for index in sorted(records)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not render_video(
        video_path,
        output_path,
        records,
        keypoint_confidence,
        style,
        smoothing,
        background,
        box_padding,
    ):
        return 1

    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print(f"Frames processed: {total_frames}")
    print(f"Frames with a swimmer detected: {len(records)}")
    print(f"Landmarks: {landmarks_output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect one swimmer and draw pose landmarks on a video."
    )
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path for the annotated output video; default "
        "outputs/<video name>/pose.mp4",
    )
    parser.add_argument(
        "--pick",
        help="The swimmer's position as normalized x,y (e.g. 0.5,0.6); "
        "default picks the largest person",
    )
    parser.add_argument(
        "--pick-time",
        type=float,
        default=0.0,
        help="Seconds into the video where --pick applies; tracking starts there",
    )
    parser.add_argument(
        "--device",
        choices=("mps", "cpu"),
        default="mps",
        help="mps runs the pose model on the Apple GPU (CoreML); cpu is slower",
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Skip the sideways pose passes (about 3x faster, a bit less sure)",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.35,
        help="Minimum model detection confidence from 0 to 1",
    )
    parser.add_argument(
        "--keypoint-confidence",
        type=float,
        default=0.35,
        help="Minimum keypoint confidence for drawing from 0 to 1",
    )
    parser.add_argument(
        "--roi",
        default="0,0,1,1",
        help="Normalized crop as x1,y1,x2,y2; default uses the full frame",
    )
    parser.add_argument(
        "--landmarks-output",
        type=Path,
        help="Path for exported pose landmarks; default landmarks.json next "
        "to the output video",
    )
    parser.add_argument(
        "--style",
        choices=("simple", "full"),
        default="full",
        help="full draws every keypoint and the box; simple draws one "
        "smoothed stick figure",
    )
    parser.add_argument(
        "--background",
        choices=("black", "dim", "keep"),
        default="keep",
        help="How to show everything outside the swimmer box",
    )
    parser.add_argument(
        "--box-padding",
        type=float,
        default=0.1,
        help="Margin around the box, as a fraction of its size, where keypoints "
        "are still drawn (and kept visible with --background black/dim)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        help="Weight on the previous frame for simple-style smoothing; 0 disables",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip detection and redraw the video from the existing landmarks",
    )
    args = parser.parse_args()

    if not 0 <= args.detection_confidence <= 1:
        parser.error("--detection-confidence must be between 0 and 1")
    if not 0 <= args.keypoint_confidence <= 1:
        parser.error("--keypoint-confidence must be between 0 and 1")
    if args.box_padding < 0:
        parser.error("--box-padding must be at least 0")
    if not 0 <= args.smoothing < 1:
        parser.error("--smoothing must be at least 0 and less than 1")
    pick = None
    if args.pick:
        try:
            pick = tuple(float(value) for value in args.pick.split(","))
        except ValueError:
            parser.error("--pick must be two comma-separated numbers")
        if len(pick) != 2 or any(value < 0 or value > 1 for value in pick):
            parser.error("--pick must be x,y with values between 0 and 1")
    if args.pick_time < 0:
        parser.error("--pick-time must be at least 0")
    try:
        roi = tuple(float(value) for value in args.roi.split(","))
    except ValueError:
        parser.error("--roi must contain four comma-separated numbers")
    if len(roi) != 4 or any(value < 0 or value > 1 for value in roi):
        parser.error("--roi values must be between 0 and 1")
    if roi[0] >= roi[2] or roi[1] >= roi[3]:
        parser.error("--roi must have x1 < x2 and y1 < y2")

    # Each video gets its own folder holding everything the dashboard needs.
    output = args.output or Path("outputs") / args.video.stem / "pose.mp4"
    landmarks_output = args.landmarks_output or output.parent / "landmarks.json"

    return detect_pose(
        args.video,
        output,
        args.detection_confidence,
        args.keypoint_confidence,
        roi,
        landmarks_output,
        args.style,
        args.smoothing,
        args.background,
        args.box_padding,
        args.render_only,
        args.device,
        not args.no_rotate,
        pick,
        args.pick_time,
    )


if __name__ == "__main__":
    raise SystemExit(main())