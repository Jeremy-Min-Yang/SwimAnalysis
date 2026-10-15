"""Find lane ropes in pool footage and follow one swimmer's lane.

Lane ropes are long, straight runs of strongly colored floats (red or blue),
which are far easier to find than a swimmer in splash. A swimmer stays
between the same two ropes for a whole length, so knowing those two ropes
keeps the tracker off people on the deck and swimmers in other lanes.
"""

from __future__ import annotations

import cv2
import numpy as np


# HSV ranges for rope floats: red wraps around hue 0, blue sits near 110.
RED_RANGES = (((0, 90, 70), (10, 255, 255)), ((165, 90, 70), (180, 255, 255)))
BLUE_RANGES = (((100, 120, 40), (130, 255, 255)),)
# Pool water is cyan to teal.
WATER_RANGE = ((75, 40, 60), (105, 255, 255))
# Ropes run roughly across the frame; steeper lines are poles, ladders, edges.
MAX_ROPE_ANGLE_DEG = 40
# Segments within these limits are merged into one rope.
MERGE_ANGLE_DEG = 5
MERGE_OFFSET = 0.025  # fraction of frame height, measured at the swimmer
# A rope needs at least this much total segment length, as a fraction of width.
MIN_ROPE_LENGTH = 0.25


class Rope:
    """A rope as the line y = slope * x + intercept in frame pixels."""

    def __init__(self, slope: float, intercept: float, length: float):
        self.slope = slope
        self.intercept = intercept
        self.length = length

    def y_at(self, x: float) -> float:
        return self.slope * x + self.intercept

    def angle(self) -> float:
        return float(np.degrees(np.arctan(self.slope)))


def rope_mask(frame) -> np.ndarray:
    """Return a mask of rope-colored pixels."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], np.uint8)
    for low, high in RED_RANGES + BLUE_RANGES:
        mask |= cv2.inRange(hsv, low, high)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def water_fraction(frame, box) -> float:
    """Return the share of pixels inside box that look like pool water."""
    height, width = frame.shape[:2]
    x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
    x2, y2 = min(int(box[2]), width), min(int(box[3]), height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(cv2.inRange(hsv, *WATER_RANGE).mean() / 255)


def find_ropes(frame) -> list[Rope]:
    """Return the lane ropes in a frame, top to bottom at the frame center."""
    height, width = frame.shape[:2]
    segments = cv2.HoughLinesP(
        rope_mask(frame),
        1,
        np.pi / 360,
        threshold=60,
        minLineLength=width // 6,
        maxLineGap=width // 15,
    )
    if segments is None:
        return []

    center_x = width / 2
    groups: list[list] = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4).astype(float):
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(np.degrees(np.arctan(slope))) > MAX_ROPE_ANGLE_DEG:
            continue
        offset = y1 + slope * (center_x - x1)
        length = np.hypot(x2 - x1, y2 - y1)
        for group in groups:
            mean_slope, mean_offset = group[0], group[1]
            if (
                abs(np.degrees(np.arctan(slope) - np.arctan(mean_slope)))
                <= MERGE_ANGLE_DEG
                and abs(offset - mean_offset) <= MERGE_OFFSET * height
            ):
                group[2].append((x1, y1, x2, y2, length))
                total = sum(item[4] for item in group[2])
                group[0] = (mean_slope * (total - length) + slope * length) / total
                group[1] = (mean_offset * (total - length) + offset * length) / total
                break
        else:
            groups.append([slope, offset, [(x1, y1, x2, y2, length)]])

    ropes = []
    for _, _, items in groups:
        total = sum(item[4] for item in items)
        if total < MIN_ROPE_LENGTH * width:
            continue
        # Fit through all endpoints, weighting each segment by its length.
        xs = np.array([x for item in items for x in (item[0], item[2])])
        ys = np.array([y for item in items for y in (item[1], item[3])])
        weights = np.array([item[4] for item in items for _ in range(2)])
        slope, intercept = np.polyfit(xs, ys, 1, w=weights)
        ropes.append(Rope(float(slope), float(intercept), float(total)))
    ropes.sort(key=lambda rope: rope.y_at(center_x))
    return ropes


def lane_around(ropes: list[Rope], x: float, y: float):
    """Return the ropes just above and just below (x, y); either may be None."""
    above = [rope for rope in ropes if rope.y_at(x) < y]
    below = [rope for rope in ropes if rope.y_at(x) >= y]
    return (
        max(above, key=lambda rope: rope.y_at(x)) if above else None,
        min(below, key=lambda rope: rope.y_at(x)) if below else None,
    )


def closest_rope(ropes: list[Rope], previous: Rope, x: float, tolerance: float):
    """Return the rope nearest to previous at x, if within tolerance pixels."""
    best = None
    for rope in ropes:
        if abs(rope.angle() - previous.angle()) > 2 * MERGE_ANGLE_DEG:
            continue
        distance = abs(rope.y_at(x) - previous.y_at(x))
        if distance <= tolerance and (best is None or distance < best[0]):
            best = (distance, rope)
    return None if best is None else best[1]


class LaneTracker:
    """Follow the two ropes around one swimmer from frame to frame.

    The camera pans and tilts, so the ropes move a little each frame; each
    one is matched to the nearest rope of the same angle in the next frame.
    A rope that is not found (hidden by splash, out of frame) keeps its last
    position, and a missing side is estimated from the lane width last seen.
    """

    def __init__(self, lost_frames: int):
        self.lost_frames = lost_frames
        self.reset()

    def reset(self) -> None:
        self.upper: Rope | None = None
        self.lower: Rope | None = None
        self.width: float | None = None
        self.missed = 0

    def locked(self) -> bool:
        return self.upper is not None or self.lower is not None

    def start(self, frame, x: float, y: float) -> bool:
        """Lock onto the ropes on both sides of (x, y); False if not both found."""
        self.reset()
        upper, lower = lane_around(find_ropes(frame), x, y)
        if upper is None or lower is None:
            return False
        self.upper, self.lower = upper, lower
        self.remember_width(x)
        return True

    def update(self, frame, x: float) -> None:
        """Move the lane ropes to this frame; x is the swimmer's position.

        If neither rope is found for lost_frames frames (a scene cut, or the
        camera turned away), the lane is dropped so it can be found again.
        """
        ropes = find_ropes(frame)
        tolerance = 0.04 * frame.shape[0]
        if self.width:
            tolerance = max(tolerance, 0.35 * self.width)
        upper = lower = None
        if self.upper is not None:
            upper = closest_rope(ropes, self.upper, x, tolerance)
            self.upper = upper or self.upper
        if self.lower is not None:
            lower = closest_rope(ropes, self.lower, x, tolerance)
            self.lower = lower or self.lower
        if upper is None and lower is None:
            self.missed += 1
            if self.missed > self.lost_frames:
                self.reset()
                return
        else:
            self.missed = 0
        self.remember_width(x)

    def contains(self, x: float, y: float, margin: float) -> bool:
        """Whether (x, y) is in the lane, widened by margin lane widths per side."""
        band = self.band(x)
        if band is None:
            return True
        top, bottom = band
        extra = margin * (bottom - top)
        return top - extra <= y <= bottom + extra

    def center_y(self, x: float) -> float | None:
        band = self.band(x)
        return None if band is None else (band[0] + band[1]) / 2

    def as_record(self):
        """Return the ropes as {"upper": [slope, intercept], "lower": ...}."""
        return {
            name: None if rope is None else [rope.slope, rope.intercept]
            for name, rope in (("upper", self.upper), ("lower", self.lower))
        }

    def remember_width(self, x: float) -> None:
        if self.upper is not None and self.lower is not None:
            self.width = self.lower.y_at(x) - self.upper.y_at(x)

    def band(self, x: float):
        """Return (top, bottom) of the lane at x, or None if unknown."""
        top = self.upper.y_at(x) if self.upper is not None else None
        bottom = self.lower.y_at(x) if self.lower is not None else None
        if top is None and bottom is None:
            return None
        if top is None:
            top = bottom - (self.width or 0)
        if bottom is None:
            bottom = top + (self.width or 0)
        if bottom <= top:
            return None
        return top, bottom
