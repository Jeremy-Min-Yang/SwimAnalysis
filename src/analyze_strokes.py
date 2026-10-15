"""Calculate freestyle metrics from exported swimmer landmarks.

Writes one metrics JSON per video with three linked views of the same swim,
all keyed to video frame numbers so a dashboard can sync them to playback:

- summary: whole-swim numbers
- strokes: one row per stroke with its frame range, recovery/underwater
  split, and head position
- frames: per-frame series (columnar) for plotting under the video
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1

# Plausible time between strokes (near-arm recoveries), in seconds.
MIN_STROKE_S = 0.6
MAX_STROKE_S = 3.0
# Autocorrelation below this means no clear stroke rhythm was found.
MIN_PERIODICITY = 0.2
# Peaks must be the highest point within this fraction of a stroke.
PEAK_SPACING = 0.4
# Minimum wrist rise, in torso lengths, to count as an arm recovery.
PEAK_PROMINENCE = 0.3
# The recovery phase is where the wrist is above this fraction of the way
# from the surrounding low point to the stroke's peak height.
RECOVERY_LEVEL = 0.3

HEAD_KEYPOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")


def angle(first: dict, vertex: dict, third: dict) -> float | None:
    """Return the 2D angle at vertex in degrees."""
    first_point = np.array([first["x"], first["y"]], dtype=float)
    vertex_point = np.array([vertex["x"], vertex["y"]], dtype=float)
    third_point = np.array([third["x"], third["y"]], dtype=float)
    first_vector = first_point - vertex_point
    third_vector = third_point - vertex_point
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(third_vector)
    if denominator == 0:
        return None
    cosine = np.dot(first_vector, third_vector) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def valid_point(frame: dict, name: str, minimum_confidence: float) -> dict | None:
    point = frame["keypoints"].get(name)
    if point is None or point["confidence"] < minimum_confidence:
        return None
    return point


def midpoint(frame: dict, first: str, second: str) -> np.ndarray:
    keypoints = frame["keypoints"]
    return np.array(
        [
            (keypoints[first]["x"] + keypoints[second]["x"]) / 2,
            (keypoints[first]["y"] + keypoints[second]["y"]) / 2,
        ]
    )


def reference_frames(landmarks: list[dict]) -> list[dict]:
    """Frames from RTMPose, whose hips are reliable, when there are any."""
    frames = [f for f in landmarks if f.get("model", "rtmpose") == "rtmpose"]
    return frames or landmarks


def torso_length(landmarks: list[dict]) -> float | None:
    """Median shoulder-to-hip distance in pixels."""
    lengths = [
        np.linalg.norm(
            midpoint(f, "left_shoulder", "right_shoulder")
            - midpoint(f, "left_hip", "right_hip")
        )
        for f in reference_frames(landmarks)
    ]
    length = float(np.median(lengths)) if lengths else 0.0
    return length if length > 1 else None


def swim_direction(landmarks: list[dict]) -> float:
    """+1 if the swimmer heads toward +x in the image, -1 otherwise."""
    offsets = [
        f["keypoints"]["nose"]["x"] - midpoint(f, "left_hip", "right_hip")[0]
        for f in reference_frames(landmarks)
    ]
    return -1.0 if offsets and np.median(offsets) < 0 else 1.0


def wrist_height_signal(
    landmarks: list[dict],
    total_frames: int,
    minimum_confidence: float,
    scale: float,
) -> np.ndarray:
    """Per frame, how far the highest wrist is above the shoulders in torsos.

    In side-view freestyle the near arm swings high above the water once per
    stroke, so this peaks once per stroke. Taking the higher of the
    two wrists keeps it working when the model mixes up left and right.
    """
    signal = np.full(total_frames, np.nan)
    for frame in landmarks:
        index = frame["frame"]
        if not 0 <= index < total_frames:
            continue
        shoulder_y = midpoint(frame, "left_shoulder", "right_shoulder")[1]
        heights = [
            (shoulder_y - wrist["y"]) / scale
            for name in ("left_wrist", "right_wrist")
            if (wrist := valid_point(frame, name, minimum_confidence))
        ]
        if heights:
            signal[index] = max(heights)
    return signal


def head_signals(
    landmarks: list[dict],
    total_frames: int,
    minimum_confidence: float,
    scale: float,
    direction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per frame head angle against the body line, and head height.

    Head angle is measured at the hips, between the hip-to-shoulder line and
    the hip-to-head line, in degrees; positive means the head is lifted above
    the body line, negative means tucked below it. Measuring from the hips
    rather than the shoulders keeps shoulder roll during the arm recovery
    from swamping it, though some roll still shows up as lift. Head height
    is how far the head sits above the shoulders, in torso lengths.
    """
    head_angle = np.full(total_frames, np.nan)
    head_height = np.full(total_frames, np.nan)
    body_names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    for frame in landmarks:
        index = frame["frame"]
        if not 0 <= index < total_frames:
            continue
        if not all(valid_point(frame, n, minimum_confidence) for n in body_names):
            continue
        head_points = [
            [point["x"], point["y"]]
            for name in HEAD_KEYPOINTS
            if (point := valid_point(frame, name, minimum_confidence))
        ]
        if not head_points:
            continue
        head = np.mean(head_points, axis=0)
        shoulders = midpoint(frame, "left_shoulder", "right_shoulder")
        hips = midpoint(frame, "left_hip", "right_hip")
        body = shoulders - hips
        to_head = head - hips
        cross = body[0] * to_head[1] - body[1] * to_head[0]
        # Image y points down, so "up" flips sign with the swim direction.
        head_angle[index] = -direction * math.degrees(
            math.atan2(cross, float(np.dot(body, to_head)))
        )
        head_height[index] = (shoulders[1] - head[1]) / scale
    return head_angle, head_height


def median_filter(signal: np.ndarray, window: int) -> np.ndarray:
    """Median-filter out single-frame glitches, leaving NaN where no data."""
    half = window // 2
    filtered = np.full(len(signal), np.nan)
    for index in range(len(signal)):
        values = signal[max(0, index - half) : index + half + 1]
        if np.isfinite(values).any():
            filtered[index] = np.nanmedian(values)
    return filtered


def fill_gaps(signal: np.ndarray) -> np.ndarray | None:
    """Fill NaN gaps by linear interpolation."""
    known = np.flatnonzero(np.isfinite(signal))
    if len(known) < 2:
        return None
    return np.interp(np.arange(len(signal)), known, signal[known])


def stroke_period(signal: np.ndarray, fps: float) -> float | None:
    """Estimate the time between strokes in seconds from autocorrelation."""
    lowest = int(MIN_STROKE_S * fps)
    highest = min(int(MAX_STROKE_S * fps), len(signal) - 1)
    if highest <= lowest:
        return None
    centered = signal - signal.mean()
    correlation = np.correlate(centered, centered, "full")[len(signal) - 1 :]
    if correlation[0] <= 0:
        return None
    correlation /= correlation[0]
    lag = lowest + int(np.argmax(correlation[lowest:highest]))
    if correlation[lag] < MIN_PERIODICITY:
        return None
    return lag / fps


def find_peaks(
    signal: np.ndarray, distance: int, base_distance: int, prominence: float
) -> list[int]:
    """Return local maxima that are the highest within +/- distance samples
    and rise at least prominence above the lowest point within
    +/- base_distance samples on both sides."""
    # Pad with the minimum so a stroke already underway at the start or end
    # of the clip can still register as a peak.
    padded = np.concatenate(([signal.min()], signal, [signal.min()]))
    peaks = []
    for index in range(1, len(padded) - 1):
        value = padded[index]
        if value < padded[index - 1] or value <= padded[index + 1]:
            continue
        start = max(0, index - distance)
        end = min(len(padded), index + distance + 1)
        if value < padded[start:end].max():
            continue
        left_low = padded[max(0, index - base_distance) : index].min()
        right_low = padded[index + 1 : index + base_distance + 1].min()
        if value - max(left_low, right_low) < prominence:
            continue
        peaks.append(index - 1)
    return peaks


def detect_strokes(wrist_height: np.ndarray, fps: float) -> list[int]:
    """Return the frame of each stroke's recovery peak."""
    period = stroke_period(wrist_height, fps)
    if period is None:
        return []
    return find_peaks(
        wrist_height,
        int(PEAK_SPACING * period * fps),
        int(period * fps),
        PEAK_PROMINENCE,
    )


def recovery_span(
    wrist_height: np.ndarray, peaks: list[int], position: int
) -> tuple[int, int]:
    """Return the first and last frame of a stroke's above-water recovery."""
    peak = peaks[position]
    left_bound = peaks[position - 1] if position > 0 else 0
    right_bound = (
        peaks[position + 1]
        if position + 1 < len(peaks)
        else len(wrist_height) - 1
    )
    base = max(
        wrist_height[left_bound : peak + 1].min(),
        wrist_height[peak : right_bound + 1].min(),
    )
    level = base + RECOVERY_LEVEL * (wrist_height[peak] - base)
    start = peak
    while start > left_bound and wrist_height[start - 1] >= level:
        start -= 1
    end = peak
    while end < right_bound and wrist_height[end + 1] >= level:
        end += 1
    return start, end


def nan_stat(values: np.ndarray, function) -> float | None:
    values = values[np.isfinite(values)]
    return float(function(values)) if len(values) else None


def build_strokes(
    peaks: list[int],
    wrist_height: np.ndarray,
    head_angle: np.ndarray,
    fps: float,
) -> list[dict]:
    """One record per stroke, spanning recovery start to next recovery start.

    The first stroke is incomplete if its recovery was already underway when
    the clip starts, and the last always is because its underwater phase is
    not followed by another recovery.
    """
    spans = [recovery_span(wrist_height, peaks, i) for i in range(len(peaks))]
    strokes = []
    for position, (peak, (start, recovery_end)) in enumerate(zip(peaks, spans)):
        is_last = position + 1 == len(peaks)
        end = len(wrist_height) - 1 if is_last else spans[position + 1][0] - 1
        complete = start > 0 and not is_last
        duration = (end - start + 1) / fps
        recovery = (recovery_end - start + 1) / fps
        stroke_head = head_angle[start : end + 1]
        # Underwater phase: the arm is away from the head, so shoulder roll
        # distorts the head angle least.
        underwater_head = head_angle[recovery_end + 1 : end + 1]
        strokes.append(
            {
                "index": position + 1,
                "complete": complete,
                "start_frame": start,
                "end_frame": end,
                "peak_frame": peak,
                "recovery_end_frame": recovery_end,
                "start_s": start / fps,
                "end_s": (end + 1) / fps,
                "peak_s": peak / fps,
                "duration_s": duration if complete else None,
                "recovery_s": recovery if start > 0 else None,
                "underwater_s": duration - recovery if complete else None,
                "recovery_ratio": recovery / duration if complete else None,
                "wrist_peak_height": float(wrist_height[peak]),
                "head_angle_mean_deg": nan_stat(stroke_head, np.mean),
                "head_angle_underwater_deg": nan_stat(underwater_head, np.mean),
                "head_angle_max_deg": nan_stat(stroke_head, np.max),
                "head_angle_min_deg": nan_stat(stroke_head, np.min),
            }
        )
    return strokes


def frame_series(
    total_frames: int,
    fps: float,
    landmarks: list[dict],
    wrist_height: np.ndarray | None,
    head_angle: np.ndarray,
    head_height: np.ndarray,
    strokes: list[dict],
) -> dict:
    """Per-frame columns aligned with the video's frame numbers."""
    model = [None] * total_frames
    for frame in landmarks:
        if 0 <= frame["frame"] < total_frames:
            model[frame["frame"]] = frame.get("model", "rtmpose")

    stroke_index = [None] * total_frames
    phase = [None] * total_frames
    for stroke in strokes:
        for index in range(stroke["start_frame"], stroke["end_frame"] + 1):
            stroke_index[index] = stroke["index"]
            in_recovery = index <= stroke["recovery_end_frame"]
            phase[index] = "recovery" if in_recovery else "underwater"

    def column(values: np.ndarray | None) -> list[float | None]:
        if values is None:
            return [None] * total_frames
        return [round(float(v), 3) if np.isfinite(v) else None for v in values]

    return {
        "frame": list(range(total_frames)),
        "time_s": [round(index / fps, 4) for index in range(total_frames)],
        "model": model,
        "stroke": stroke_index,
        "phase": phase,
        "wrist_height": column(wrist_height),
        "head_angle_deg": column(head_angle),
        "head_height": column(head_height),
    }


def summary_stats(values: list[float]) -> dict:
    if not values:
        return {"average": None, "minimum": None, "maximum": None}
    return {
        "average": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def joint_angles(landmarks: list[dict], minimum_confidence: float):
    """Left elbow and knee angles from every frame where all points are valid."""
    elbow_angles = []
    knee_angles = []
    for frame in landmarks:
        points = [
            valid_point(frame, name, minimum_confidence)
            for name in ("left_shoulder", "left_elbow", "left_wrist")
        ]
        if all(points) and (value := angle(*points)) is not None:
            elbow_angles.append(value)
        points = [
            valid_point(frame, name, minimum_confidence)
            for name in ("left_hip", "left_knee", "left_ankle")
        ]
        if all(points) and (value := angle(*points)) is not None:
            knee_angles.append(value)
    return elbow_angles, knee_angles


def mean_of(strokes: list[dict], key: str) -> float | None:
    values = [stroke[key] for stroke in strokes if stroke[key] is not None]
    return float(np.mean(values)) if values else None


def analyze(
    landmarks: list[dict],
    minimum_confidence: float,
    smoothing_window: int,
    total_frames: int | None = None,
    fps: float | None = None,
) -> dict:
    """Return summary, per-stroke, and per-frame metrics."""
    if not landmarks:
        return {"summary": {"error": "No pose landmarks found"}}

    if not fps:
        last = landmarks[-1]
        fps = last["frame"] / last["timestamp"] if last["timestamp"] > 0 else 0
    if not total_frames:
        total_frames = landmarks[-1]["frame"] + 1

    scale = torso_length(landmarks)
    wrist_height = head_angle = head_height = None
    strokes = []
    if scale is not None and fps > 0:
        wrist_height = fill_gaps(
            median_filter(
                wrist_height_signal(
                    landmarks, total_frames, minimum_confidence, scale
                ),
                smoothing_window,
            )
        )
        raw_angle, raw_height = head_signals(
            landmarks,
            total_frames,
            minimum_confidence,
            scale,
            swim_direction(landmarks),
        )
        head_angle = median_filter(raw_angle, smoothing_window)
        head_height = median_filter(raw_height, smoothing_window)
        if wrist_height is not None:
            peaks = detect_strokes(wrist_height, fps)
            strokes = build_strokes(peaks, wrist_height, head_angle, fps)
    if head_angle is None:
        head_angle = head_height = np.full(total_frames, np.nan)

    peak_frames = [stroke["peak_frame"] for stroke in strokes]
    period = (
        float(np.median(np.diff(peak_frames))) / fps
        if len(peak_frames) >= 2
        else None
    )
    complete = [stroke for stroke in strokes if stroke["complete"]]
    elbow_angles, knee_angles = joint_angles(landmarks, minimum_confidence)

    summary = {
        "frames_with_landmarks": len(landmarks),
        "valid_frame_ratio": len(landmarks) / total_frames,
        "torso_length_px": scale,
        "stroke_count": len(strokes) if period else None,
        "complete_strokes": len(complete) if period else None,
        "stroke_rate_spm": 60 / period if period else None,
        "stroke_period_s": period,
        "recovery_s_mean": mean_of(complete, "recovery_s"),
        "underwater_s_mean": mean_of(complete, "underwater_s"),
        "recovery_ratio_mean": mean_of(complete, "recovery_ratio"),
        "head_angle_deg": {
            **summary_stats(
                [float(v) for v in head_angle if np.isfinite(v)]
            ),
            "per_stroke_mean": mean_of(strokes, "head_angle_mean_deg"),
            "underwater_mean": mean_of(complete, "head_angle_underwater_deg"),
        },
        "elbow_angle_deg": summary_stats(elbow_angles),
        "knee_angle_deg": summary_stats(knee_angles),
    }
    return {
        "summary": summary,
        "strokes": strokes,
        "frames": frame_series(
            total_frames,
            fps,
            landmarks,
            wrist_height,
            head_angle,
            head_height,
            strokes,
        ),
    }


def relink(path: str | None, source_dir: Path, target_dir: Path) -> str | None:
    """Re-express a path stored relative to source_dir as relative to target_dir."""
    if not path:
        return None
    absolute = (source_dir / path).resolve()
    return Path(os.path.relpath(absolute, target_dir.resolve())).as_posix()


def print_report(metrics: dict) -> None:
    summary = metrics["summary"]

    def show(value, digits=2):
        return "-" if value is None else f"{value:.{digits}f}"

    print(f"Strokes: {summary.get('stroke_count')}"
          f" ({summary.get('complete_strokes')} complete)")
    print(f"Stroke rate: {show(summary.get('stroke_rate_spm'), 1)} per minute")
    print(
        f"Recovery / underwater: {show(summary.get('recovery_s_mean'))} s /"
        f" {show(summary.get('underwater_s_mean'))} s"
        f" (recovery ratio {show(summary.get('recovery_ratio_mean'))})"
    )
    head = summary.get("head_angle_deg", {})
    print(
        f"Head angle: average {show(head.get('average'), 1)} deg"
        f" (underwater {show(head.get('underwater_mean'), 1)} deg),"
        f" range {show(head.get('minimum'), 1)} to {show(head.get('maximum'), 1)}"
    )
    print()
    print(f"{'#':>3}{'peak s':>8}{'stroke s':>10}{'recov s':>9}"
          f"{'under s':>9}{'ratio':>7}{'head deg':>10}{'head uw':>9}")
    for stroke in metrics["strokes"]:
        print(
            f"{stroke['index']:>3}"
            f"{stroke['peak_s']:>8.2f}"
            f"{show(stroke['duration_s']):>10}"
            f"{show(stroke['recovery_s']):>9}"
            f"{show(stroke['underwater_s']):>9}"
            f"{show(stroke['recovery_ratio']):>7}"
            f"{show(stroke['head_angle_mean_deg'], 1):>10}"
            f"{show(stroke['head_angle_underwater_deg'], 1):>9}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze freestyle metrics from landmarks JSON."
    )
    parser.add_argument("landmarks", type=Path, help="Landmarks JSON path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Metrics JSON output path; default metrics.json next to the "
        "landmarks file",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--smoothing-window", type=int, default=7)
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.smoothing_window < 1:
        parser.error("--smoothing-window must be at least 1")

    if not args.landmarks.is_file():
        print(f"Error: landmarks file not found: {args.landmarks}")
        return 1
    payload = json.loads(args.landmarks.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        payload = {"frames": payload}
    landmarks = payload.get("frames", [])
    output = args.output or args.landmarks.parent / "metrics.json"

    metrics = analyze(
        landmarks,
        args.confidence,
        args.smoothing_window,
        payload.get("total_frames"),
        payload.get("fps"),
    )
    total_frames = len(metrics.get("frames", {}).get("frame", [])) or None
    fps = payload.get("fps")
    source_dir = args.landmarks.parent
    target_dir = output.parent
    document = {
        "schema_version": SCHEMA_VERSION,
        # Paths are relative to this metrics file.
        "video": relink(payload.get("video"), source_dir, target_dir),
        "annotated_video": relink(
            payload.get("annotated_video"), source_dir, target_dir
        ),
        "landmarks": relink(args.landmarks.name, source_dir, target_dir),
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": total_frames / fps if total_frames and fps else None,
        "width": payload.get("width"),
        "height": payload.get("height"),
        "settings": {
            "confidence": args.confidence,
            "smoothing_window": args.smoothing_window,
        },
        **metrics,
    }

    target_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=1), encoding="utf-8")
    print_report(metrics)
    print(f"\nMetrics: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
