"""Run several pose models on the same video and compare them side by side."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from detect_pose import RtmposeRunner, YoloRunner, draw_swimmer


DEFAULT_MODELS = ("rtmpose-x", "yolo11x-pose.pt")

ARM_KEYPOINTS = (7, 8, 9, 10)
SWIM_KEYPOINTS = tuple(range(5, 17))
TILE_SIZE = (640, 360)


def build_runner(model_name: str, image_size: int, confidence: float):
    """Create the runner for a model name."""
    if model_name == "rtmpose-x":
        return RtmposeRunner(confidence)
    return YoloRunner(model_name, image_size, confidence)


def crop_window(poses, roi_box, previous):
    """Return a smoothed 16:9 crop around every model's swimmer box."""
    boxes = [pose[0] for pose in poses if pose is not None]
    if boxes:
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
    else:
        x1, y1, x2, y2 = roi_box

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = max(x2 - x1, (y2 - y1) * 16 / 9) * 1.5
    if previous is not None:
        center_x = 0.8 * previous[0] + 0.2 * center_x
        center_y = 0.8 * previous[1] + 0.2 * center_y
        width = 0.8 * previous[2] + 0.2 * width
    return center_x, center_y, width


def crop_tile(frame, window):
    """Cut the crop window out of a frame and resize it to one tile."""
    height, width = frame.shape[:2]
    center_x, center_y, crop_width = window
    crop_width = min(crop_width, width, height * 16 / 9)
    crop_height = crop_width * 9 / 16
    x1 = int(min(max(center_x - crop_width / 2, 0), width - crop_width))
    y1 = int(min(max(center_y - crop_height / 2, 0), height - crop_height))
    tile = frame[y1 : y1 + int(crop_height), x1 : x1 + int(crop_width)]
    return cv2.resize(tile, TILE_SIZE)


def label_tile(tile, text: str) -> None:
    """Draw a model name banner on a tile."""
    cv2.rectangle(tile, (0, 0), (TILE_SIZE[0], 32), (0, 0, 0), -1)
    cv2.putText(
        tile,
        text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def make_grid(tiles):
    """Arrange tiles in rows of up to three."""
    columns = min(3, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    blank = np.zeros_like(tiles[0])
    padded = tiles + [blank] * (rows * columns - len(tiles))
    return np.vstack(
        [
            np.hstack(padded[row * columns : (row + 1) * columns])
            for row in range(rows)
        ]
    )


def summarize(stats, threshold: float):
    """Turn per-frame records into per-model summary numbers."""
    summary = {}
    for name, records in stats.items():
        poses = [record["pose"] for record in records]
        detected = [pose for pose in poses if pose is not None]
        swim_scores = [
            pose[2][index] for pose in detected for index in SWIM_KEYPOINTS
        ]
        arm_visible = [
            pose[2][index] >= threshold
            for pose in detected
            for index in ARM_KEYPOINTS
        ]

        jitter = []
        for previous, current in zip(poses, poses[1:]):
            if previous is None or current is None:
                continue
            box = current[0]
            diagonal = math.hypot(box[2] - box[0], box[3] - box[1]) or 1.0
            for index in (9, 10):
                if (
                    previous[2][index] >= threshold
                    and current[2][index] >= threshold
                ):
                    step = math.dist(previous[1][index], current[1][index])
                    jitter.append(step / diagonal)

        summary[name] = {
            "detection_rate": len(detected) / len(poses) if poses else 0.0,
            "mean_keypoint_confidence": float(np.mean(swim_scores))
            if swim_scores
            else 0.0,
            "arm_keypoints_visible": float(np.mean(arm_visible))
            if arm_visible
            else 0.0,
            "wrist_jitter": float(np.median(jitter)) if jitter else None,
            "ms_per_frame": 1000
            * float(np.mean([record["seconds"] for record in records])),
        }
    return summary


def print_summary(summary) -> None:
    """Print the comparison table."""
    print()
    print(
        f"{'model':<18}{'detected':>10}{'mean conf':>11}"
        f"{'arms seen':>11}{'wrist jit':>11}{'ms/frame':>10}"
    )
    for name, values in summary.items():
        jitter = values["wrist_jitter"]
        jitter_text = "-" if jitter is None else f"{jitter:.3f}"
        print(
            f"{name:<18}"
            f"{values['detection_rate']:>10.0%}"
            f"{values['mean_keypoint_confidence']:>11.2f}"
            f"{values['arm_keypoints_visible']:>11.0%}"
            f"{jitter_text:>11}"
            f"{values['ms_per_frame']:>10.0f}"
        )
    print()
    print("detected: frames with a swimmer found")
    print("mean conf: average shoulder-to-ankle keypoint confidence")
    print("arms seen: elbow/wrist points above the keypoint threshold")
    print("wrist jit: median wrist movement per frame / box diagonal")
    print("RTMPose confidences use a different scale than YOLO's.")


def compare_models(
    video_path: Path,
    output_dir: Path,
    model_names: list[str],
    detection_confidence: float,
    keypoint_confidence: float,
    image_size: int,
    roi: tuple[float, float, float, float],
    stride: int,
    still_count: int,
) -> int:
    """Run every model over the video and save the comparison outputs."""
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
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    roi_box = (
        int(roi[0] * width),
        int(roi[1] * height),
        int(roi[2] * width),
        int(roi[3] * height),
    )

    print("Loading models...")
    runners = {
        name: build_runner(name, image_size, detection_confidence)
        for name in model_names
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    columns = min(3, len(model_names))
    rows = math.ceil(len(model_names) / columns)
    writer = cv2.VideoWriter(
        str(output_dir / "comparison.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(fps / stride, 1),
        (TILE_SIZE[0] * columns, TILE_SIZE[1] * rows),
    )
    if not writer.isOpened():
        capture.release()
        print(f"Error: could not create output video in {output_dir}")
        return 1

    still_frames = {
        int(total_frames * (index + 1) / (still_count + 1))
        // stride
        * stride
        for index in range(still_count)
    }
    stats = {name: [] for name in model_names}
    window = None
    frame_number = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_number % stride:
                frame_number += 1
                continue

            x1, y1, x2, y2 = roi_box
            inference_frame = frame[y1:y2, x1:x2]
            poses = []
            for name, runner in runners.items():
                started = time.perf_counter()
                pose = runner(inference_frame, x1, y1)
                stats[name].append(
                    {"pose": pose, "seconds": time.perf_counter() - started}
                )
                poses.append(pose)

            window = crop_window(poses, roi_box, window)
            tiles = []
            for name, pose in zip(model_names, poses):
                annotated = frame.copy()
                if pose is not None:
                    draw_swimmer(annotated, pose, keypoint_confidence)
                tile = crop_tile(annotated, window)
                label_tile(tile, f"{name}  frame {frame_number}")
                tiles.append(tile)

            grid = make_grid(tiles)
            writer.write(grid)
            if frame_number in still_frames:
                cv2.imwrite(
                    str(output_dir / f"frame_{frame_number:05d}.jpg"), grid
                )

            frame_number += 1
            if frame_number % (stride * 25) < stride:
                print(f"  processed frame {frame_number}/{total_frames}")
    finally:
        capture.release()
        writer.release()

    summary = summarize(stats, keypoint_confidence)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print_summary(summary)
    print(f"\nOutputs: {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare pose models side by side on one video."
    )
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs/model_comparison"),
        help="Folder for the comparison video, stills, and summary",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated YOLO pose weights and/or rtmpose-x",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.15,
        help="Minimum person detection confidence from 0 to 1",
    )
    parser.add_argument(
        "--keypoint-confidence",
        type=float,
        default=0.25,
        help="Minimum keypoint confidence for drawing and stats",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO inference image size in pixels",
    )
    parser.add_argument(
        "--roi",
        default="0,0.15,1,0.85",
        help="Normalized crop as x1,y1,x2,y2",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame",
    )
    parser.add_argument(
        "--stills",
        type=int,
        default=6,
        help="Number of evenly spaced grid images to save",
    )
    args = parser.parse_args()

    model_names = [name.strip() for name in args.models.split(",") if name]
    if not model_names:
        parser.error("--models must name at least one model")
    if not 0 <= args.detection_confidence <= 1:
        parser.error("--detection-confidence must be between 0 and 1")
    if not 0 <= args.keypoint_confidence <= 1:
        parser.error("--keypoint-confidence must be between 0 and 1")
    if args.imgsz <= 0 or args.stride <= 0 or args.stills < 0:
        parser.error("--imgsz and --stride must be positive, --stills >= 0")
    try:
        roi = tuple(float(value) for value in args.roi.split(","))
    except ValueError:
        parser.error("--roi must contain four comma-separated numbers")
    if len(roi) != 4 or any(value < 0 or value > 1 for value in roi):
        parser.error("--roi values must be between 0 and 1")
    if roi[0] >= roi[2] or roi[1] >= roi[3]:
        parser.error("--roi must have x1 < x2 and y1 < y2")

    return compare_models(
        args.video,
        args.output_dir,
        model_names,
        args.detection_confidence,
        args.keypoint_confidence,
        args.imgsz,
        roi,
        args.stride,
        args.stills,
    )


if __name__ == "__main__":
    raise SystemExit(main())
