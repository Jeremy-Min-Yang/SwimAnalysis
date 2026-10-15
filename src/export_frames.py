"""Export video frames with draft swimmer labels for correcting and training.

Samples frames from each analyzed video and writes them in the Ultralytics
YOLO pose format: one image plus one label file per frame, and a data.yaml
for training. The labels come from detect_pose.py's landmarks, so a labeling
tool such as CVAT opens them with the swimmer already outlined; fix what is
wrong, add swimmers that are missing, and export back to the same format.

Keypoints below --confidence are left out of the draft (marked not visible)
so they are added by hand rather than trusted. Frames where the tracker was
unsure get an empty label file, to be labeled from scratch.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

from detect_pose import COCO_LANDMARK_NAMES

# Mirror pairs for left/right flip augmentation, in COCO keypoint order.
FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
# The box around a draft swimmer: keypoint extent plus this margin per side.
BOX_MARGIN = 0.15
# A draft needs this many confident keypoints, or the frame is left unlabeled.
MIN_KEYPOINTS = 6


def draft_label(record: dict, width: int, height: int, confidence: float) -> str:
    """Return one YOLO pose label line for a landmarks record, or ''."""
    points = [record["keypoints"][name] for name in COCO_LANDMARK_NAMES]
    confident = [p for p in points if p["confidence"] >= confidence]
    if len(confident) < MIN_KEYPOINTS:
        return ""
    xs = [p["x"] for p in confident]
    ys = [p["y"] for p in confident]
    pad_x = (max(xs) - min(xs)) * BOX_MARGIN + 0.01 * width
    pad_y = (max(ys) - min(ys)) * BOX_MARGIN + 0.01 * height
    x1, x2 = max(min(xs) - pad_x, 0), min(max(xs) + pad_x, width)
    y1, y2 = max(min(ys) - pad_y, 0), min(max(ys) + pad_y, height)
    values = [
        0,
        (x1 + x2) / 2 / width,
        (y1 + y2) / 2 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]
    for point in points:
        if point["confidence"] >= confidence:
            x = min(max(point["x"] / width, 0), 1)
            y = min(max(point["y"] / height, 0), 1)
            values += [x, y, 2]
        else:
            values += [0, 0, 0]
    return " ".join(
        str(v) if isinstance(v, int) else f"{v:.6f}" for v in values
    )


def export_video(
    video_path: Path,
    landmarks_path: Path | None,
    output: Path,
    split: str,
    every_s: float,
    confidence: float,
) -> int:
    """Write sampled frames and draft labels for one video; return the count."""
    records = {}
    if landmarks_path is not None and landmarks_path.is_file():
        payload = json.loads(landmarks_path.read_text(encoding="utf-8"))
        records = {record["frame"]: record for record in payload["frames"]}

    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    step = max(int(round(every_s * fps)), 1)
    images = output / "images" / split
    labels = output / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    count = 0
    frame_number = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_number % step == 0:
                height, width = frame.shape[:2]
                name = f"{video_path.stem}_{frame_number:05d}"
                cv2.imwrite(str(images / f"{name}.jpg"), frame)
                record = records.get(frame_number)
                line = draft_label(record, width, height, confidence) if record else ""
                (labels / f"{name}.txt").write_text(
                    line + "\n" if line else "", encoding="utf-8"
                )
                count += 1
            frame_number += 1
    finally:
        capture.release()
    return count


def write_data_yaml(output: Path) -> None:
    (output / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output.resolve()}",
                "train: images/train",
                "val: images/val",
                "kpt_shape: [17, 3]",
                f"flip_idx: {FLIP_INDEX}",
                "names:",
                "  0: swimmer",
                "kpt_names:",
                "  0:",
                *[f"    - {name}" for name in COCO_LANDMARK_NAMES],
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export frames with draft swimmer pose labels for training."
    )
    parser.add_argument("videos", type=Path, nargs="+", help="Videos to export")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("datasets/swim_pose"),
        help="Dataset folder (default datasets/swim_pose)",
    )
    parser.add_argument(
        "--val",
        nargs="*",
        default=[],
        help="Video names (without extension) to hold out for validation; "
        "default puts about 1 in 5 videos there",
    )
    parser.add_argument(
        "--every",
        type=float,
        default=0.33,
        help="Seconds between exported frames (default 0.33)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Keypoints below this confidence are left out of the draft",
    )
    args = parser.parse_args()
    if args.every <= 0:
        parser.error("--every must be greater than 0")

    val = set(args.val)
    if not val and len(args.videos) >= 5:
        # Split by video so validation footage is never seen in training.
        stems = sorted(video.stem for video in args.videos)
        val = set(random.Random(0).sample(stems, max(len(stems) // 5, 1)))

    total = 0
    for video in args.videos:
        if not video.is_file():
            print(f"Skipping missing video: {video}")
            continue
        landmarks = Path("outputs") / video.stem / "landmarks.json"
        split = "val" if video.stem in val else "train"
        count = export_video(
            video,
            landmarks if landmarks.is_file() else None,
            args.output,
            split,
            args.every,
            args.confidence,
        )
        draft = "with drafts" if landmarks.is_file() else "no landmarks, unlabeled"
        print(f"{video.name}: {count} frames to {split} ({draft})")
        total += count
    write_data_yaml(args.output)
    print(f"Exported {total} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
