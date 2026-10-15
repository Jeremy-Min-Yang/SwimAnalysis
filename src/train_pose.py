"""Fine-tune a YOLO pose model on corrected swimmer frames.

Starts from a COCO-trained YOLO11 pose model, which already knows the 17 body
keypoints, and teaches it what swimmers look like: lying flat, half
underwater, in splash. Run export_frames.py first and correct its labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO pose model on the swimmer dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/swim_pose/data.yaml"),
        help="Dataset yaml written by export_frames.py",
    )
    parser.add_argument(
        "--model",
        default="yolo11s-pose.pt",
        help="Starting weights; n is fastest, m is more accurate",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Training image size; swimmers are small, so above the usual 640",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps", help="mps, cpu, or a CUDA id")
    parser.add_argument(
        "--name", default="swim_pose", help="Run name under runs/pose/"
    )
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"dataset not found: {args.data} (run export_frames.py)")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=25,
        # The camera is never upside down, but swimmers go both ways and
        # the camera tilts, so mirror left/right and allow some rotation.
        fliplr=0.5,
        flipud=0.0,
        degrees=10.0,
        # Phone footage varies a lot in light and water color.
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
        mosaic=1.0,
    )
    metrics = model.val(device=args.device)
    print(f"Pose mAP50-95: {metrics.pose.map:.3f}  box mAP50-95: {metrics.box.map:.3f}")
    print(f"Best weights: {Path(model.trainer.best)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
