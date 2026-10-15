"""Display basic properties for a video file."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def inspect_video(video_path: Path) -> int:
    """Print video metadata and return a process exit code."""
    if not video_path.is_file():
        print(f"Error: video file not found: {video_path}")
        return 1

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Error: OpenCV could not open: {video_path}")
        return 1

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        codec_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(
            chr((codec_value >> (8 * index)) & 0xFF) for index in range(4)
        ).strip() or "unknown"

        duration = frame_count / fps if fps > 0 else None

        print(f"File: {video_path}")
        print(f"Resolution: {int(width)} x {int(height)} pixels")
        print(f"FPS: {fps:.2f}")
        print(f"Frames: {int(frame_count)}")
        if duration is None:
            print("Duration: unknown")
        else:
            print(f"Duration: {duration:.2f} seconds")
        print(f"Codec: {codec}")
    finally:
        capture.release()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Display basic properties for a swimming video."
    )
    parser.add_argument("video", type=Path, help="Path to the video file")
    args = parser.parse_args()
    return inspect_video(args.video)


if __name__ == "__main__":
    raise SystemExit(main())