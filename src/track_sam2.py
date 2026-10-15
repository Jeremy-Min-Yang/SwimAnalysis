"""Follow one swimmer through a video with SAM 2 segmentation.

SAM 2 is prompted with the swimmer's box on the first frame and then follows
the swimmer's mask from frame to frame using a memory of earlier masks, so it
does not depend on a person detector seeing a swimmer in splash.

Plain SAM 2 loses the swimmer for good at the dive entry: the camera whips
around, the mask jumps to something else and never comes back. So the tracker
runs frame by frame and checks every mask (like SAMURAI, which adds motion
cues to SAM 2):

- A mask is rejected when SAM 2 says the swimmer is absent, when it is a
  thin line (a lane rope, MAX_ASPECT), or when it grows or shrinks by more
  than MAX_AREA_CHANGE against the last few seconds or lands away from the
  last box (MAX_JUMP box sizes).
- Accepted masks go into SAM 2's memory, which keeps the prompt plus the last
  MEMORY_FRAMES - 1 accepted masks. Lost for LOST_RESET_FRAMES frames, the
  memory goes back to the prompt alone, so SAM 2 looks for the swimmer as
  first prompted instead of whatever the last masks drifted onto.
- A frame where detect_pose.py's keypoints are confident (from
  outputs/<video>/landmarks.json) restarts the track from those keypoints
  when the swimmer is lost, or when the confident pose has sat outside the
  mask for DISAGREE_FRAMES frames.
- --seed frame:x1,y1,x2,y2 restarts it by hand, e.g. after a cut.

By default the first-frame box comes from frame 0's keypoints in
landmarks.json; pass --box to set it by hand.

Writes outputs/<video>/sam2_track.json (per frame: mask box, mask area,
mask centroid, presence score and source, or null when the swimmer is lost)
and sam2_track.mp4 (the mask and its box drawn on the video).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from detect_pose import COCO_LANDMARK_NAMES


COLORS = {
    "track": (255, 160, 0),
    "seed-pose": (0, 220, 0),
    "seed-manual": (0, 220, 220),
    "seed-first": (0, 220, 220),
}
SEED_KEYPOINT_CONFIDENCE = 0.4
SEED_MARGIN = 0.1
# A pose restarts the track when its 12 shoulder-to-ankle keypoints average
# this confidence and at least SEED_MIN_KEYPOINTS reach SEED_KEYPOINT_CONFIDENCE.
SEED_MEAN_CONFIDENCE = 0.55
SEED_MIN_KEYPOINTS = 9
# Memory: the prompt plus the latest accepted masks, MEMORY_FRAMES in all
# (SAM 2.1 is trained with 7), adding a mask every MEMORY_EVERY frames.
MEMORY_FRAMES = 7
MEMORY_EVERY = 2
# Mask checks against the last accepted mask, if it is at most RECENT_FRAMES old.
MAX_AREA_CHANGE = 3.0
MAX_JUMP = 1.0
RECENT_FRAMES = 15
MIN_AREA_FRACTION = 0.0003
# Mask area is also compared with the median accepted area over this many
# seconds, so a slow drift onto something smaller or larger is caught.
AREA_HISTORY_S = 2.0
# A swimmer's box is at most this many times wider than tall (or taller than
# wide); a lane rope is far thinner.
MAX_ASPECT = 12.0
LOST_RESET_FRAMES = 5
# Restart from a confident pose that has been outside the mask this long.
DISAGREE_FRAMES = 3
DISAGREE_INSIDE = 0.5


def keypoint_box(record: dict, confidence: float):
    """Box around a landmarks record's confident keypoints, padded by SEED_MARGIN."""
    points = np.array(
        [
            (record["keypoints"][name]["x"], record["keypoints"][name]["y"])
            for name in COCO_LANDMARK_NAMES
            if record["keypoints"][name]["confidence"] >= confidence
        ]
    )
    if len(points) < 4:
        return None
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    pad_x, pad_y = (x2 - x1) * SEED_MARGIN, (y2 - y1) * SEED_MARGIN
    return [float(x1 - pad_x), float(y1 - pad_y), float(x2 + pad_x), float(y2 + pad_y)]


def confident_pose(record: dict | None):
    """The record's body keypoints (12, 2) if confident enough to seed, else None."""
    if record is None:
        return None
    names = COCO_LANDMARK_NAMES[5:]
    scores = np.array([record["keypoints"][name]["confidence"] for name in names])
    if (
        scores.mean() < SEED_MEAN_CONFIDENCE
        or (scores >= SEED_KEYPOINT_CONFIDENCE).sum() < SEED_MIN_KEYPOINTS
    ):
        return None
    points = np.array([(record["keypoints"][n]["x"], record["keypoints"][n]["y"]) for n in names])
    return points[scores >= SEED_KEYPOINT_CONFIDENCE]


def mask_record(mask: np.ndarray):
    """Box, area, and centroid of a boolean mask, or None when it is empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return {
        "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "area": int(len(xs)),
        "centroid": [float(xs.mean()), float(ys.mean())],
    }


def plausible(record: dict, last: dict | None, typical_area: float | None) -> bool:
    """Whether a mask could be the same swimmer as the last accepted one,
    whose recent masks had typical_area."""
    x1, y1, x2, y2 = record["box"]
    width, height = x2 - x1 + 1, y2 - y1 + 1
    if max(width / height, height / width) > MAX_ASPECT:
        return False
    for area in (last and last["area"], typical_area):
        if area and not 1 / MAX_AREA_CHANGE <= record["area"] / area <= MAX_AREA_CHANGE:
            return False
    if last is None:
        return True
    x1, y1, x2, y2 = last["box"]
    reach_x, reach_y = (x2 - x1) * MAX_JUMP, (y2 - y1) * MAX_JUMP
    bx1, by1, bx2, by2 = record["box"]
    return bx2 >= x1 - reach_x and bx1 <= x2 + reach_x and by2 >= y1 - reach_y and by1 <= y2 + reach_y


def inside_fraction(points: np.ndarray, box) -> float:
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * 0.25, (y2 - y1) * 0.25
    inside = (
        (points[:, 0] >= x1 - pad_x)
        & (points[:, 0] <= x2 + pad_x)
        & (points[:, 1] >= y1 - pad_y)
        & (points[:, 1] <= y2 + pad_y)
    )
    return float(inside.mean())


class Sam2Follower:
    """SAM 2 with a memory the caller controls: seed, predict, remember."""

    def __init__(self, model: str, image_size: int, device: str):
        from ultralytics.models.sam import SAM2DynamicInteractivePredictor

        self.predictor = SAM2DynamicInteractivePredictor(
            overrides=dict(
                model=model,
                task="segment",
                mode="predict",
                imgsz=image_size,
                conf=0.0,
                device=device,
                save=False,
                verbose=False,
            ),
            max_obj_num=1,
        )

    @staticmethod
    def _unpack(result):
        if result.masks is None or len(result.masks.data) == 0:
            return None, 0.0
        score = float(result.boxes.conf[0]) if result.boxes is not None else 1.0
        return result.masks.data[0].cpu().numpy() > 0.5, score

    def seed(self, frame, box):
        """Forget everything and start again from box on this frame."""
        self.predictor.memory_bank.clear()
        result = self.predictor(source=frame, bboxes=[box], obj_ids=[0], update_memory=True)[0]
        return self._unpack(result)

    def forget(self) -> None:
        """Keep only the prompt in memory."""
        del self.predictor.memory_bank[1:]

    def predict(self, frame):
        return self._unpack(self.predictor(source=frame)[0])

    @torch.inference_mode()
    def remember(self, mask: np.ndarray) -> None:
        """Add the last predicted frame, with this mask, to the memory."""
        predictor = self.predictor
        # (1, H, W, 1): Ultralytics letterboxes each mask as an image, which
        # needs a channel axis when the frame must be padded.
        _, _, masks = predictor._prepare_prompts(
            dst_shape=predictor.imgsz,
            src_shape=predictor.src_shape,
            masks=mask[None, :, :, None].astype(np.uint8),
        )
        points = torch.zeros((1, 0, 2), dtype=predictor.torch_dtype, device=predictor.device)
        labels = torch.zeros((1, 0), dtype=torch.int32, device=predictor.device)
        predictor.update_memory([0], points, labels, masks)
        bank = predictor.memory_bank
        if len(bank) > MEMORY_FRAMES:
            del bank[1 : len(bank) - MEMORY_FRAMES + 1]


def track(
    video_path: Path,
    first_box,
    seeds: dict[int, list[float]],
    poses: dict[int, dict],
    model: str,
    output_dir: Path,
    image_size: int,
    device: str,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Error: OpenCV could not open: {video_path}")
        return 1
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_dir / "sam2_track.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    follower = Sam2Follower(model, image_size, device)
    min_area = MIN_AREA_FRACTION * width * height

    frames = []
    last = None
    last_frame = -RECENT_FRAMES - 1
    disagree = 0
    accepted_since_memory = 0
    areas: list[tuple[int, int]] = []
    history = int(AREA_HISTORY_S * fps)
    started = time.perf_counter()
    index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        pose_points = confident_pose(poses.get(index))
        lost = index - last_frame > 1

        source = None
        if index == 0:
            source, box = "seed-first", first_box
        elif index in seeds:
            source, box = "seed-manual", seeds[index]
        elif pose_points is not None and (lost or disagree >= DISAGREE_FRAMES):
            source, box = "seed-pose", keypoint_box(poses[index], SEED_KEYPOINT_CONFIDENCE)

        if source is not None:
            mask, score = follower.seed(frame, box)
            record = mask_record(mask) if mask is not None else None
            disagree = 0
            accepted_since_memory = 0
            areas = []
        else:
            source = "track"
            if index - last_frame == LOST_RESET_FRAMES + 1:
                follower.forget()
            mask, score = follower.predict(frame)
            record = mask_record(mask) if mask is not None and score > 0 else None
            recent = last if index - last_frame <= RECENT_FRAMES else None
            areas = [(frame_index, area) for frame_index, area in areas if index - frame_index <= history]
            typical = float(np.median([area for _, area in areas])) if areas else None
            if record is not None and (
                record["area"] < min_area or not plausible(record, recent, typical)
            ):
                record = None
            if record is not None:
                accepted_since_memory += 1
                if accepted_since_memory >= MEMORY_EVERY:
                    follower.remember(mask)
                    accepted_since_memory = 0

        if record is not None and pose_points is not None:
            if inside_fraction(pose_points, record["box"]) < DISAGREE_INSIDE:
                disagree += 1
            else:
                disagree = 0

        if record is not None:
            record["score"] = score
            record["source"] = source
            areas.append((index, record["area"]))
            last, last_frame = record, index
            color = np.array(COLORS[source], dtype=np.uint8)
            frame[mask] = (0.5 * frame[mask] + 0.5 * color).astype(np.uint8)
            x1, y1, x2, y2 = record["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[source], 3)
            label = f"{source} {score:.2f}"
        else:
            label = "LOST"
        cv2.putText(frame, f"frame {index}  {label}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8)
        cv2.putText(frame, f"frame {index}  {label}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        writer.write(frame)
        frames.append({"frame": index, "timestamp": index / fps, "mask": record})

        index += 1
        if index % 100 == 0:
            found = sum(f["mask"] is not None for f in frames)
            rate = index / (time.perf_counter() - started)
            print(f"  {index}/{total} frames, found {found}, {rate:.1f} frames/s", flush=True)
    capture.release()
    writer.release()

    document = {
        "video": str(video_path),
        "model": model,
        "image_size": image_size,
        "first_box": first_box,
        "seeds": {str(k): v for k, v in seeds.items()},
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frames,
    }
    (output_dir / "sam2_track.json").write_text(json.dumps(document), encoding="utf-8")
    found = sum(f["mask"] is not None for f in frames)
    restarts = sum(1 for f in frames if f["mask"] and f["mask"]["source"] == "seed-pose")
    print(f"Swimmer found in {found}/{len(frames)} frames ({restarts} restarts from pose).")
    print(f"Outputs in {output_dir}")
    return 0


def parse_box(text: str) -> list[float]:
    values = [float(value) for value in text.split(",")]
    if len(values) != 4:
        raise ValueError(f"expected x1,y1,x2,y2, got {text}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--box", help="First-frame swimmer box in pixels: x1,y1,x2,y2")
    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        help="Restart from a box on a frame: frame:x1,y1,x2,y2 (repeatable)",
    )
    parser.add_argument("--no-pose-seeds", action="store_true", help="Never restart from keypoints")
    parser.add_argument("--model", default="sam2.1_t.pt", help="SAM 2 weights")
    parser.add_argument("--imgsz", type=int, default=640, help="SAM 2 input size (1024 is ~5x slower)")
    parser.add_argument("--device", default="mps", help="mps, cpu, or a CUDA index")
    parser.add_argument("-o", "--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or Path("outputs") / args.video.stem
    landmarks = output_dir / "landmarks.json"
    records = []
    if landmarks.is_file():
        records = json.loads(landmarks.read_text(encoding="utf-8"))["frames"]

    try:
        seeds = {}
        for item in args.seed:
            frame, box = item.split(":")
            seeds[int(frame)] = parse_box(box)
        first_box = parse_box(args.box) if args.box else None
    except ValueError as error:
        parser.error(str(error))
    if first_box is None:
        if not records:
            parser.error(f"no --box given and {landmarks} not found")
        first_box = keypoint_box(records[0], 0.3)
        if first_box is None:
            parser.error("frame 0 has too few confident keypoints; pass --box")
    poses = {} if args.no_pose_seeds else {record["frame"]: record for record in records}

    print(f"First box: {[round(value) for value in first_box]}")
    return track(
        args.video, first_box, seeds, poses, args.model, output_dir, args.imgsz, args.device
    )


if __name__ == "__main__":
    raise SystemExit(main())
