"""Follow one swimmer through a video with SAM 2 segmentation.

You mark the swimmer once (--seed), on a frame where they are already
swimming, and SAM 2 follows their outline from frame to frame using a memory
of earlier masks. Mark them again wherever the overlay drifts off them.

Each mask is checked before it is used (like SAMURAI, which adds motion cues
to SAM 2), since SAM 2 alone drifts onto splash, lane ropes, or other people:

- A mask is rejected when SAM 2 says the swimmer is absent, when it is a
  thin line (a lane rope, MAX_ASPECT), or when it grows or shrinks by more
  than MAX_AREA_CHANGE against the last few seconds or lands away from the
  last box (MAX_JUMP box sizes).
- Accepted masks go into SAM 2's memory, which keeps the seed plus the last
  MEMORY_FRAMES - 1 accepted masks. Lost for LOST_RESET_FRAMES frames, the
  memory goes back to the seed alone, so SAM 2 looks for the swimmer as
  marked instead of whatever the last masks drifted onto.

Writes outputs/<video>/sam2_track.json (per frame: mask box, mask area,
mask centroid, presence score and source, or null when the swimmer is lost
or not yet marked) and sam2_track.mp4 (the mask and its box drawn on the
video).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch


COLORS = {"track": (255, 160, 0), "seed": (0, 220, 220)}
# Memory: the seed plus the latest accepted masks, MEMORY_FRAMES in all
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


def pick_device(requested: str) -> str:
    """"auto": an NVIDIA or AMD (ROCm) GPU, else an Apple GPU, else the CPU."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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

    def seed(self, frame, prompt: list[float]):
        """Forget everything and start again from a point (x, y) or a box
        (x1, y1, x2, y2) on the swimmer in this frame."""
        self.predictor.memory_bank.clear()
        if len(prompt) == 2:
            prompts = dict(points=[prompt], labels=[1])
        else:
            prompts = dict(bboxes=[prompt])
        result = self.predictor(source=frame, obj_ids=[0], update_memory=True, **prompts)[0]
        return self._unpack(result)

    def forget(self) -> None:
        """Keep only the seed in memory."""
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
    seeds: dict[int, list[float]],
    model: str,
    output_dir: Path,
    image_size: int,
    device: str,
    reseed=None,
) -> int:
    """reseed, if given, is called on each frame after the first seed as
    reseed(index, last_record, last_frame) and returns a prompt to seed the
    swimmer afresh on that frame, or None."""
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
    first_seed = min(seeds)

    frames = []
    last = None
    last_frame = -RECENT_FRAMES - 1
    accepted_since_memory = 0
    areas: list[tuple[int, int]] = []
    history = int(AREA_HISTORY_S * fps)
    started = time.perf_counter()
    index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break

        record = None
        prompt = seeds.get(index)
        if prompt is None and reseed is not None and index > first_seed:
            prompt = reseed(index, last, last_frame)
        if prompt is not None:
            source = "seed"
            mask, score = follower.seed(frame, prompt)
            record = mask_record(mask) if mask is not None else None
            accepted_since_memory = 0
            areas = []
        elif index > first_seed:
            source = "track"
            if index - last_frame == LOST_RESET_FRAMES + 1:
                follower.forget()
            mask, score = follower.predict(frame)
            record = mask_record(mask) if mask is not None and score > 0 else None
            recent = last if index - last_frame <= RECENT_FRAMES else None
            areas = [(i, area) for i, area in areas if index - i <= history]
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
            label = "LOST" if index > first_seed else "not marked yet"
        for color, thickness in (((0, 0, 0), 8), ((255, 255, 255), 3)):
            cv2.putText(frame, f"frame {index}  {label}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, thickness)
        writer.write(frame)
        frames.append({"frame": index, "timestamp": index / fps, "mask": record})

        index += 1
        if index % 50 == 0:
            found = sum(f["mask"] is not None for f in frames)
            rate = index / (time.perf_counter() - started)
            print(f"  {index}/{total} frames, found {found}, {rate:.1f} frames/s", flush=True)
    capture.release()
    writer.release()

    document = {
        "video": str(video_path),
        "model": model,
        "image_size": image_size,
        "seeds": {str(k): v for k, v in seeds.items()},
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frames,
    }
    (output_dir / "sam2_track.json").write_text(json.dumps(document), encoding="utf-8")
    tracked = frames[first_seed:]
    found = sum(f["mask"] is not None for f in tracked)
    print(f"Swimmer found in {found}/{len(tracked)} frames from the first seed on.")
    print(f"Outputs in {output_dir}")
    return 0


def parse_seed(text: str) -> tuple[int, list[float]]:
    """frame:x,y (a point on the swimmer) or frame:x1,y1,x2,y2 (a box around them)."""
    frame, _, coordinates = text.partition(":")
    values = [float(value) for value in coordinates.split(",")]
    if len(values) not in (2, 4):
        raise ValueError(f"expected frame:x,y or frame:x1,y1,x2,y2, got {text}")
    return int(frame), values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--seed",
        action="append",
        required=True,
        help="Mark the swimmer on a frame, as frame:x,y (a point on them) or "
        "frame:x1,y1,x2,y2 (a box around them), in pixels. Repeat to correct "
        "the track; a box is more reliable in splash.",
    )
    parser.add_argument("--model", default="sam2.1_t.pt", help="SAM 2 weights")
    parser.add_argument("--imgsz", type=int, default=640, help="SAM 2 input size (1024 is ~5x slower)")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (default), cpu, mps (Apple GPU), or a GPU index such as 0",
    )
    parser.add_argument("-o", "--output-dir", type=Path)
    args = parser.parse_args()

    try:
        seeds = dict(parse_seed(item) for item in args.seed)
    except ValueError as error:
        parser.error(str(error))
    output_dir = args.output_dir or Path("outputs") / args.video.stem
    device = pick_device(args.device)
    print(f"Device: {device}")
    return track(args.video, seeds, args.model, output_dir, args.imgsz, device)


if __name__ == "__main__":
    raise SystemExit(main())
