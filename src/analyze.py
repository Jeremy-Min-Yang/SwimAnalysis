"""Analyze a side-view swim video with no input beyond the video itself.

1. Find the swimmer: YOLO-World, asked for "swimmer", checks every few
   frames. A swimmer lies flat, so tall boxes (people on the deck) are
   dropped, and the boxes are chained into one path: the swimmer the camera
   follows, starting from the strongest detection near the middle.
2. Follow them with SAM 2 (track_sam2.py), seeded from that path. Wherever
   SAM 2 loses the swimmer, or its mask sits outside the path's box for a
   while, it is seeded again from the path.
3. Find the start, wall touches and finish from the swimmer's motion
   (timing.py), the start beep if there is one, and the stroke rhythm.
4. Decide whether a stroke peak is one arm or a full arm cycle.

Writes outputs/<video>/sam2_track.json, sam2_track.mp4 and analysis.json.
Prints "PROGRESS <fraction> <message>" lines for the page to show.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import race_stats as rs  # noqa: E402
import timing  # noqa: E402
import track_sam2  # noqa: E402

DETECTOR = ROOT / "models" / "yolov8s-worldv2.pt"
SAM_MODEL = ROOT / "sam2.1_t.pt"
DETECTIONS_PER_S = 6
MIN_CONFIDENCE = 0.08
# Boxes taller than this many times their width are people standing up.
MAX_TALL = 1.1
# Seed or re-seed only from detections at least this confident.
SEED_CONFIDENCE = 0.25
# A path step may move at most this many box widths per second.
MAX_PATH_SPEED = 2.0
# and at most this many swimmer heights per second up or down: the camera
# pans along the pool, so a big vertical step is another lane. Without it
# the path hops to a far-lane swimmer whenever the near one is underwater
# and not detected. The height is the median of the last PATH_HEIGHTS boxes.
MAX_PATH_VERTICAL_SPEED = 1.0
PATH_HEIGHTS = 6
# Re-seed when lost this many frames, or when the mask centroid is outside
# the path box (grown by BOX_MARGIN) on this many path frames in a row.
LOST_FRAMES = 10
OUTSIDE_CHECKS = 3
BOX_MARGIN = 0.25
# Finish: the last stroke plus this many cycles (the reach to the wall),
# when strokes were seen until at most FINISH_STROKE_GAP_S before the swimmer
# stopped moving; otherwise (the track lost them near the wall) the moment
# they stopped, less the usual drift after touching, MOTION_OVERRUN_S.
FINISH_REACH_CYCLES = 0.75
FINISH_STROKE_GAP_S = 4.0
MOTION_OVERRUN_S = 1.5
# A beep at most this many seconds before the swimmer moves is the start.
BEEP_BEFORE_MOVE_S = 2.0
# Arm or cycle: successive peaks of similar height that also repeat two
# peaks on mean both arms are seen (one stroke per peak). Set from six clips.
ARM_HEIGHT_STEP = 0.06
ARM_REPEAT = 0.84


def progress(fraction: float, message: str) -> None:
    print(f"PROGRESS {fraction:.3f} {message}", flush=True)


def detect(video: Path, device: str) -> tuple[dict[int, list[tuple[list[float], float]]], dict]:
    """Swimmer boxes and confidences on every few frames."""
    from ultralytics import YOLOWorld

    # Ultralytics downloads the weights by name if they are not in models/.
    model = YOLOWorld(str(DETECTOR) if DETECTOR.exists() else DETECTOR.name)
    model.set_classes(["swimmer", "person swimming"])
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    info = {
        "fps": fps,
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
    }
    step = max(1, round(fps / DETECTIONS_PER_S))
    detections = {}
    index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if index % step == 0:
            result = model(frame, conf=MIN_CONFIDENCE, verbose=False, device=device)[0]
            boxes = []
            for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                width, height = box[2] - box[0], box[3] - box[1]
                if height <= MAX_TALL * width:
                    boxes.append(([float(v) for v in box], float(confidence)))
            detections[index] = boxes
            if index % (step * 20) == 0:
                progress(0.1 * index / max(1, info["frames"]), "Looking for the swimmer")
        index += 1
    capture.release()
    del model
    return detections, info


def center(box: list[float]) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def swimmer_path(detections: dict, info: dict) -> dict[int, tuple[list[float], float]]:
    """One box per detection frame (where found) for the swimmer the camera
    follows: chained from the best detection near the middle of the picture,
    each step taking the nearest box within reach."""
    def score(box, confidence):
        offset = abs(center(box)[0] - info["width"] / 2) / (info["width"] / 2)
        area = (box[2] - box[0]) * (box[3] - box[1])
        return confidence * np.sqrt(area) * (1 - 0.5 * offset)

    candidates = [(score(b, c), f, b, c) for f, boxes in detections.items() for b, c in boxes]
    if not candidates:
        return {}
    _, anchor, box, confidence = max(candidates)
    path = {anchor: (box, confidence)}
    frames = sorted(detections)
    position = frames.index(anchor)
    for direction in (1, -1):
        last_frame, last_box = anchor, box
        heights = [box[3] - box[1]]
        order = frames[position + 1 :] if direction == 1 else frames[:position][::-1]
        for frame in order:
            seconds = abs(frame - last_frame) / info["fps"]
            reach = (last_box[2] - last_box[0]) * (0.5 + MAX_PATH_SPEED * seconds)
            vertical_reach = np.median(heights[-PATH_HEIGHTS:]) * (0.5 + MAX_PATH_VERTICAL_SPEED * seconds)
            near = [
                (np.linalg.norm(center(b) - center(last_box)), b, c)
                for b, c in detections[frame]
                if np.linalg.norm(center(b) - center(last_box)) <= reach
                and abs(center(b)[1] - center(last_box)[1]) <= vertical_reach
            ]
            if near:
                _, chosen, chosen_confidence = min(near, key=lambda item: item[0] - 50 * item[2])
                path[frame] = (chosen, chosen_confidence)
                last_frame, last_box = frame, chosen
                heights.append(chosen[3] - chosen[1])
    return dict(sorted(path.items()))


def make_reseeder(path: dict):
    """reseed() for track_sam2.track: seeds from the path where SAM 2 lost
    the swimmer or strayed outside the path's box."""
    outside = {"count": 0}

    def reseed(index, last, last_frame):
        found = path.get(index)
        if found is None or found[1] < SEED_CONFIDENCE:
            return None
        box = found[0]
        if index - last_frame >= LOST_FRAMES:
            outside["count"] = 0
            return box
        if last is not None and last_frame == index - 1:
            grow_x, grow_y = BOX_MARGIN * (box[2] - box[0]), BOX_MARGIN * (box[3] - box[1])
            x, y = last["centroid"]
            inside = box[0] - grow_x <= x <= box[2] + grow_x and box[1] - grow_y <= y <= box[3] + grow_y
            outside["count"] = 0 if inside else outside["count"] + 1
            if outside["count"] >= OUTSIDE_CHECKS:
                outside["count"] = 0
                return box
        return None

    return reseed


def strokes_per_peak(track: dict, stats: dict, start: int, end: int) -> int:
    """2 if each stroke peak is a full arm cycle (only the near arm shows),
    1 if both arms show and each peak is one stroke."""
    fps = track["fps"]
    if not stats["peak_s"]:
        return 2
    period = stats["peak_s"] * fps
    raw = rs.fill_gaps(rs.mask_shape_signal(track["frames"]))
    signal = raw - rs.moving_average(raw, max(3, round(period)))
    signal = rs.moving_average(signal, max(1, int(rs.SMOOTH_CYCLES * period)))
    window = signal[start : end + 1] - signal[start : end + 1].mean()
    correlation = rs.autocorrelation(window)
    one, two = int(round(period)), int(round(2 * period))
    if two >= len(correlation) or correlation[one] <= 0:
        return 2
    peaks = [p for length in stats["lengths"] for p in (length["peak_frames"] or [])]
    if len(peaks) < 4:
        return 2
    height_step = float(np.mean(np.abs(np.diff(signal[peaks]))))
    repeat = correlation[two] / correlation[one]
    return 1 if height_step < ARM_HEIGHT_STEP and repeat > ARM_REPEAT else 2


def find_beep(video: Path) -> list[float]:
    # PyAV and OpenCV both bundle FFmpeg; keep PyAV in its own process.
    result = subprocess.run(
        [sys.executable, str(SRC / "start_beep.py"), "--json", str(video)],
        capture_output=True, text=True,
    )
    return json.loads(result.stdout)[str(video)] if result.returncode == 0 else []


def find_times(video: Path, track: dict, reseeds: int) -> dict:
    """Start, wall touches and finish (or a time window), and the stroke type."""
    fps = track["fps"]
    found = [f for f in track["frames"] if f["mask"] is not None]
    progress(0.85, "Finding the start and the walls")
    boxes = [f["mask"]["box"] if f["mask"] else None for f in track["frames"]]
    centroids = np.array([f["mask"]["centroid"] if f["mask"] else [np.nan, np.nan] for f in track["frames"]])
    speed = timing.along_pool_speed(centroids, timing.camera_flow(str(video), boxes), fps)
    found_timing = timing.find_timing(speed, fps)
    last_tracked = max(f["frame"] for f in found) / fps
    first_tracked = min(f["frame"] for f in found) / fps

    result = {
        "video": str(video),
        "fps": fps,
        "frames": len(track["frames"]),
        "found_frames": len(found),
        "first_tracked_s": first_tracked,
        "reseeds": reseeds,
        "start_source": None,
    }
    # A clip already moving when it begins, or never stopping, has no start
    # or no finish: it shows part of a swim, and only the rhythm is measured.
    start = found_timing.get("start")
    whole = bool(found_timing) and found_timing["finished"] and start > 0.5
    if whole:
        beeps = [b for b in find_beep(video) if 0 <= start - b <= BEEP_BEFORE_MOVE_S]
        if beeps:
            start, result["start_source"] = beeps[0], "beep"
        else:
            result["start_source"] = "motion"
        touches = list(found_timing["touches"])
        stopped = touches[-1]
        result["stopped_s"] = round(stopped, 2)
        provisional = rs.race_stats(track, 25, "m", start, touches, 2)
        last_peaks = provisional["lengths"][-1]["peak_frames"]
        if provisional["peak_s"] and last_peaks and stopped - last_peaks[-1] / fps <= FINISH_STROKE_GAP_S:
            touches[-1] = min(stopped, last_peaks[-1] / fps + FINISH_REACH_CYCLES * provisional["peak_s"])
            result["finish_source"] = "last stroke"
        else:
            touches[-1] = stopped - MOTION_OVERRUN_S
            result["finish_source"] = "motion"
        result.update(mode="lengths", start=round(start, 2), touches=[round(t, 2) for t in touches])
    else:
        result.update(mode="window", start=round(max(first_tracked, start or 0), 2),
                      touches=[round(last_tracked, 2)])

    stats = rs.race_stats(track, 25, "m", result["start"], result["touches"], 2)
    result["strokes_per_peak"] = strokes_per_peak(
        track, stats, round(result["start"] * fps), round(result["touches"][-1] * fps)
    )
    return result


def analyze(video: Path, output_dir: Path, device: str) -> dict:
    progress(0.0, "Looking for the swimmer")
    detections, info = detect(video, device)
    path = swimmer_path(detections, info)
    seeds = [f for f, (_, c) in path.items() if c >= SEED_CONFIDENCE]
    if not seeds:
        raise RuntimeError("No swimmer found in the video.")
    first = seeds[0]

    progress(0.1, "Following the swimmer")
    import torch

    if device == "mps":
        torch.mps.empty_cache()
    status = track_sam2.track(
        video, {first: path[first][0]}, str(SAM_MODEL), output_dir, 640, device,
        reseed=make_reseeder(path),
    )
    if status != 0:
        raise RuntimeError("Tracking failed.")
    track = json.loads((output_dir / "sam2_track.json").read_text())
    reseeds = sum(f["mask"] is not None and f["mask"]["source"] == "seed" for f in track["frames"]) - 1

    result = find_times(video, track, reseeds)
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=1))
    progress(1.0, "Done")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retime", action="store_true",
                        help="Only find the times again, from the existing track")
    args = parser.parse_args()
    output_dir = args.output_dir or ROOT / "outputs" / args.video.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.retime:
        track = json.loads((output_dir / "sam2_track.json").read_text())
        old = json.loads((output_dir / "analysis.json").read_text())
        result = find_times(args.video, track, old["reseeds"])
        (output_dir / "analysis.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        return 0
    try:
        result = analyze(args.video, output_dir, track_sam2.pick_device(args.device))
    except RuntimeError as error:
        print(f"Error: {error}", flush=True)
        return 1
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
