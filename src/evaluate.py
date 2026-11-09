"""Run analyze.py on every labeled clip and score it against the hand labels.

Labels are in data/labels.csv, times in frame numbers of the video. Clips
whose row in the labeling sheet was empty are not in the file and are not
analyzed. Each clip's analysis goes to outputs/eval/<clip>/ and is reused
unless --rerun is given; the scores go to outputs/eval/results.csv.

Scored separately, so the stage that fails shows:
- tracking: share of frames with a swimmer mask, and re-seeds;
- start: the app's start against the labeled start;
- breakout: the app's first stroke against the labeled breakout;
- finish: the app's wall touch nearest the labeled finish;
- strokes: the app's strokes from its start to that touch.

Each clip is analyzed only up to END_MARGIN_S after its labeled finish, so a
long race video is not tracked past the length that was labeled. The app
itself reads the whole video; this is an easier test than a real upload.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import race_stats as rs  # noqa: E402

LABELS = ROOT / "data" / "labels.csv"
VIDEOS = ROOT / "swimvideos"
OUTPUT = ROOT / "outputs" / "eval"
# The longest clip (6578 frames) takes about 25 minutes at 4.5 frames/s.
TIMEOUT_S = 3600
# Seconds kept after the labeled finish, so the app can see the swimmer stop.
END_MARGIN_S = 5.0
# Seconds to let analyze.py exit after it has written its analysis.
EXIT_GRACE_S = 15


def find_video(clip: str) -> Path | None:
    found = [p for p in VIDEOS.iterdir() if p.stem == clip]
    return found[0] if found else None


def frame(value: str) -> int | None:
    return int(value) if value.strip().isdigit() else None


def end_frame(video: Path, finish: int) -> int:
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return finish + round(END_MARGIN_S * fps)


def run(video: Path, output_dir: Path, end: int, rerun: bool) -> dict | None:
    analysis = output_dir / "analysis.json"
    stale = analysis.exists() and json.loads(analysis.read_text()).get("end_frame") != end
    if rerun or stale or not analysis.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        analysis.unlink(missing_ok=True)
        # On ROCm the process can hang after it has written its analysis, or
        # after a crash, so it is killed once the analysis is there or on timeout.
        with (output_dir / "log.txt").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(SRC / "analyze.py"), str(video), "-o", str(output_dir), "--end-frame", str(end)],
                stdout=log, stderr=subprocess.STDOUT,
            )
            started = time.monotonic()
            written = None
            while process.poll() is None:
                if analysis.exists() and written is None:
                    written = time.monotonic()
                if (written and time.monotonic() - written > EXIT_GRACE_S) or time.monotonic() - started > TIMEOUT_S:
                    process.kill()
                    process.wait()
                    break
                time.sleep(1)
        if not analysis.exists():
            return None
    return json.loads(analysis.read_text())


def score(label: dict, analysis: dict, track: dict) -> dict:
    fps = analysis["fps"]
    start, breakout, finish = (frame(label[k]) for k in ("start_frame", "breakout_frame", "finish_frame"))
    touches = analysis["touches"]
    nearest = int(np.argmin([abs(t - finish / fps) for t in touches]))
    stats = rs.race_stats(track, 25, "m", analysis["start"], touches[: nearest + 1], analysis["strokes_per_peak"])
    first_stroke = stats["lengths"][0]["first_stroke_s"]
    return {
        "mode": analysis["mode"],
        "tracked": round(analysis["found_frames"] / analysis["frames"], 2),
        "reseeds": analysis["reseeds"],
        "start_err_s": round(analysis["start"] - start / fps, 2),
        "breakout_err_s": round(analysis["start"] + first_stroke - breakout / fps, 2)
        if first_stroke is not None and breakout else None,
        "finish_err_s": round(touches[nearest] - finish / fps, 2),
        "touch_used": f"{nearest + 1}/{len(touches)}",
        "app_strokes": stats["total_strokes"],
        "stroke_err": stats["total_strokes"] - int(label["strokes"]) if stats["total_strokes"] else None,
        "strokes_per_peak": analysis["strokes_per_peak"],
    }


def summarize(rows: list[dict]) -> None:
    def show(name, values):
        signed = np.array([v for v in values if v is not None], dtype=float)
        values = np.abs(signed)
        if len(values):
            print(f"  {name:<16} n={len(values):<3} median |err| {np.median(values):5.2f}  "
                  f"mean {values.mean():5.2f}  max {values.max():6.2f}  bias {np.median(signed):+5.2f}")
        else:
            print(f"  {name:<16} n=0")

    groups = {
        "side view, freestyle": [r for r in rows if r["group"] == "side" and r["stroke"] == "free"],
        "race (far stands), freestyle": [r for r in rows if r["group"] == "race" and r["stroke"] == "free"],
        "other strokes": [r for r in rows if r["stroke"] != "free"],
    }
    for name, group in groups.items():
        scored = [r for r in group if r.get("mode")]
        print(f"\n{name}: {len(group)} clips, {len(group) - len(scored)} failed")
        show("start error s", [r["start_err_s"] for r in scored])
        show("breakout err s", [r["breakout_err_s"] for r in scored])
        show("finish error s", [r["finish_err_s"] for r in scored])
        show("stroke error", [r["stroke_err"] for r in scored])
        print(f"  no stroke count  {sum(r['app_strokes'] is None for r in scored)}"
              f"   window mode {sum(r['mode'] == 'window' for r in scored)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("clips", nargs="*", help="Only these clips (default: all labeled)")
    parser.add_argument("--rerun", action="store_true", help="Analyze again even if a result exists")
    args = parser.parse_args()

    labels = list(csv.DictReader(LABELS.open(encoding="utf-8")))
    if args.clips:
        labels = [l for l in labels if l["clip"] in args.clips]
    rows = []
    for number, label in enumerate(labels, start=1):
        video = find_video(label["clip"])
        row = {"clip": label["clip"], "stroke": label["stroke"],
               "group": "race" if "stands" in label["view"].lower() else "side",
               "label_strokes": label["strokes"]}
        if video is None:
            print(f"[{number}/{len(labels)}] {label['clip']}: no video, skipped", flush=True)
            continue
        print(f"[{number}/{len(labels)}] {label['clip']}", flush=True)
        output_dir = OUTPUT / label["clip"]
        analysis = run(video, output_dir, end_frame(video, int(label["finish_frame"])), args.rerun)
        if analysis is None:
            row["mode"] = None
            print("  analysis failed, see log.txt", flush=True)
        else:
            track = json.loads((output_dir / "sam2_track.json").read_text())
            try:
                row.update(score(label, analysis, track))
            except ValueError as error:
                row["mode"] = None
                print(f"  scoring failed: {error}", flush=True)
            print("  " + json.dumps({k: v for k, v in row.items() if k not in ("clip", "group")}), flush=True)
        rows.append(row)

    fields = ["clip", "group", "stroke", "mode", "tracked", "reseeds", "start_err_s", "breakout_err_s",
              "finish_err_s", "touch_used", "label_strokes", "app_strokes", "stroke_err", "strokes_per_peak"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
