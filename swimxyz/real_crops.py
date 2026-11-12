"""Crop the tracked swimmer from real clips, using Swimline's saved SAM 2 boxes, at 30 fps.

    .venv/bin/python real_crops.py swimvid3 swimvid4 ...
Reads ../<clip>.mp4, ../outputs/<clip>/sam2_track.json and ../data/labels.csv; writes data/real/<clip>.npz.
"""
import csv, json, os, sys

import cv2
import numpy as np

from build_crops import CROP_H, CROP_W, crop

ROOT = ".."


def build(clip, label):
    track = json.load(open(f"{ROOT}/outputs/{clip}/sam2_track.json"))
    tracked = {f["frame"]: f["mask"]["box"] for f in track["frames"] if f.get("mask")}
    t = np.array(sorted(tracked))
    B = np.array([tracked[i] for i in t], float)
    # fill tracking gaps by straight-line interpolation between tracked frames (no extrapolation past the ends)
    boxes = {i: [np.interp(i, t, B[:, k]) for k in range(4)] for i in range(t[0], t[-1] + 1)}
    cap = cv2.VideoCapture(f"{ROOT}/{clip}.mp4")
    fps = cap.get(5)
    step = max(1, round(fps / 30))  # 60 fps clips -> every 2nd frame
    crops, valid, f = [], [], 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if f % step == 0:
            b = boxes.get(f)
            valid.append(b is not None)
            crops.append(crop(im, np.array([[b[0], b[1]], [b[2], b[3]]], float)) if b else np.zeros((CROP_H, CROP_W, 3), np.uint8))
        f += 1
    to30 = lambda frame: int(frame) // step  # labels.csv frames are in the clip's own frame rate
    return dict(crops=np.stack(crops), valid=np.array(valid), fps=fps / step, tracked_share=len(t) / (t[-1] - t[0] + 1),
                breakout=to30(label["breakout_frame"]), finish=to30(label["finish_frame"]), strokes=int(label["strokes"]))


if __name__ == "__main__":
    labels = {r["clip"]: r for r in csv.DictReader(open(f"{ROOT}/data/labels.csv"))}
    os.makedirs("data/real", exist_ok=True)
    for clip in sys.argv[1:]:
        d = build(clip, labels[clip])
        assert len(d["crops"]) == len(d["valid"]) and d["breakout"] < d["finish"] <= len(d["crops"]) + 1, clip
        np.savez_compressed(f"data/real/{clip}.npz", **d)
        print(f"{clip}: {len(d['valid'])} frames at {d['fps']:.0f} fps, real boxes {d['tracked_share']:.0%} of the tracked span, "
              f"swim window {d['breakout']}-{d['finish']}, strokes {d['strokes']}")
