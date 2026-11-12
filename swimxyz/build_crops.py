"""Build the training set from SwimXYZ freestyle videos: swimmer crops (30 fps) + per-arm stroke events.

    .venv/bin/python build_crops.py <videos_per_view> [views...]
Each video is pulled out of the remote Zenodo zip (range requests), cropped, and deleted again; only the small crops
are kept in data/crops/. Already-built videos are skipped, so the script can be stopped and restarted.
"""
import io, os, random, sys, zipfile

import cv2
import numpy as np
import pandas as pd
from remotezip import RemoteZip
from scipy.signal import find_peaks

URLS = {"Aerial": "https://zenodo.org/records/8402009/files/Freestyle_part1.zip?download=1",
        "Side_above_water": "https://zenodo.org/records/8402031/files/Freestyle_part2.zip?download=1",
        "Side_underwater": "https://zenodo.org/records/8402031/files/Freestyle_part2.zip?download=1",
        "Side_water_level": "https://zenodo.org/records/8402031/files/Freestyle_part2.zip?download=1"}
LABELS = "data/raw/Freestyle_labels.zip"
OUT = "data/crops"
CROP_H, CROP_W = 64, 128
FPS_IN, STEP = 60, 2  # SwimXYZ is 60 fps; keep every 2nd frame to match the 30 fps real clips


def load_joints(zf, path):
    df = pd.read_csv(io.BytesIO(zf.read(path)), sep=";", decimal=",")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    names = list(dict.fromkeys(c.rsplit(".", 1)[0] for c in df.columns))
    return df.to_numpy(np.float32).reshape(len(df), len(names), 3)[:-2], names  # last 2 rows are broken reset frames


def stroke_events(J3, names):
    """Input-rate frames where each hand is furthest forward along the body (start of the pull): one per stroke."""
    g = lambda j: J3[:, names.index(j)]
    axis = g("Neck") - g("MidHip")
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    return {s: find_peaks(np.sum((g(s + "Wrist") - g(s + "Shoulder")) * axis, 1),
                          distance=int(0.4 * FPS_IN), prominence=0.2)[0] for s in "LR"}


def crop(im, pts):
    """Fixed 2:1 crop around the joints, a bit larger than the body, resized to CROP_W x CROP_H."""
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    w = max(hi[0] - lo[0], 2 * (hi[1] - lo[1])) * 1.25
    h = w / 2
    s = CROP_W / w
    M = np.float32([[s, 0, CROP_W / 2 - c[0] * s], [0, s, CROP_H / 2 - c[1] * s]])
    return cv2.warpAffine(im, M, (CROP_W, CROP_H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def build_one(video_path, name, zf):
    cap = cv2.VideoCapture(video_path)
    W, H = int(cap.get(3)), int(cap.get(4))
    base = "Freestyle/" + name[:-len(".webm")] + "/body25/"
    J2, names = load_joints(zf, base + "2D_cam.txt")
    J3, _ = load_joints(zf, base + "3D_cam.txt")
    J2[:, :, 1] = H - J2[:, :, 1]  # Unity measures y from the bottom of the image
    inside = ((J2[..., 0] > 0) & (J2[..., 0] < W) & (J2[..., 1] > 0) & (J2[..., 1] < H)).mean(1) >= 0.8  # most of the body in view
    crops, valid = [], []
    for f in range(len(J2)):
        ok, im = cap.read()
        if not ok:
            break
        if f % STEP == 0:
            valid.append(bool(inside[f]))
            crops.append(crop(im, J2[f, :, :2]) if inside[f] else np.zeros((CROP_H, CROP_W, 3), np.uint8))
    ev = stroke_events(J3, names)
    return dict(crops=np.stack(crops), valid=np.array(valid),
                events_L=ev["L"] / FPS_IN, events_R=ev["R"] / FPS_IN, fps=FPS_IN / STEP)  # events in seconds


def main(n_per_view, views):
    zf = zipfile.ZipFile(LABELS)
    rng = random.Random(0)
    for url in dict.fromkeys(URLS[v] for v in views):
        with RemoteZip(url) as rz:
            all_names = [i.filename for i in rz.infolist() if i.filename.endswith(".webm")]
            for view in [v for v in views if URLS[v] == url]:
                names = sorted(n for n in all_names if n.startswith(view + "/"))
                for name in rng.sample(names, min(n_per_view, len(names))):
                    out = os.path.join(OUT, name[:-len(".webm")] + ".npz")
                    if os.path.exists(out):
                        continue
                    local = os.path.join("data/videos/Freestyle", name)
                    tmp = os.path.join("data", f"tmp_{os.getpid()}.webm")  # closed before reading: Windows can't reopen an open file
                    try:
                        if not os.path.exists(local):
                            with open(tmp, "wb") as f:
                                f.write(rz.read(name))
                            local = tmp
                        d = build_one(local, name, zf)
                    finally:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    n_ev = len(d["events_L"]) + len(d["events_R"])
                    # a 5 s clip at ~1 s per cycle has about 4-6 events per arm; far outside that means a bad label
                    if not (d["valid"].shape[0] == d["crops"].shape[0] and 4 <= n_ev <= 16):
                        print(f"skipped (bad labels, {n_ev} events): {name}", flush=True)
                        continue
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with open(out + ".part", "wb") as f:  # rename when complete, so a stopped run never leaves a half file
                        np.savez_compressed(f, **d)
                    os.replace(out + ".part", out)
                    print(f"{view:17s} valid {d['valid'].sum():3d}/{len(d['valid'])} events L{len(d['events_L'])} R{len(d['events_R'])}  {name}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2:] or list(URLS))
