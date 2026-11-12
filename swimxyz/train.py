"""Train the stroke-signal model on SwimXYZ crops and count strokes on the real clips.

    .venv/bin/python train.py [steps]
Model (from scratch): per-frame CNN -> dilated temporal convs -> 3 signals per frame (left hand, right hand, any hand).
Targets are Gaussian bumps at each stroke event; strokes are counted as peaks of the "any" signal.
"""
import glob, json, random, sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks

T = 32            # frames per training window (~1.1 s at 30 fps)
SIGMA = 2.0       # width of the target bump, in frames
DEV = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)


# ---------------- data ----------------
def load_synth():
    vids = []
    for p in sorted(glob.glob("data/crops/**/*.npz", recursive=True)):
        d = np.load(p)
        ev = [d["events_L"] * d["fps"], d["events_R"] * d["fps"]]  # event times in output frames
        vids.append(dict(path=p, crops=d["crops"], valid=d["valid"], ev=ev))
    return vids


def targets(ev, frames):
    """(len(frames), 3) Gaussian bumps at the events: left, right, any."""
    t = np.asarray(frames, float)[:, None]
    lr = [np.exp(-0.5 * ((t - np.asarray(e)[None]) / SIGMA) ** 2).max(1) if len(e) else np.zeros(len(t)) for e in ev]
    return np.stack(lr + [np.maximum(*lr)], 1).astype(np.float32)


def degrade(im):
    """Make a clean synthetic crop look like poor real footage."""
    h, w = im.shape[:2]
    if rng.random() < 0.7:  # zoom / shift, like an imperfect tracker box
        s = rng.uniform(0.75, 1.3)
        M = np.float32([[s, 0, (1 - s) * w / 2 + rng.uniform(-0.1, 0.1) * w], [0, s, (1 - s) * h / 2 + rng.uniform(-0.1, 0.1) * h]])
        im = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:  # low resolution
        f = rng.uniform(2, 5)
        im = cv2.resize(cv2.resize(im, (int(w / f), int(h / f)), interpolation=cv2.INTER_AREA), (w, h))
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5, 7]))
        im = cv2.GaussianBlur(im, (k, k), 0)
    if rng.random() < 0.6:  # heavy compression
        _, b = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(8, 50))])
        im = cv2.imdecode(b, cv2.IMREAD_COLOR)
    return im


def sample(vids):
    """One window: random valid run, random time-stretch (real swimmers stroke faster), augment."""
    while True:
        v = vids[rng.integers(len(vids))]
        speed = rng.uniform(0.9, 1.9)
        need = int(np.ceil(T * speed)) + 1
        ok = np.where(np.convolve(v["valid"], np.ones(need), "valid") == need)[0]  # window starts fully in view
        if len(ok):
            break
    start = ok[rng.integers(len(ok))]
    src = start + np.arange(T) * speed
    frames = np.round(src).astype(int)
    y = targets(v["ev"], src)
    flip = rng.random() < 0.5
    gain, bias = rng.uniform(0.6, 1.4, 3), rng.uniform(-30, 30, 3)  # per-window colour shift
    xs = []
    for f in frames:
        im = np.clip(v["crops"][f] * gain + bias, 0, 255).astype(np.uint8)
        im = degrade(im)
        if rng.random() < 0.3:  # cutout: splash or another swimmer hiding part of the body
            ch, cw = int(rng.integers(10, 40)), int(rng.integers(20, 70))
            y0, x0 = int(rng.integers(0, 64 - ch)), int(rng.integers(0, 128 - cw))
            im[y0:y0 + ch, x0:x0 + cw] = rng.integers(150, 255)
        xs.append(im[:, ::-1] if flip else im)
    if flip:
        y = y[:, [1, 0, 2]]  # mirroring swaps the left and right arm
    return np.stack(xs), y


def to_tensor(x):  # (B, T, H, W, 3) uint8 -> (B, T, 3, H, W) float
    return torch.from_numpy(np.ascontiguousarray(x)).float().div(255).permute(0, 1, 4, 2, 3)


# ---------------- model ----------------
class StrokeNet(nn.Module):
    def __init__(self, c=96):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.frame = nn.Sequential(block(3, 24), block(24, 48), block(48, 64), block(64, c), nn.AdaptiveAvgPool2d(1))
        self.temporal = nn.ModuleList([nn.Sequential(nn.Conv1d(c, c, 3, padding=d, dilation=d), nn.BatchNorm1d(c), nn.ReLU())
                                       for d in (1, 2, 4, 8)])
        self.head = nn.Conv1d(c, 3, 1)

    def forward(self, x):  # (B, T, 3, H, W) -> (B, T, 3) logits
        B, Tn = x.shape[:2]
        h = self.frame(x.flatten(0, 1)).view(B, Tn, -1).transpose(1, 2)
        for layer in self.temporal:
            h = h + layer(h)  # residual: keeps the per-frame evidence while adding context
        return self.head(h).transpose(1, 2)


# ---------------- real-clip evaluation ----------------
@torch.no_grad()
def predict(model, crops, stride=8):
    """Average overlapping window predictions over a whole clip -> (n, 3) probabilities."""
    model.eval()
    n = len(crops)
    out, cnt = np.zeros((n, 3)), np.zeros((n, 1))
    for s in list(range(0, max(n - T, 0) + 1, stride)) + [max(n - T, 0)]:
        x = to_tensor(crops[None, s:s + T]).to(DEV)
        p = torch.sigmoid(model(x))[0].cpu().numpy()
        out[s:s + len(p)] += p
        cnt[s:s + len(p)] += 1
    model.train()
    return out / np.maximum(cnt, 1)


def count_strokes(signal, fps=30.0):
    pk, _ = find_peaks(signal, height=0.3, distance=int(0.25 * fps))
    return len(pk), pk


def evaluate(model):
    rows = []
    for p in sorted(glob.glob("data/real/*.npz")):
        d = np.load(p)
        lo, hi = int(d["breakout"]), int(d["finish"])
        prob = predict(model, d["crops"][lo:hi])
        n, _ = count_strokes(prob[:, 2])
        rows.append((p.split("/")[-1][:-4], int(d["strokes"]), n))
    err = np.mean([abs(n - gt) / gt for _, gt, n in rows])
    print("  real clips (clip, true, predicted):", rows, f"| mean abs error {err:.0%}", flush=True)
    return rows, err


# ---------------- training ----------------
if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    vids = load_synth()
    random.Random(0).shuffle(vids)
    n_val = max(1, len(vids) // 8)
    val, train = vids[:n_val], vids[n_val:]
    print(f"{len(train)} training videos, {len(val)} validation videos, device {DEV}", flush=True)
    model = StrokeNet().to(DEV)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0, device=DEV))  # bumps are rare; weight them up
    best = 1e9
    for step in range(1, steps + 1):
        batch = [sample(train) for _ in range(16)]
        x = to_tensor(np.stack([b[0] for b in batch])).to(DEV)
        y = torch.from_numpy(np.stack([b[1] for b in batch])).to(DEV)
        loss = lossf(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0 or step == steps:
            with torch.no_grad():
                model.eval()
                vb = [sample(val) for _ in range(32)]
                vl = lossf(model(to_tensor(np.stack([b[0] for b in vb])).to(DEV)), torch.from_numpy(np.stack([b[1] for b in vb])).to(DEV)).item()
                model.train()
            print(f"step {step}: train loss {loss.item():.4f}, val loss {vl:.4f}", flush=True)
            if vl < best:
                best = vl
                torch.save(model.state_dict(), "data/stroke_net.pt")
    model.load_state_dict(torch.load("data/stroke_net.pt"))
    rows, err = evaluate(model)
    json.dump(dict(rows=rows, mean_abs_error=err, steps=steps, n_train=len(train)), open("data/real_results.json", "w"), indent=1)
