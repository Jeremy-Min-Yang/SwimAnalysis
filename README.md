# Swim Analyzer

Race stats from a phone video of a freestyle swimmer: splits, speed, strokes
per length, stroke rate, distance per stroke, and time to the first stroke
after the start and each turn. No pose skeleton: only what the video shows
reliably is measured, and a length without a clear stroke rhythm gets no
stroke numbers rather than a guess.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

`track_sam2.py` uses the SAM 2.1 tiny weights, `sam2.1_t.pt` (Ultralytics
downloads them on first use). Videos and weights are not in the repository;
copy the videos over yourself.

### Which processor runs it

`--device auto` (the default) picks an NVIDIA GPU, or an AMD GPU through
ROCm, then an Apple GPU, then the CPU, and prints which. Speed at the
default `--imgsz 640`:

| machine | speed |
|---|---|
| M2 MacBook (8 GB), Apple GPU | about 3 frames/s |
| M2 CPU | about 2x slower |

AMD Radeon RX 6000 cards (such as the RX 6700 XT) have no official PyTorch
GPU support on Windows, so there it runs on the CPU. For the GPU, use Linux
with ROCm (set `HSA_OVERRIDE_GFX_VERSION=10.3.0` for the RX 6700 XT) or a
cloud NVIDIA GPU such as Google Colab.

## 1. Follow the swimmer

```bash
python src/track_sam2.py swimvid5.mp4 --seed 300:350,890,1079,1060
```

Mark the swimmer with `--seed frame:x1,y1,x2,y2` (a box around them, in
pixels) or `--seed frame:x,y` (a point on them) on a frame where they are
already swimming; a swimmer standing on the block or at the wall looks
nothing like one swimming. A box is more reliable in splash, where a point
can pick up the splash instead. Tracking starts at the first seed.

SAM 2 then follows the swimmer's outline frame to frame. Masks that vanish,
jump, balloon, or turn into a thin line (a lane rope) are rejected. Watch
`outputs/<video>/sam2_track.mp4`; where the overlay drifts off the swimmer
or says LOST, add another `--seed` on that frame and run again. Runs at
about 3 frames/s on an M2 (`--imgsz 1024` is about 5x slower).

## 2. Race stats

```bash
python src/race_stats.py swimvid4.mp4 --pool 25 --start 3.13 --touch 15.22,28.63
```

- `--pool 25` or `50`, `--unit m` (default) or `yd`
- `--start`: second the swimmer leaves the block or wall
- `--touch`: second of each wall touch, turns then the finish
- `--peak cycle` (default) or `--peak arm`: see below

The distance is the pool length times the number of touches. Prints a table
and writes `outputs/<video>/race_stats.json`.

Strokes come from how far the swimmer's mask reaches above its own center:
a recovering arm lifts it. Usually only the near arm clears the swimmer's
outline, so each rise is a full arm cycle, two strokes (`--peak cycle`;
counts are then even and can be one off a hand count). Filmed close and
side-on at water level, both arms clear it and each rise is one stroke
(`--peak arm`, as in swimvid7). The video alone does not tell which. Strokes
count each arm; distance per stroke uses the whole length, underwater part
included.

## Accuracy so far

| video | truth | measured |
|---|---|---|
| swimvid4, 50 m (25 m pool) | 11.9 / 25.5 official; 14 + 20 strokes by hand | 12.08 / 25.49; 12 + 22 strokes |
| swimvid5, 25 yd | 15 strokes | 14 |
| swimvid3, 25 m | 13 strokes | 12 |
| swimvid7, 25 m (`--peak arm`) | 11 strokes | 10 |

Seeds used: swimvid4 `--seed 0:594,953,766,1284 --seed 260:580,1220,1079,1350`,
swimvid5 `--seed 0:505,625,808,787 --seed 300:350,890,1079,1060`,
swimvid3 `--seed 130:95,275,280,315`, swimvid7 `--seed 232:200,300,1290,550`.

Times are only as good as `--start` and `--touch`. Stroke counts are within
about 2 and depend on the track: strokes during a stretch where the swimmer
was lost, or before the first seed, are missed (swimvid4 length 1, swimvid7),
and underwater dolphin kicks just before a breakout can count as a stroke
(swimvid4 length 2).

## Next

- A tap-to-mark page for seeds and start/touch times instead of pixel
  coordinates.
- A swimmer detector trained on box labels (bootstrapped from good tracks),
  so the swimmer is found and re-found without seeds.

Older experiments (RTMPose/YOLO pose tracking, ViTPose-SwimXYZ, pose
fine-tuning) are in the first git commit.
