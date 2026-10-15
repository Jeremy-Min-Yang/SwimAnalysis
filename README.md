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

`track_sam2.py` uses the SAM 2.1 tiny weights, `sam2.1_t.pt` (Ultralytics
downloads them on first use).

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

The distance is the pool length times the number of touches. Prints a table
and writes `outputs/<video>/race_stats.json`.

Strokes come from how far the swimmer's mask reaches above its own center:
the recovering arm lifts it once per arm cycle. Strokes count both arms (two
per cycle), so counts are even and can be one off a hand count. Distance per
stroke uses the whole length, underwater part included.

## Accuracy so far

| video | truth | measured |
|---|---|---|
| swimvid4, 50 m (25 m pool) | 11.9 / 25.5 official; 14 + 20 strokes by hand | 12.08 / 25.49; 14 + 20 strokes |
| swimvid5, 25 yd | 15 strokes | 14 strokes (one extra seed needed) |

Times are only as good as `--start` and `--touch`. swimvid3 and swimvid7 are
not counted yet: the tracker lost the swimmer (to a person on deck, splash,
or another lane) and still needs seeds.

## Next

- A tap-to-mark page for seeds and start/touch times instead of pixel
  coordinates.
- A swimmer detector trained on box labels (bootstrapped from good tracks),
  so the swimmer is found and re-found without seeds.

Older experiments (RTMPose/YOLO pose tracking, ViTPose-SwimXYZ, pose
fine-tuning) are in the first git commit.
