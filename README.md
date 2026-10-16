# Swimline

Race stats from a side-view phone video of a freestyle swimmer: splits, speed, strokes
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

## Portfolio

A one-page summary of the project is in `docs/portfolio/`
(`Swimline_Portfolio.pdf`, Korean `Swimline_Portfolio_KO.pdf`), built from
`swimline.html` and `swimline_ko.html`.

## The app

```bash
streamlit run src/app.py
```

Drop a side-view freestyle video on the page and pick the pool length. The
analysis (`src/analyze.py`, in its own process) takes a few minutes, about
3 frames a second on an M2, then the page shows the overlay video, time,
speed, strokes, stroke rate, distance per stroke and splits per length, the
frames used as the start, walls and finish, suggestions, and charts.
Everything is found automatically:

- **Swimmer**: YOLO-World asked for "swimmer" every few frames (tall boxes,
  people standing on the deck, are dropped), chained into one path; SAM 2
  follows the outline and is re-seeded from that path wherever it loses the
  swimmer or strays off them.
- **Times** (`src/timing.py`): the swimmer's motion along the pool, with the
  camera's pan (optical flow) taken out. The start is the first steady
  motion, or the start beep just before it; a wall is where the direction
  flips; the finish is the reach after the last stroke, or the moment they
  stop less 1.5 s of drift. A clip already moving when it begins, or never
  stopping, shows part of a swim and gets stroke rhythm only.
- **Stroke type**: whether each peak is one arm or a full cycle, from how
  alike successive peaks are (set from only six clips; weak).

"Adjust" under the results corrects any time or the stroke type, using
pictures of the first strokes. Results are kept per video (by content) in
`outputs/upload_<hash>/`, so the same video reopens instantly.

Automatic against truth so far: swimvid4 start 2.70 / turn 14.95 / finish
29.40 against 3.13 / 15.22 / 28.63 (total 26.7 s against 25.5 s, 12 + 16
strokes against 14 + 20); swimvid5 start 3.23 / finish 16.92 against 3 / 16,
16 strokes against 15. The finish is the weakest time (up to about 2 s off).

The scripts below run the same steps by hand.

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
| swimvid13, side view, mid-pool | 23 strokes, about 56/min (counter in the video) | 21 strokes, 56/min |
| swimvid2, side view from above, mid-pool | about 22 strokes, 1.2 s cycle (hand count) | 20 strokes, 1.21 s cycle |
| swimvid14, swimvid15, race view, one lane | 39.83 (100 yd), 26.76 (50 m) | swimmer lost: the mask settles on splash or a lane rope |

Seeds used: swimvid13 `--seed 150:105,157,896,349`, swimvid2
`--seed 73:541,288,1413,698`, swimvid4 `--seed 0:594,953,766,1284 --seed 260:580,1220,1079,1350`,
swimvid5 `--seed 0:505,625,808,787 --seed 300:350,890,1079,1060`,
swimvid3 `--seed 130:95,275,280,315`, swimvid7 `--seed 232:200,300,1290,550`.

Times are only as good as `--start` and `--touch`. Stroke counts are within
about 2 and depend on the track: strokes during a stretch where the swimmer
was lost, or before the first seed, are missed (swimvid4 length 1, swimvid7),
and underwater dolphin kicks just before a breakout can count as a stroke
(swimvid4 length 2).

On the race views, SAM 2's presence score is the same (about 0.27) on good
and bad tracks, so a lost swimmer can only be spotted in the overlay video.

## Next

- Race view (v2): crop to the chosen swimmer's lane before tracking, so they
  fill the frame and the mask cannot jump to another lane or a rope.
- A swimmer detector trained on box labels (bootstrapped from good tracks),
  so the swimmer is found and re-found without seeds.

Older experiments (RTMPose/YOLO pose tracking, ViTPose-SwimXYZ, pose
fine-tuning) are in the first git commit.
