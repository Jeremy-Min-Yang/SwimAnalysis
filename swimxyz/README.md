# SwimXYZ class project

Class project by Jeong Min Yang, kept separate from Swimline (the app in this repo). It trains a stroke-detection
model on the synthetic [SwimXYZ](https://g-fiche.github.io/research-pages/swimxyz/) dataset (CC BY 4.0), tests it on
real freestyle footage, and compares it with Swimline.

## What it does

Video in → pick the swimmer (in a race: type the lane number and click the swimmer once) → the swimmer is tracked and
cropped every frame → **our model** reads a short window of crops and outputs a smooth stroke signal for each arm, with
peaks at the hand entries → stroke count, stroke rate, and left/right timing. Combined with speed from the track:
distance per stroke (DPS) and stroke index (SI = speed × DPS).

It doesn't use pose estimation. Off-the-shelf pose models lose the arms in race footage because of splash,
compression and other swimmers (see `STATUS.md`, "How we got here").

## Data

| Data | Use |
|---|---|
| SwimXYZ freestyle videos (partial download, Zenodo 8402009/8402031) + labels (Zenodo 8399376) | Training: swimmer crops from the 2D joints, stroke targets from the 3D joints |
| `data/labels.csv`, swimvid3–37: single swimmer, side view | Real test, easier tier: stroke counts per clip |
| `data/labels.csv`, swimvid40–56, and `*.50Free*.mp4`: race footage from the stands | Real test, harder tier: stroke counts, plus tapped hand-entry times on the 3 new clips |

SwimXYZ contains no real swimming speed (forward motion is a constant set by the generator), so speed is measured
from the track and checked against `data/labels.csv` (pool length ÷ (finish − start)).

## Running it

**1. Build the training set** (download-heavy: run it on a fast connection). Works on macOS and Windows.

```
cd swimxyz
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip install -r requirements.txt
# put Freestyle_labels.zip (Zenodo 8399376, 1.7 GB) in data/raw/
.venv/bin/python build_crops.py 120 Side_water_level     # Windows: .venv\Scripts\python build_crops.py ...
.venv/bin/python build_crops.py 100 Side_above_water
.venv/bin/python build_crops.py 80 Side_underwater
.venv/bin/python build_crops.py 60 Aerial
```

Run the four commands in separate terminals to download in parallel. Each video (15–36 MB) is pulled out of the
Zenodo zip, cropped and deleted; only `data/crops/` (about 1 MB per video) is kept. Stopping and restarting is safe:
finished videos are skipped. If a full video zip has already been downloaded and extracted to
`data/videos/Freestyle/<View>/...`, those local files are used instead of the network.

**2. Crop the real clips** (needs the clip `.mp4` in the repo root and Swimline's `outputs/<clip>/sam2_track.json`):
`.venv/bin/python real_crops.py swimvid3 swimvid4 ...`

**3. Train and count strokes on the real clips:** `.venv/bin/python train.py 3000`

## Files

- `STATUS.md`: master status file: plan, decisions with reasons, findings, next steps.
- `build_crops.py`: SwimXYZ videos → swimmer crops + per-arm stroke events (training data).
- `real_crops.py`: real clips + Swimline tracks → swimmer crops + stroke totals (test data).
- `train.py`: StrokeNet, our from-scratch model; training and real-clip stroke counting.
- `notebooks/01_explore_labels.ipynb`: dataset exploration (Colab).
- `smpl_fk.py`, `check_fk_reference.py`: SMPL forward kinematics and its check against `smplx`, from the earlier
  joint-based plan (parked).
- `data/` (gitignored): downloads, the SMPL model (licensed, never commit), figures.

## Citation

Fiche, Sevestre, Gonzalez-Barral, Leglaive, Séguier. *SwimXYZ: A large-scale dataset of synthetic swimming motions
and videos.* ACM SIGGRAPH MIG 2023. doi:10.1145/3623264.3624440
