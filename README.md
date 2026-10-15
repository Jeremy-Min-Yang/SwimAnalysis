# Swim Analyzer

The first milestone reads a video and displays its basic properties. The second
milestone detects one swimmer and exports a copy with pose landmarks drawn on it.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inspect a video

```bash
python src/inspect_video.py path/to/video.mp4
```

The command reports the file path, resolution, frame rate, frame count,
duration, and codec. The video is not uploaded anywhere or modified.

## Detect and draw the swimmer

```bash
python src/detect_pose.py path/to/video.mp4
```

RTMPose-x finds the keypoints because it tracks the full body, including
legs below the water. It runs on the Apple GPU through CoreML (`--device mps`,
the default); `--device cpu` is about 8x slower. The first run downloads the
models.

Person detectors rarely see a swimmer lying flat in splash, so a detector
(YOLOX-m) only picks the swimmer up and rechecks now and then. Between those,
each frame's keypoints place the box for the next frame. Point at the swimmer
with `--pick x,y` (normalized, e.g. `0.45,0.3`) on the frame at `--pick-time`
seconds; without it the largest person on the first frame is used, which can
be a coach or spectator. The pose model also runs on the crop turned
sideways, which helps with a horizontal swimmer; `--no-rotate` skips that
and is about 3x faster.

Each video gets its own folder: `outputs/<video name>/pose.mp4` (the annotated
video, H.264 so browsers can play it) and `outputs/<video name>/landmarks.json`.
Use `-o` and `--landmarks-output` to choose other paths, or tune detection for
difficult footage:

```bash
python src/detect_pose.py path/to/video.mp4 \
	--pick 0.45,0.3 \
	--detection-confidence 0.15 \
	--keypoint-confidence 0.25 \
	--roi 0,0.15,1,0.85
```

The ROI uses normalized `x1,y1,x2,y2` coordinates. The example focuses on the
middle horizontal band where the swimmer appears in the sample footage while
keeping annotations positioned in the original video.

By default the video draws the swimmer box plus every shoulder-to-ankle keypoint
and bone inside it (with 10% margin, set by `--box-padding`). The drawn box keeps
one size for the whole video (big enough for the swimmer's keypoints in 90% of
frames) and follows the torso, smoothed over 0.3 s; it is saved per frame as
`display_box` in the landmarks, next to the detector's raw `box`.

Only one swimmer is tracked, and it is kept in its lane. Once the swimmer
has swum flat in the water for 0.5 s, the lane ropes on either side
(found by their red or blue floats, see `src/lanes.py`) are locked and
followed through camera pans; poses outside that lane are ignored, so
coaches, spectators, and swimmers in other lanes are never picked. The lane
is drawn in the video and saved per frame as `lane` (each rope as
`[slope, intercept]` of `y = slope * x + intercept`). While the pose is
unsure the box holds still and sweeps along the lane; lost for over 1 s, it
searches all the water in the frame for a swimmer lying flat. Once the
swimmer has left the block, upright poses out of the water (people on the
deck) are rejected. To hide the surroundings, pass
`--background black` or `--background dim`. `--style simple` draws a
single smoothed stick figure on the camera-facing side instead; `--smoothing`
(0 to below 1, default 0.5) controls its smoothing. None of these options
change the exported landmarks.

Detection takes several minutes; drawing takes seconds. To change only the
drawing options, redraw from the saved landmarks:

```bash
python src/detect_pose.py path/to/video.mp4 --render-only --keypoint-confidence 0.25
```

## Race stats

Splits, speed, strokes per length, stroke rate, and distance per stroke,
without a pose skeleton. Two steps:

```bash
python src/track_sam2.py swimvid4.mp4
python src/race_stats.py swimvid4.mp4 --pool 25 --start 3.13 --touch 15.22,28.63
```

`track_sam2.py` follows the swimmer's outline with SAM 2 (`sam2.1_t.pt`) and
writes `outputs/<video>/sam2_track.json` and `sam2_track.mp4`. It starts from
the swimmer's keypoints on frame 0 in `landmarks.json` (run `detect_pose.py`
first) or from `--box x1,y1,x2,y2`. Masks that vanish, jump, or balloon are
rejected, and the track restarts from frames with a confident pose; add
`--seed frame:x1,y1,x2,y2` to restart it by hand, e.g. after a cut. It runs
at about 3 frames/s on an M2 at the default `--imgsz 640`.

`race_stats.py` needs the pool length (`--pool 25` or `50`, `--unit m` or
`yd`), the second the swimmer leaves the block or wall (`--start`), and the
second of each wall touch, turns then finish (`--touch`). It prints a table
and writes `race_stats.json`. Strokes come from how far the swimmer's mask
reaches above its center: the recovering arm lifts it once per arm cycle.
Strokes count both arms (two per cycle); distance per stroke uses the whole
length, underwater part included. A length without a clear stroke rhythm
gets no stroke numbers rather than a guess.

On swimvid4 (50 m freestyle, 25 m pool, official 11.9 / 25.5) it measures
25.49 s and 14 + 20 strokes, matching a hand count. That is one video so
far; check other videos against hand counts before trusting it.

## Build a swimmer training set

Generic pose models rarely see a swimmer in splash, so the plan is to
fine-tune one on swimmer frames. Export frames with draft labels from the
current pipeline (run `detect_pose.py` on each video first):

```bash
python src/export_frames.py videos/*.mp4 --val swimvid3
```

This writes `datasets/swim_pose/` in the Ultralytics YOLO pose format:
images, one label file per image (the swimmer box plus 17 keypoints), and
`data.yaml`. Videos are split whole into train and val. Keypoints below
`--confidence` (default 0.4) are left out, and frames where the tracker was
unsure have empty labels.

Correct the labels in a tool that reads this format, such as CVAT (create a
task with a "swimmer" skeleton of the 17 COCO keypoints, import as
"Ultralytics YOLO Pose", fix and add swimmers, export the same way). Label
every swimmer in the frame, and only the keypoints you can actually see.

Then fine-tune, starting from the COCO-trained YOLO11 pose model:

```bash
python src/train_pose.py --model yolo11s-pose.pt --epochs 100
```

Results go to `runs/pose/swim_pose/`; `weights/best.pt` is the model.

## Compare pose models

```bash
python src/compare_models.py path/to/video.mp4
```

Runs RTMPose-x and `yolo11x-pose.pt` on the same frames and writes a
side-by-side video, still images, and `summary.json` to
`outputs/model_comparison/`. Pass `--models` to compare other YOLO weights.

## Analyze exported landmarks

```bash
python src/analyze_strokes.py outputs/<video name>/landmarks.json
```

This writes `metrics.json` next to the landmarks and prints a per-stroke table.

### Methods

- **Strokes.** In side-view freestyle the near arm swings high above the
  shoulders once per stroke. The script tracks how far the highest wrist rises
  above the shoulders (in torso lengths), estimates the time between strokes by
  autocorrelation, and counts one peak per stroke.
- **Recovery vs. underwater.** Each stroke runs from the start of one recovery
  to the start of the next. The recovery is the stretch around the peak where
  the wrist is above 30% of the way from the surrounding low point to the peak;
  the rest of the stroke is the underwater phase. The first stroke (if already
  underway when the clip starts) and the last stroke are marked incomplete and
  left out of the averages.
- **Head angle.** The angle at the hips between the hip-to-shoulder line and the
  hip-to-head line; positive means the head is lifted above the body line.
  Shoulder roll during the recovery still shows up as some lift, so
  `head_angle_underwater_deg` (the underwater phase only) is the cleanest
  per-stroke number.

Head-on or underwater-only footage is not supported by these methods.

### Output files

Everything for one video lives in `outputs/<video name>/`:

| file | contents |
|---|---|
| `pose.mp4` | annotated video (H.264) |
| `landmarks.json` | per-frame keypoints, box, and which model produced them |
| `metrics.json` | analysis, linked to the videos and landmarks |

`metrics.json` stores paths to the source video, `pose.mp4`, and
`landmarks.json` relative to itself, plus `fps` and `total_frames`. Its three
sections share video frame numbers so a dashboard can sync them to playback
(`time = frame / fps`):

- `summary`: stroke count and rate, mean recovery/underwater time and ratio,
  head angle, elbow and knee angles.
- `strokes`: one row per stroke with `start_frame`, `peak_frame`,
  `recovery_end_frame`, `end_frame` (and the same in seconds), `duration_s`,
  `recovery_s`, `underwater_s`, `recovery_ratio`, head angle stats, and
  `complete`.
- `frames`: columns with one value per video frame: `time_s`, `model`,
  `stroke` (which stroke the frame belongs to), `phase` (`recovery` or
  `underwater`), `wrist_height`, `head_angle_deg`, and `head_height`. Missing
  values are `null`.