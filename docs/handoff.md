# Swimline handoff (2026-10-23)

Context for picking up this project in a new Claude session.

## The project

**Swimline** analyzes swim videos and reports race stats: splits, speed,
strokes per length, stroke rate and distance per stroke. The owner is
Jeong Min Yang. Use that name in any document, not "Jeremy", which is only
the GitHub handle.

- Repo: `/Users/jeongyang/Desktop/compVisionSwim`, branch `streamlit-app`
  (main branch: `main`). The `diagrams/` rework is not committed yet: the old
  numbered files were deleted and `overview.md` and `technical.md` added.
- Machine: M2 Mac with 8 GB RAM. Run one model at a time, and SAM 2 at
  imgsz 640 (about 3 fps).
- Product rule: **zero input**. The user drops a video and gets results,
  with no manual marking. Whether a single "Is this your swimmer? ✓"
  confirmation would be allowed is still undecided.
- Scope: simple, reliable stats only. No pose or skeleton. A metric that
  can't be measured reliably is left out.

## How the pipeline works

Full write-up: `diagrams/technical.md`.

1. `src/app.py`: Streamlit upload page. It runs `src/analyze.py` as a
   separate process. Uploads are saved as `uploads/<hash>.mp4`, results in
   `outputs/upload_<hash>/`.
2. YOLO-World (`yolov8s-worldv2.pt`, prompts "swimmer" and "person
   swimming") runs at about 6 fps and finds boxes. `swimmer_path` picks one
   swimmer for the whole video.
3. SAM 2 tiny (`sam2.1_t.pt`, driven by `src/track_sam2.py`) masks the
   swimmer on every frame. Bad masks are rejected, memory is controlled, and
   tracking restarts from YOLO boxes when it's lost.
4. `src/timing.py` and `src/start_beep.py` find the start, walls and
   finish. Swimmer motion is the mask centroid's motion minus camera motion
   (optical flow outside the mask). An audio start beep is used when
   present.
5. `src/race_stats.py` counts strokes from the mask's shape signal
   (autocorrelation, then peaks). Each peak is either one arm or one cycle
   (`strokes_per_peak`).

Useful commands:
- `python3 src/analyze.py <video> -o <dir>` runs the full zero-input
  pipeline.
- `python3 src/analyze.py <video> -o <dir> --retime` reruns only the timing
  step, from an existing `sam2_track.json`.
- `rs.race_stats(track, pool, unit, start_s, [touch_s...], strokes_per_peak)`
  returns the stats for each length, including `peak_frames`.

## Current accuracy

Some of these numbers came from runs with hand seeds, not the zero-input app:

- Side view, one swimmer: stroke counts usually within about 2 of a hand
  count. Finish time up to about 2 s off.
- Race view (wide shot, many lanes; test clips swimvid14 and swimvid15):
  **fails**. The SAM 2 mask settles on splash or a lane rope. SAM 2's
  presence score (about 0.27) is the same on good and bad tracks, so it
  can't flag the problem.
- Everything so far was tuned on the same ~8 clips it was tested on
  (overfitting risk). Not accurate enough to ship.

## Strategy agreed: measure first

1. The user is collecting **50–60 clips** (mixed strokes; races and
   practice) and labelling them **by hand**.
2. Claude then runs the zero-input pipeline on every clip and compares the
   results with the labels.
3. Next, find which stage causes the errors (tracking, timing or stroke
   counting) and fix in that order.

### Labelling format (one row per length)

| clip | length # | stroke | pool | view | start frame | breakout frame | touch frame | strokes | notes |
|---|---|---|---|---|---|---|---|---|---|

- Times are **frame numbers**. Check each clip's frame rate is constant
  (swimvid5 is a constant 30 fps). Phone videos can have a variable frame
  rate, and slow-motion clips must be flagged.
- Start: the start signal, or the feet leaving the block or wall.
- Wall: hand touch on open turns (breast, fly), feet touch on flip turns
  (free, back).
- Finish: hand touch.
- Breakout: the head breaks the surface.
- Strokes: free and back count each arm as 1; breast and fly count each full
  cycle as 1. Count from breakout to touch only.
- For race clips, note which lane or swimmer.
- Plans: hold out about 15 clips as a test set that is never used for
  tuning, and recount about 5 clips later to measure the user's own counting
  error.

## First labelled clip: swimvid5

Freestyle, side view, wall start, one length.

| | user | app (current code) |
|---|---|---|
| start | 83F = 2.77 s | 3.23 s (0.47 s late) |
| finish | 492F = 16.40 s | 16.93 s (0.53 s late) |
| split | 13.63 s | 13.70 s |
| strokes | 15 | 16 |

App result: `outputs/upload_57d9832ee028/` (the same file as
`swimvid5.mp4`). Peaks were at frames 213, 249, 287, 325, 364, 401, 439 and
479, with `strokes_per_peak` = 2.

What this clip showed:
1. The start and finish are each about 0.5 s late. The split looks fine only
   because the two errors cancel, so start and finish need scoring
   separately.
2. The underwater glide makes fake stroke peaks. Using the user's start time
   adds peaks at frames 86 and 124, before the breakout at 202, and the
   count jumps to 20. The fix: count strokes only after the breakout.
3. With 2 strokes per peak, the counter can only output even numbers, so
   the true 15 is impossible. This is a built-in ±1 error in the counting
   method.

**Open question:** is swimvid5's pool 25 m or 25 yd? The user said 25 m
this time but yards earlier. It affects only speed and distance per stroke.

## Other known ground truth

- swimvid4: 50 m in a 25 m pool. Official 11.9 / 25.5; hand count 14 + 20
  strokes.
- swimvid3: 25 m, 13 strokes. The swimmer is Cameron McEvoy, so about
  2.75 m/s is real.
- swimvid7: 25 m, 11 strokes.
- swimvid13: 23 strokes, about 56/min.
- swimvid2: about 22 strokes, 1.2 s cycle.
- swimvid14: Jordan Crooks, 100 yd, 39.83.
- swimvid15: 50 m long course, 26.76 (unconfirmed).

## Next steps

1. Wait for the user's labelled sheet.
2. Build an eval harness: run `analyze.py` on every clip, then score start,
   wall and finish errors, split errors, and stroke-count errors per length.
3. Find which stage the errors come from, for example by feeding the stroke
   counter the hand-marked times.
4. Fixes already suggested by the first clip:
   - correct the late start and finish
   - detect the breakout and drop peaks before it
   - allow odd stroke counts
5. Add a confidence score, and leave out results that score low instead of
   showing them. Possible signals: whether the forward and reverse SAM 2
   tracks agree, how well the mask and YOLO box agree, how periodic the
   stroke signal is, and how many re-seeds there were.
6. Tracking upgrades to test with the harness: SAM 3 (text prompt
   "swimmer"), SAMURAI, forward and reverse tracking, a fine-tuned small
   detector, and for race views, cropping to the lane using lane-line
   detection.
7. Still open: where the shipped app will run. A cloud GPU would remove the
   8 GB limit.
