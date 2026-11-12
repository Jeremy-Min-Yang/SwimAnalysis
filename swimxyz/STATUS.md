# SwimXYZ class project: status

Master file for picking up this project in a new session. **Read this first, then update it after any change**
(decisions, findings, new files, next steps, plus a line in the change log). The section "How we got here" is
material for the report: keep it, and add to it when the plan changes again.

## The project

- Korean university course final project. Due **about 2026-12-09**. Author name for documents: **Jeong Min Yang**
  (not "Jeremy", which is only the GitHub handle).
- Requirements: my own prepared dataset (open datasets allowed), my own model, the training process and results.
- Deliverables: PDF slides (at most 10, for a strict 5-minute talk), a report (at most 10 pages), all model code.
- Expect the Q&A question "why this model?": every design choice needs a reason. This file records them.
- Kept separate from **Swimline**, the Streamlit app in the same repo (`src/`, see `docs/handoff.md`). Swimline uses
  SAM 2 tracking and no pose estimation. Its weak point is stroke counting, which halves or doubles. At the end, both
  are compared on real clips using `data/labels.csv` (hand labels for swimvid3–56: start/breakout/finish frames and
  total strokes).

## End goal

Plug in **real race footage** → the user **types the lane number and clicks the swimmer once** → that swimmer's
analysis. **No improvement tips or coaching advice** (user decision, 2026-11-12).

Outputs: stroke count, stroke rate (per length), left/right stroke timing and rhythm consistency, and with
Swimline-style speed: DPS and stroke index (SI = speed × DPS).

## Current plan: pose-free stroke-signal model (pivot, 2026-11-12)

Pipeline: race video → lane + click → track the swimmer (reuse Swimline's SAM 2 track) → crop the swimmer every frame
→ **our model** reads a short window of crops and predicts a smooth **per-arm stroke signal** per frame (peaks = hand
entries) → peak picking → metrics.

| Decision | Reason |
|---|---|
| Model input = **cropped video frames of the tracked swimmer**, not joints | Off-the-shelf pose fails on race footage (see "How we got here"). A model that learns the swimmer's appearance, splash included, doesn't need visible arms |
| Output = smooth 1D signal per arm, peaks at hand entries (Victor et al. 2017 style), not hard per-frame classes | Victor et al. found softened targets work better than hard labels. Peaks give count, rate and timing directly |
| Own model from scratch: small per-frame CNN encoder → dilated temporal conv stack → 2 output signals (L/R) | Small enough for free Colab. The temporal stack sees about 2 stroke cycles, so it can rely on rhythm when single frames are hidden by splash |
| Training data = **SwimXYZ freestyle video frames** (partial download), crop boxes from `body25/2D_cam`, targets from the video's own 3D joints using the phase rules already prototyped | Synthetic frames come with exact labels at no labelling cost |
| Strong augmentation: blur, heavy compression, low resolution, random occlusion and cutout | Targets exactly the real problems: poor video quality, splash, other swimmers in the way |
| Real labels = **key taps at each hand entry** on the race clips (minutes per clip), plus `labels.csv` stroke totals | Much cheaper than marking joints. Used for fine-tuning (if needed) and testing |
| Freestyle only | All 3 race clips are 50 m freestyle. The Backstroke video labels are unusable |
| Speed: geometric from the track, not learned | SwimXYZ has no real speed (see findings) |
| Preprocessing runs **locally**; Colab only for training | User preference |

Real race test clips (50 m freestyle, in `swimxyz/`, gitignored as `*.mp4`; lane in the file name):
- `1.50FreeLane1.mp4`: 1280×720, 29.97 fps, 40 s. Elevated side view from the deck; swimmers fairly large.
- `2.50FreeLane7.mp4`: 1080×1920 portrait, 35 s. High up, heavy motion blur, small swimmers. The hardest clip: keep it
  as a stress test.
- `3.50FreeLane7.mp4`: 1280×720, 47 s. Includes the start on the blocks; mid-race is very splashy.

## Scope: race vs technique analysis (discussed 2026-11-12)

- **Technique (form) analysis** needs clear, close footage of one swimmer, and then pose estimation works. That's
  a different product from **race analysis** (far, splashy, many swimmers). No single model does both well.
- The user wants a **finished project with results**; race analysis is the more interesting one.
- Proposal (awaiting the user's choice): one pose-free stroke model, evaluated on two real tiers:
  - **Easier tier:** swimvid3–37, 35 single-swimmer side-view freestyle clips with hand-counted strokes in `data/labels.csv`.
  - **Harder tier:** race footage, swimvid40–56 (17 clips from the stands) plus the 3 new 50 m free clips.

  This guarantees results on the easier tier, keeps race as the headline, and turns the race failure cases into the
  limitations section. Technique analysis (joints → phases, IdC) becomes documented future work.
- **The `swimvid*.mp4` files are no longer in the repo root.** Find out where they are.

## How we got here (for the report)

1. **Dataset exploration (2026-11-09).** SwimXYZ labels have problems not mentioned in the paper: the last 2 frames of
   every clip are broken, the swimmer turns around mid-clip, the Front view gets within 0.4 m of the camera, and all
   2,880 Backstroke clips replay **one** motion. Forward speed is synthetic: a constant 2.5 m/s in every SMPL motion,
   ~2 m/s in the videos. So **SwimXYZ cannot teach speed**, and speed has to be measured geometrically.
2. **First plan: joints → phases.** Train a per-arm phase-segmentation model (Chollet et al. 2000's 4 phases:
   entry+catch / pull / push / recovery, plus IdC) on 2D joints from the 240 SMPL motions, split by motion id. A real
   pose detector would supply the joints at test time.
   - Wrote numpy SMPL forward kinematics (`smpl_fk.py`) and **verified it against the official `smplx` code
     (0.001 mm)**.
   - The 60 SMPL motions per stroke are genuinely different (nearest neighbour about 0.08 rad; between strokes 0.29 rad).
   - Defined phase boundaries from the motion itself (most forward / most backward hand point, shoulder plane), so
     IdC doesn't depend on the unknown water surface. Freestyle prototype: ~1.0 s cycle, IdC −28 to −20%.
3. **Video label checks.** The **Backstroke** video labels show breaststroke-like arms (both wrists move together,
   L/R correlation 0.98) and match none of the 240 SMPL motions. The **Freestyle** video labels are real alternating
   freestyle with several motions, none of them in the SMPL set.
4. **Off-the-shelf pose test on the real race clips (YOLO11m-pose).** On full frames of clip 1 it finds 2–5 of about
   6 swimmers. Mid-race clip 3: none. Clip 2: almost none. Cropping and 4× upscaling don't fix the **arms**: they're
   missed or folded along the body, and the phase model depends on them. Root causes: video quality and compression,
   splash hiding the limbs, other swimmers in the way. Even a human finds the arms hard to see.
5. **Literature check.** Accurate systems avoid this problem rather than solve it:
   - **Controlled footage** (Einfalt, Zecha, Lienhart, Augsburg): a swimming channel, so the swimmer stays in place
     in front of fixed high-quality cameras.
   - **Pose-free stroke detection**: Victor et al. (CVPRW 2017) map a window of frames to a smooth stroke signal;
     Woinoski & Bajić (2021) estimate stroke rate from overhead race video, lane by lane.
   - Most GitHub demos use easy, close-up, single-swimmer footage.
6. **Pivot (2026-11-12): pose-free stroke-signal model** (the current plan above). The pose test becomes evidence in
   the report for why the design changed. The SMPL and forward-kinematics work is parked, not deleted.

## Data findings

- Zenodo record 8399376: `{Backstroke,Breaststroke,Butterfly,Freestyle}_labels.zip` (about 1.6 GB each) and
  `smpl_swimming_motions.zip` (60 MB). Licence CC BY 4.0.
  Citation: Fiche et al., SwimXYZ, ACM SIGGRAPH MIG 2023, doi:10.1145/3623264.3624440.
- **Videos**: Zenodo records 8402009 (Freestyle part 1, 40 GB) and 8402031 (Freestyle part 2, 36 GB); also backstroke
  8399837/8401680, breaststroke 8401898/8401923, butterfly 8401954/8401974. Zenodo serves HTTP range requests
  (tested: 206), so single clips can be pulled out of a zip without downloading it all.
- Labels: 2,880 clip folders per stroke = 5 views × 576 parameter groups. Path:
  `Stroke/View/Swimmer_*/Water_*/Lighting_*/Speed_{2,3}/position_*/{COCO,body25,base}/{2D,3D}_{cam,pelvis}.txt`.
  Files are semicolon-separated with decimal commas: `pd.read_csv(p, sep=";", decimal=",")`; drop the trailing `Unnamed` column.
- `COCO/` is 28% NaN. Use `body25/`. `2D_cam` is in pixels (z = depth); `3D_cam` is in metres.
- **The last 2 rows (300, 301) of every clip are broken reset frames.** Drop them (`load_clean` in notebook 01 does).
  Some clips are one frame shorter (299 usable frames).
- The swimmer **turns around** once per clip (in `3D_cam`, at a different frame per view). This explains the
  `2D_pelvis` flip and the 3D "jumps". Front view gets within 0.4 m of the camera, where the 2D projection blows up: drop it.
- **Video files**: `.webm`, 1920×1080, 60 fps, 301 frames, stored under the same folder path as the labels
  (`<View>/Swimmer_*/.../position_*.webm`). Freestyle part 1 (8402009) = Aerial + Front (1,152 videos, ~36 MB each);
  part 2 (8402031) = the 3 side views (1,728 videos; Side_above_water ~36 MB, underwater/water-level ~15 MB).
  Read with `remotezip` (range requests): no full download needed.
- **Label ↔ video alignment: `2D_cam` y is measured from the bottom (Unity).** Use `y_img = 1080 − y`. Then the joints sit
  on the rendered swimmer in every view checked (Aerial, 3 side views), and label row i = video frame i.
- The swimmer is **not always in view**: Aerial frames 0–154 only; Side_above_water 0–248; Side_underwater 0–217;
  Side_water_level 111 frames, split around the turn. Use only frames where all joints are inside the image.
- Swimmer size in the side views: about 310–440 × 75–110 px (bounding box of the joints).
- The renders **do show splash** at the kick and at hand entries.
- Backstroke video labels: one motion for all 2,880 clips (1 outlier), with breaststroke-like arms. Unused.
- Freestyle video labels: arms alternate (L/R correlation about −0.7), hands reach 0.55 m behind the shoulder; several
  motions (6.4° median joint-angle difference between clips); none matches an SMPL motion (26–28° away).
- SMPL motions: AMASS-style `.npz` with `poses` (360×156, SMPL-H axis-angle), `trans` (360×3, constant 2.5 m/s along y),
  `betas` (16), `gender` (all male), `mocap_framerate` 30. Each motion is 12 s; cycle about 0.85–1.07 s.
- Motion frame: the body's long axis is +y and the chest faces +z in every stroke. "Up" is inferred from the recovery
  motion: freestyle and butterfly face down, backstroke faces up.
- Backstroke prototype IdC was +14 to +20%, which is suspicious (published values are usually negative). Not
  investigated further: backstroke is out of scope now.

## Files

- `notebooks/01_explore_labels.ipynb`: data exploration, Parts 1–5. Colab-oriented (Drive mount in the first cells).
- `smpl_fk.py` (parked): numpy SMPL forward kinematics (motion `.npz` → 24 joints), no torch/chumpy. Self-check:
  `.venv/bin/python smpl_fk.py data/smpl_model/basicmodel_m_lbs_10_207_0_v1.1.0.pkl data/smpl/freestyle/freestyle_1_poses.npz`.
  Shortcut: uses the first 10 of the 16 SMPL-H betas with the SMPL model.
- `check_fk_reference.py` (parked): one-time check of `smpl_fk.py` against official `smplx` (needs torch, so it runs
  in a throwaway venv; instructions inside the file).
- `build_crops.py` (Windows-safe, writes `*.part` then renames; skips clips with bad labels instead of stopping): SwimXYZ freestyle videos (pulled with `remotezip`, deleted after use) → 64×128 swimmer crops at
  30 fps, `valid` flags (≥80% of joints in view), per-arm stroke events in seconds (hand furthest forward along the body,
  from `3D_cam`). Writes `data/crops/<View>/.../*.npz`. Restartable. Run as `build_crops.py <n_per_view> <View>`.
- `real_crops.py`: same crops from real clips, using Swimline's `outputs/<clip>/sam2_track.json` boxes (gaps filled by
  interpolation), plus the swim window (breakout → finish) and stroke total from `data/labels.csv`. Writes `data/real/<clip>.npz`.
- `train.py`: StrokeNet (from scratch, ~207k parameters): per-frame CNN (4 strided convs) → 4 residual dilated
  temporal convs (1, 2, 4, 8) → 3 signals per frame (left, right, any hand). Gaussian targets (σ = 2 frames), BCE with
  positive weight 4. Augmentation: time-stretch 0.9–1.9× (synthetic strokes are slower than real racing), zoom/shift,
  low resolution, blur, JPEG quality 8–50, colour shift, cutout, mirror (swaps L/R). Counts peaks of "any" (height
  0.3, ≥0.25 s apart) in the real swim window. Saves `data/stroke_net.pt` and `data/real_results.json`.
- `requirements.txt`, `.venv/`: local Python 3.13 environment, separate from Swimline's.
- `data/` (gitignored): `raw/` (Backstroke + Freestyle label zips, SMPL zip), `smpl/` (extracted motions),
  `smpl_model/` (SMPL v1.1.0 m/f/neutral `.pkl`, licensed: never commit or share), `weights/` (`yolo11m-pose.pt`, used
  for the pose test), `figures/pose_test/` (pose-test images for the report).
- `1.50FreeLane1.mp4`, `2.50FreeLane7.mp4`, `3.50FreeLane7.mp4`: real race test clips.
- `README.md`: public overview (updated 2026-11-12 for the pose-free plan and the two test tiers).

## Next steps

1. **User builds the training set at home (Windows 11 PC, faster internet).** Steps are in `README.md` → "Running it".
   Local Mac downloads were stopped at **28 built videos** (Side_underwater 9, Side_water_level 10, Aerial 5,
   Side_above_water 4), at ~0.5 videos/min; the home run continues from those if `data/crops/` is copied over, or starts
   fresh. Targets: Side_water_level 120, Side_above_water 100, Side_underwater 80, Aerial 60 (more is fine).
   Afterwards copy `data/crops/` back to the Mac (or upload to Drive for Colab).
2. Train for real (a few thousand steps; locally on MPS or on Colab) and read the real-clip stroke counts. Tune the
   peak threshold on **synthetic validation**, not on the real clips (they are the test set).
3. Compare with Swimline on the same clips. Swimline counts (from `outputs/<clip>/race_stats.json`): swimvid3 12
   (true 13), 4 34 (34), 5 14 (15), 7 10 (11), 13 21 (23); none saved for 11 and 14. **Swimline is already close on these easy
   clips**, possibly tuned on them, so the harder tier is where a difference should show.
4. Make Swimline tracks for swimvid6, 8, 9, 10, 12, and later 15–56 (on the other computer) and the 3 race clips.
5. Then the report and slides.

## Open questions

- Baseline(s): Swimline's non-learned stroke counting (numbers already saved), and/or StrokeNet without the temporal convs.
- Split for synthetic validation: currently random videos; better by motion (the freestyle videos hold several motions) or by swimmer.
- Does fine-tuning a pretrained model count as "my own model"? The user is emailing the professor. It doesn't block
  anything: we build from scratch.
- Do the rendered SwimXYZ frames show splash at all? If not, the cutout augmentation has to stand in for it.
- Can the Woinoski & Bajić race-video dataset be reused (licence, view type)? It could add real training data.
- Presentation language (Korean or English).

## References

- Fiche et al., SwimXYZ (ACM SIGGRAPH MIG 2023), doi:10.1145/3623264.3624440.
- Chollet, Chalies, Chatard (2000), Index of Coordination in front crawl.
- Victor, He, Morgan, Miniutti, "Continuous Video to Simple Signals for Swimming Stroke Detection with CNNs", CVPRW 2017.
- Woinoski & Bajić, "Swimmer Stroke Rate Estimation From Overhead Race Video", ICME workshops 2021 (arXiv:2104.12056);
  "Towards Automated Swimming Analytics Using Deep Neural Networks", arXiv:2001.04433.
- Einfalt, Zecha, Lienhart, "Activity-conditioned continuous human pose estimation ... swimming", WACV 2018.
- Lienhart, Einfalt, Zecha, "Mining Automatically Estimated Poses from Video Recordings of Top Athletes", IJCSS 2018.

## Change log

- 2026-11-09: notebook 01 Parts 3–5 added. Findings: one motion per stroke in the videos, no real speed, 60 distinct SMPL motions.
- 2026-11-12: moved preprocessing local (`swimxyz/.venv`, `swimxyz/data/`). Added `smpl_fk.py`. Up direction and
  kinematic phase prototype done. Chose Chollet's 4 phases. Created this file.
- 2026-11-12: plan = random forest baseline + from-scratch network. MotionBERT fine-tuning optional, for comparison, if time allows.
- 2026-11-12: review fixes in notebook 01 (no more saving wrong speeds to CSV, drop 2 reset frames, no `allow_pickle`); removed `nbclient`.
- 2026-11-12: FK verified against smplx (0.001 mm). Backstroke video labels show breaststroke-like symmetric arms and match no SMPL motion.
- 2026-11-12: looked into training on the videos: range download works.
- 2026-11-12: Freestyle video labels checked: real alternating freestyle, several motions, none in the SMPL set. User PC: Windows 11 + RX 9060 XT.
- 2026-11-12: end goal set: race footage → one swimmer → analysis. No improvement tips. Selection = lane number + one click. 3 race clips added.
- 2026-11-12: off-the-shelf YOLO11-pose fails on swimmers' arms in the race clips. Literature check.
  **Pivot to a pose-free stroke-signal model** trained on SwimXYZ freestyle video frames. Joint/SMPL plan parked.
  Restructured this file and added "How we got here" for the report.
- 2026-11-12: README rewritten for the pose-free plan. Race vs technique scope discussed; proposal: evaluate on two real tiers (swimvid3–37 easier, race clips harder). Awaiting user's choice.
- 2026-11-12: user chose to test on swimvid3–14 first (15–56 are on another computer). Swimline SAM 2 tracks exist for swimvid3, 4, 5, 7, 11, 13, 14. SwimXYZ video layout found; label y must be flipped; alignment verified on 4 views; renders show splash. Added `remotezip` and `opencv-python-headless` to requirements.
- 2026-11-12: pipeline built end to end: `build_crops.py` (4 background jobs running), `real_crops.py` (7 real clips with Swimline tracks: 3, 4, 5, 7, 11, 13, 14), `train.py` (StrokeNet, 207k params). Smoke test on 14 videos / 250 steps: runs in 1 min, over-counts (77% error); swimvid3 exact. Added torch to requirements.
- 2026-11-12: stopped the local download (28 videos built); the user will run it at home. `build_crops.py` made Windows-safe (temp file closed before reading), atomic saves, skips bad clips. README: "Running it" instructions. Deleted cached sample videos.
