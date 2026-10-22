# Swimline: how it works

The [overview diagram](overview.md) shows the pieces. This explains each
one.

## The big idea

Two pretrained models answer **where is the swimmer?** Everything else, the
race numbers included, is plain NumPy and OpenCV working from that answer.
Nothing is trained in this project. We use the models as they come and add
rules around them.

| piece | type | job |
|---|---|---|
| YOLO-World | pretrained model (`yolov8s-worldv2.pt`) | finds swimmer boxes every few frames |
| SAM 2 | pretrained model (`sam2.1_t.pt`) | outlines the swimmer's pixels on every frame |
| `analyze.py` | our code | runs the steps in order |
| `track_sam2.py` | our code | drives SAM 2 and rejects bad masks |
| `timing.py`, `start_beep.py` | our code | start, walls and finish |
| `race_stats.py` | our code | strokes, rate, distance per stroke, splits |
| `app.py` | our code | upload page and results |

Both models are loaded through the Ultralytics library, which downloads the
weights the first time.

## 1. The app (`app.py`)

The Streamlit page saves the uploaded video under a hash of its contents,
so the same video opens instantly the next time. It starts `analyze.py` as
a separate process and turns that process's `PROGRESS` lines into the
progress bar. When the process finishes, the page reads the JSON files it
wrote and draws the results.

The model work runs in its own process to keep the page responsive and
memory use low, which matters on an 8 GB Mac.

## 2. Finding the swimmer: YOLO-World

A normal YOLO detector only knows the fixed classes it was trained on
(COCO's 80), and "swimmer" isn't one of them. YOLO-World is
**open-vocabulary**: you give it class names as text.

- A CLIP text encoder turns "swimmer" and "person swimming" into vectors.
  This happens once (`set_classes`).
- The YOLOv8 backbone extracts image features. A fusion neck (RepVL-PAN)
  mixes the text vectors into them.
- For each candidate box, the head outputs a box and a vector. The box's
  confidence is how similar its vector is to the text vector.

So the class is not a fixed output. It is a match against the meaning of
the word, which explains why confidences are low (0.1 to 0.4 is normal) and
why it sometimes fires on people on the deck.

Our code around it (`analyze.py`):

- It runs about 6 times a second, not on every frame, because that is
  enough to locate the swimmer.
- Boxes taller than they are wide are dropped (`MAX_TALL`): a swimmer lies
  flat.
- `swimmer_path` picks one swimmer for the whole video. It starts from the
  strongest box near the centre of the picture (the swimmer the camera
  follows), then walks forward and backward in time, taking the nearest box
  within reach at each step. Vertical reach is kept small because the
  camera pans along the pool, so a big jump up or down means another lane.

## 3. Following the swimmer: SAM 2

SAM (Segment Anything) turns a prompt, such as a box, into a pixel mask of
the object inside it. SAM 2 adds **memory** for video:

- An image encoder (Hiera, tiny version) turns each frame into features.
- Memory attention compares the current frame with a memory bank of
  earlier frames and their masks.
- A mask decoder outlines the object and gives a score for whether it is
  present.
- Each new mask can be encoded and added to the memory.

After one prompt, SAM 2 keeps finding "the same thing" in later frames.

Out of the box, SAM 2 drifts onto splash, lane ropes and other swimmers.
`track_sam2.py` wraps it with three controls:

1. **Mask checks** (`plausible`). A mask is rejected if it is empty or tiny,
   if it is a thin line (a lane rope), if it grows or shrinks more than 3×,
   or if it jumps more than one box size from the last mask. A rejected
   frame counts as "lost".
2. **Controlled memory.** Only accepted masks go into memory, and memory
   holds the seed plus the latest 6 accepted masks. After 5 lost frames,
   memory is reset to the seed alone so SAM 2 searches for the swimmer as
   first marked.
3. **Re-seeding from YOLO** (`make_reseeder` in `analyze.py`). If SAM 2 is
   lost for 10 frames, or its mask sits outside YOLO's box 3 checks in a
   row, it is restarted from the YOLO path box.

Put simply, YOLO is the coarse "where", SAM 2 is the precise "which
pixels", and each covers the other's weakness.

The output, `sam2_track.json`, holds the mask's box, area and centroid for
each frame. The masks themselves are not kept.

## 4. Times (`timing.py`, `start_beep.py`)

The camera pans with the swimmer, so their position in the picture barely
changes. Their real motion is:

> swimmer motion = centroid change − camera motion

Camera motion is the median optical flow (OpenCV Farneback) of everything
outside the swimmer's box. The main direction of the remaining motion gives
a signed speed along the pool.

- **Start**: the first steady motion. If the audio has a start beep up to
  2 s earlier, the beep is used instead.
- **Wall**: the speed's sign flips and stays flipped (a turn).
- **Finish**: the last stroke plus the reach to the wall, or the moment
  motion stops minus 1.5 s of drift.
- If there's no clear start or stop, the clip is treated as part of a swim
  and gets stroke rhythm only.

Pixels are never converted to metres. Only times come from the video;
speed is the pool length divided by the split.

## 5. Strokes (`race_stats.py`)

A recovering arm lifts out of the water and stretches the mask upward. Per
frame:

> signal = (centroid y − box top) ÷ √area

This rises once per arm that clears the outline.

1. Autocorrelation over the race finds the typical cycle time (0.6 to
   3.5 s). If the signal isn't periodic enough, no stroke numbers are given.
2. Detrending and smoothing remove zoom changes and splash.
3. Peaks are kept only where the signal repeats one cycle later, which
   rules out glides and push-offs. Peaks one or two cycles from a kept peak
   are added too, which covers a cycle hidden by splash.
4. Peaks per length give strokes, stroke rate and distance per stroke.

Whether one peak is one arm or a full cycle (two strokes) depends on the
camera. `strokes_per_peak` guesses from how alike successive peaks are, and
the page's "Adjust" panel can override it.

## Where it is weak

- Race views with many lanes: SAM 2 settles on splash or a lane rope.
- Finish time: up to about 2 s off.
- The arm-or-cycle guess is tuned on only six clips.
- Stroke counts are usually within about 2 of a hand count.
