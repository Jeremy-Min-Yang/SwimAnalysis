# 4. Times and strokes

No neural network from here on. Input is `sam2_track.json` (a box, area
and centroid per frame) plus the video's pixels and audio.

## Times: start, walls, finish

Code: `src/timing.py`, `find_times` in `src/analyze.py`, `src/start_beep.py`.

The camera pans with the swimmer, so the swimmer barely moves in the
picture. Their real motion = motion in the picture − camera motion.

```mermaid
flowchart TD
    cent[("mask centroid per frame")] --> pic["picture motion<br/>centroid change per frame"]
    video[("video frames")] --> flow["camera_flow<br/>Farneback optical flow at 320 px wide,<br/>median outside the swimmer's box"]
    pic --> own["own motion = picture − camera"]
    flow --> own
    own --> axis["main direction = first principal axis<br/>(SVD) of the strong motion"]
    axis --> speed["signed speed along the pool,<br/>smoothed over SMOOTH_S (1 s)"]

    speed --> moving["moving = |speed| > 30% of typical<br/>runs ≥ 1 s, pauses ≤ 0.7 s bridged"]
    moving --> start["start = first moving run"]
    moving --> turn["wall = sign flips and holds<br/>TURN_HOLD_S (2 s) each side"]
    moving --> stop["stopped = end of last run,<br/>if ≥ 0.5 s before video end"]

    stop --> whole{"has a start > 0.5 s<br/>and a stop?"}
    whole -- no --> window["mode = window<br/>part of a swim: rhythm only"]
    whole -- yes --> beep["start_beep.py: beep ≤ 2 s<br/>before the motion start? use it"]
    beep --> finish{"strokes seen until<br/>≤ 4 s before the stop?"}
    finish -- yes --> f1["finish = last stroke peak<br/>+ 0.75 cycle (the reach)"]
    finish -- no --> f2["finish = stop − 1.5 s<br/>(drift after the touch)"]
    f1 & f2 --> lengths["mode = lengths<br/>start, touches"]
```

Pixels are never turned into meters: the camera zooms and sees the pool at
an angle. Only the *times* come from the video; speed = pool length ÷ split.

## Strokes: from the mask's shape

Code: `src/race_stats.py`, `strokes_per_peak` in `src/analyze.py`.

A recovering arm lifts out of the water and stretches the mask upward. So
per frame:

> signal = (centroid y − box top) ÷ √area

rises once per arm that clears the outline.

```mermaid
flowchart TD
    m[("box, area, centroid per frame")] --> sig["mask_shape_signal<br/>how far the mask reaches above its center,<br/>scaled by size"]
    sig --> fill["fill_gaps<br/>interpolate lost frames"]
    fill --> ac["autocorrelation over the race<br/>best lag in 0.6 – 3.5 s<br/>= typical cycle (peak_s)"]
    ac --> ok{"correlation there<br/>≥ MIN_PERIODICITY (0.3)?"}
    ok -- no --> none["no stroke numbers<br/>(null, not a guess)"]
    ok -- yes --> clean["remove slow trend (1 cycle average),<br/>smooth splash (0.15 cycle), standardize"]
    clean --> peaks["find_peaks<br/>≥ 0.6 cycle apart, prominent"]
    clean --> local["local_periodicity<br/>does the window repeat one cycle later?<br/>(glide, push-off ≈ 0)"]
    peaks --> filt["drop peaks next to lost frames<br/>or within 0.5 cycle before a touch"]
    filt --> chain["keep peaks where rhythm is strong,<br/>extend_chain: add peaks 1–2 cycles<br/>from a kept one, same length"]
    local --> chain
    chain --> per["per length: peaks → cycles,<br/>counting a hidden cycle from the span"]
    per --> arm{"strokes_per_peak<br/>one arm (1) or full cycle (2)?"}
    arm --> out["strokes, stroke rate, DPS,<br/>first stroke after start/turn, splits"]
```

### One arm or a full cycle per peak?

Usually only the near arm clears the outline: one peak = one arm cycle =
two strokes. Filmed close and side-on, both arms do: one peak = one stroke.
`strokes_per_peak` guesses which:

- successive peaks of similar height (`ARM_HEIGHT_STEP`), and
- the signal repeats as well two peaks on as one peak on (`ARM_REPEAT`)

→ both arms seen, 1 stroke per peak; otherwise 2. Set from six clips; the
"Adjust" panel lets the user override it.

## Whole pipeline in one line

```mermaid
flowchart LR
    v["video"] --> y["YOLO-World<br/>boxes"] --> p["one swimmer path"] --> s["SAM 2<br/>mask per frame"]
    s --> b["box, area, centroid"]
    b --> t["motion − camera<br/>→ start, walls, finish"]
    b --> k["mask shape rhythm<br/>→ stroke peaks"]
    t & k --> r["splits, speed, strokes,<br/>rate, DPS"]
```
