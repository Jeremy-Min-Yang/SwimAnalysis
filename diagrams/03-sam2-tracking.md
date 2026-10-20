# 3. Following the swimmer: SAM 2

Code: `src/track_sam2.py` (`Sam2Follower`, `plausible`, `track`) and
`make_reseeder` in `src/analyze.py`. Weights: `sam2.1_t.pt` (tiny),
run at `imgsz` 640 through Ultralytics' `SAM2DynamicInteractivePredictor`.

## What SAM 2 is

SAM (Segment Anything) turns a prompt (a point or box) into a pixel mask of
"the thing there". SAM 2 adds **memory** so it can keep segmenting the same
thing in later video frames with no new prompt.

```mermaid
flowchart LR
    frame["frame t"] --> enc["image encoder<br/>(Hiera-tiny)<br/>frame → feature map"]
    enc --> attn["memory attention<br/>current features look at<br/>the memory bank"]
    bank[("memory bank<br/>features + masks of<br/>earlier frames")] -.-> attn
    prompt["prompt: box or point<br/>(only on seed frames)"] --> penc["prompt encoder"]
    attn --> dec["mask decoder"]
    penc -.-> dec
    dec --> mask["mask for frame t<br/>+ score: is the object there?"]
    mask --> menc["memory encoder<br/>mask + features → memory"]
    menc -. "if accepted" .-> bank
```

- **Seed frame**: the prompt tells the decoder what to segment; the result
  goes into memory.
- **Later frames**: no prompt. Memory attention finds the region of the new
  frame that looks like what memory holds, and the decoder outlines it.

In the stock SAM 2 the memory is updated automatically every frame. Here the
code **controls** it (`Sam2Follower`): only masks that pass the checks go
in, and it keeps the seed plus the latest accepted masks, `MEMORY_FRAMES`
(7) in all, adding one every `MEMORY_EVERY` (2) accepted frames. A bad mask
never enters memory, so it cannot teach SAM 2 to follow splash.

## Per-frame loop

```mermaid
flowchart TD
    start(["next frame"]) --> seedq{"seed for this frame?<br/>(first path box, or reseed() says so)"}

    seedq -- yes --> seed["follower.seed(box)<br/>clear memory, prompt with the box"]
    seed --> accept

    seedq -- no --> lostlong{"lost exactly<br/>LOST_RESET_FRAMES + 1 frames?"}
    lostlong -- yes --> forget["follower.forget()<br/>memory = seed only"]
    lostlong -- no --> predict
    forget --> predict["follower.predict(frame)<br/>mask + score"]

    predict --> checks{"plausible?<br/>• score > 0<br/>• area ≥ MIN_AREA_FRACTION of frame<br/>• not a thin line (aspect ≤ MAX_ASPECT): lane rope<br/>• area within ×3 of last mask and of 2 s median<br/>• box within one box size of last box"}
    checks -- no --> lost["record = None<br/>overlay says LOST"]
    checks -- yes --> accept["accept: save box, area,<br/>centroid, score, source"]
    accept --> mem{"2nd accepted<br/>since last remember?"}
    mem -- yes --> remember["follower.remember(mask)<br/>add to memory, trim to 7"]
    mem -- no --> write
    remember --> write
    lost --> write["draw overlay, append frame record"]
    write --> start
```

## When to re-seed: `make_reseeder`

Called on every frame after the first seed. Returns a path box (from
[2](02-yolo-world.md)) or nothing.

```mermaid
flowchart TD
    q["frame has a path box<br/>with conf ≥ SEED_CONFIDENCE?"] -- no --> none["no re-seed"]
    q -- yes --> lost{"SAM 2 lost the swimmer for<br/>≥ LOST_FRAMES (10) frames?"}
    lost -- yes --> reseed["re-seed with the path box"]
    lost -- no --> inside{"mask centroid inside path box<br/>(grown by BOX_MARGIN 25%)?"}
    inside -- yes --> reset["outside count = 0"] --> none
    inside -- no --> count["outside count + 1"]
    count --> three{"outside on OUTSIDE_CHECKS (3)<br/>path frames in a row?"}
    three -- yes --> reseed
    three -- no --> none
```

Two ways SAM 2 goes wrong, two fixes:

| failure | seen as | fix |
|---|---|---|
| loses the swimmer (dive, splash, turn) | no accepted mask for a while | forget to seed-only memory after 5 frames; re-seed from YOLO after 10 |
| follows the wrong thing confidently (splash, rope, other lane) | mask keeps passing checks but sits outside YOLO's box | re-seed after 3 path frames outside |

## Output

`sam2_track.json`: per frame `{box, area, centroid, score, source}` or
`null`. `source` is `seed` or `track`. Everything in
[4](04-timing-and-strokes.md) is computed from these numbers alone; the
masks themselves are not saved.
