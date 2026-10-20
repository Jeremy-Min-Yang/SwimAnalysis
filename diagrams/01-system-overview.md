# 1. System overview

The page (`src/app.py`) never runs a model itself. It starts
`src/analyze.py` in its own process, reads its `PROGRESS` lines for the
progress bar, and when it ends reads the JSON files it wrote.

```mermaid
flowchart TD
    user(["User drops a video<br/>and picks the pool length"])

    subgraph app["src/app.py: Streamlit page"]
        save["save_upload<br/>file named by content hash"]
        cached{"outputs/upload_hash/<br/>already analyzed?"}
        run["run_analysis<br/>starts analyze.py, reads PROGRESS lines"]
        results["results<br/>race_stats + suggestions → cards, charts,<br/>overlay video, Adjust panel"]
    end

    subgraph analyze["src/analyze.py: separate process"]
        detect["1. detect<br/>YOLO-World 'swimmer' boxes,<br/>6 times a second"]
        path["2. swimmer_path<br/>chain boxes into one swimmer"]
        track["3. track_sam2.track<br/>SAM 2 mask on every frame,<br/>re-seeded from the path"]
        times["4. find_times<br/>start, walls, finish,<br/>one arm or full cycle per peak"]
    end

    subgraph files["outputs/upload_hash/"]
        trackjson[("sam2_track.json<br/>per frame: box, area,<br/>centroid, score, source")]
        trackmp4[("sam2_track.mp4<br/>mask drawn on video")]
        analysisjson[("analysis.json<br/>start, touches, mode,<br/>strokes_per_peak")]
    end

    user --> save --> cached
    cached -- no --> run --> detect
    cached -- yes --> results
    detect --> path --> track --> times
    track --> trackjson & trackmp4
    times --> analysisjson
    trackjson & analysisjson --> results
    trackmp4 --> results
```

## Who does what

| file | job | uses a neural network? |
|---|---|---|
| `app.py` | page, upload, cache, results | no |
| `analyze.py` | runs the four steps | calls the two below |
| YOLO-World (in `analyze.detect`) | "where is a swimmer?" → boxes | yes |
| `track_sam2.py` | "which pixels are that swimmer?" → mask per frame | yes, SAM 2 |
| `timing.py` | start, walls, finish from motion | no: optical flow + signal rules |
| `start_beep.py` | start beep in the audio | no |
| `race_stats.py` | strokes, rate, DPS, splits from the mask shape | no: signal processing |
| `suggestions.py` | advice text from the stats | no |
| `web_video.py` | re-encode for the browser | no |

The two networks only answer *where* the swimmer is. Every race number is
computed afterwards from the mask's box, area and centroid, with plain
NumPy/OpenCV.

## Why the detector *and* the segmenter

- **YOLO-World alone** gives a box every few frames, often misses a swimmer
  under water or in splash, and sometimes boxes another lane. A box is too
  coarse to see an arm lift out of the water.
- **SAM 2 alone** gives a precise outline on every frame, but must be told
  once where the swimmer is, and drifts onto splash or a lane rope.

So YOLO-World finds and re-finds the swimmer (seeds), and SAM 2 does the
frame-by-frame outline. See [2](02-yolo-world.md) and [3](03-sam2-tracking.md).

## Progress bar split

```mermaid
flowchart LR
    a["0 – 0.10<br/>detect"] --> b["0.10 – 0.85<br/>SAM 2 tracking<br/>(the slow part, ~3 frames/s on M2)"] --> c["0.85 – 1.0<br/>timing + strokes"]
```
