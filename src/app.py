"""Swim analyzer page: drop a side-view swim video, get the numbers.

Run with:  streamlit run src/app.py

The video is analyzed by analyze.py in its own process (finding and
following the swimmer, then the start, walls and stroke type), which takes a
few minutes. Stats are then computed here, so changing the pool length or
correcting a time under "Adjust" is instant. Results are kept per video
(by content), so dropping the same video again shows them straight away.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import race_stats as rs  # noqa: E402
from suggestions import cycle_times, rhythm_variation, suggestions  # noqa: E402

UPLOADS = ROOT / "uploads"
VIDEO_TYPES = ["mp4", "mov", "m4v"]
LINE_COLOR = "#2a6fdb"
WALL_COLOR = "#9aa0a6"
# Plausible freestyle speeds, m/s; outside them the walls were probably misread.
SPEED_RANGE = (0.4, 2.8)
# Tracker progress maps onto this part of the bar (analyze.py reports the rest).
TRACK_PROGRESS = (0.1, 0.85)


def run_helper(script: str, *args: str) -> str:
    """Run a helper script in its own process. PyAV and OpenCV each bundle
    FFmpeg, and loading both into one process can crash it."""
    result = subprocess.run(
        [sys.executable, str(SRC / script), *args], cwd=ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr else script)
    return result.stdout


def web_video(source: Path) -> Path:
    """An H.264 copy of an OpenCV mp4v video, which browsers cannot play."""
    target = source.with_name(source.stem + "_web.mp4")
    if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
        run_helper("web_video.py", str(source), str(target))
    return target


@st.cache_data(max_entries=32)
def read_frame(path: str, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    success, frame = capture.read()
    capture.release()
    if not success:
        raise ValueError(f"cannot read frame {index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------- analysis


def save_upload(upload) -> Path:
    data = upload.getbuffer()
    digest = hashlib.sha1(data).hexdigest()[:12]
    UPLOADS.mkdir(exist_ok=True)
    path = UPLOADS / f"{digest}{Path(upload.name).suffix.lower()}"
    if not path.exists():
        path.write_bytes(data)
    return path


def run_analysis(video: Path, output_dir: Path) -> bool:
    command = [sys.executable, str(SRC / "analyze.py"), str(video), "-o", str(output_dir)]
    bar = st.progress(0.0, "Starting…")
    log = []
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in process.stdout:
        log.append(line)
        if line.startswith("PROGRESS "):
            _, fraction, message = line.rstrip("\n").split(" ", 2)
            bar.progress(float(fraction), message)
        elif match := re.search(r"(\d+)/(\d+) frames", line):
            done, total = int(match[1]), int(match[2])
            low, high = TRACK_PROGRESS
            bar.progress(low + (high - low) * done / total, f"Following the swimmer: frame {done} of {total}")
    process.wait()
    bar.empty()
    if process.returncode != 0:
        errors = [line for line in log if line.startswith("Error")]
        st.error(errors[-1] if errors else "The analysis failed.")
        with st.expander("Details"):
            st.code("".join(log[-30:]))
        return False
    return True


# -------------------------------------------------------------------- results


def load_settings(output_dir: Path) -> dict:
    analysis = json.loads((output_dir / "analysis.json").read_text())
    adjusted = output_dir / "adjusted.json"
    settings = {key: analysis[key] for key in ("mode", "start", "touches", "strokes_per_peak")}
    if adjusted.exists():
        settings.update(json.loads(adjusted.read_text()))
    return {"analysis": analysis, **settings}


def stroke_signal(track: dict, stats: dict, begin: int, end: int) -> pd.DataFrame:
    """The arm-height signal race_stats counts peaks on, for plotting."""
    fps = track["fps"]
    raw = rs.fill_gaps(rs.mask_shape_signal(track["frames"]))
    cycle = max(3, round(stats["peak_s"] * fps))
    signal = raw - rs.moving_average(raw, max(3, int(rs.DETREND_CYCLES * cycle)))
    signal = rs.moving_average(signal, max(1, int(rs.SMOOTH_CYCLES * cycle)))
    frames = np.arange(begin, min(end + 1, len(signal)))
    return pd.DataFrame({"frame": frames, "time_s": frames / fps, "arm_height": signal[frames]})


def rate_through_swim(stats: dict, fps: float) -> pd.DataFrame:
    rows = []
    for length in stats["lengths"]:
        for a, b, cycle in cycle_times(length["peak_frames"] or []):
            # A gap hiding more than one stroke gives only an average rate.
            if (b - a) / cycle > 2.5:
                continue
            rows.append({
                "time_s": (a + b) / 2 / fps,
                "rate": 60 * stats["strokes_per_peak"] / (cycle / fps),
                "length": length["length"],
            })
    return pd.DataFrame(rows)


def charts(track: dict, stats: dict, whole_lengths: bool) -> None:
    fps = track["fps"]
    lengths = stats["lengths"]
    begin, end = round(lengths[0]["start_s"] * fps), round(lengths[-1]["touch_s"] * fps)
    signal = stroke_signal(track, stats, begin, end)
    peaks = [p for length in lengths for p in (length["peak_frames"] or [])]
    peak_rows = signal[signal["frame"].isin(peaks)]
    walls = pd.DataFrame({"time_s": [length["touch_s"] for length in lengths][:-1] if whole_lengths else []})

    rate = rate_through_swim(stats, fps)
    if len(rate):
        st.subheader("Stroke rate through the swim")
        st.caption("From the time between neighbouring strokes; each wall breaks the line.")
        chart = alt.Chart(rate).encode(
            x=alt.X("time_s:Q", title="Time (s)",
                    scale=alt.Scale(domain=[float(signal.time_s.min()), float(signal.time_s.max())])),
            y=alt.Y("rate:Q", title="Strokes per minute", scale=alt.Scale(zero=False)),
            detail="length:N",
            tooltip=[alt.Tooltip("time_s:Q", title="Time (s)", format=".2f"),
                     alt.Tooltip("rate:Q", title="Strokes/min", format=".0f"),
                     alt.Tooltip("length:N", title="Length")],
        )
        st.altair_chart(
            (chart.mark_line(color=LINE_COLOR, strokeWidth=2)
             + chart.mark_circle(color=LINE_COLOR, size=64)).properties(height=200),
            width="stretch",
        )

    st.subheader("Strokes found")
    st.caption("How far the swimmer's outline reaches above its middle: each rise is an arm "
               "clearing the water, and dots are the counted ones"
               + ("; gray lines are the walls." if len(walls) else "."))
    base = alt.Chart(signal).encode(x=alt.X("time_s:Q", title="Time (s)"))
    layers = [
        base.mark_line(color=LINE_COLOR, strokeWidth=2).encode(
            y=alt.Y("arm_height:Q", title="Arm height", axis=alt.Axis(labels=False, ticks=False)),
            tooltip=[alt.Tooltip("time_s:Q", title="Time (s)", format=".2f")],
        ),
        alt.Chart(peak_rows).mark_circle(color=LINE_COLOR, size=80, stroke="white", strokeWidth=2).encode(
            x="time_s:Q", y="arm_height:Q",
            tooltip=[alt.Tooltip("time_s:Q", title="Stroke at (s)", format=".2f")],
        ),
    ]
    if len(walls):
        layers.append(alt.Chart(walls).mark_rule(color=WALL_COLOR, strokeDash=[4, 4]).encode(x="time_s:Q"))
    st.altair_chart(alt.layer(*layers).properties(height=200), width="stretch")


def peak_pictures(video: Path, track: dict, stats: dict, count: int = 4) -> None:
    peaks = next((length["peak_frames"] for length in stats["lengths"] if length["peak_frames"]), [])
    for column, peak in zip(st.columns(count), peaks[:count]):
        mask = track["frames"][peak]["mask"]
        image = read_frame(str(video), peak)
        if mask:
            x1, y1, x2, y2 = mask["box"]
            # Pad to at least a 2:1 crop so a long, flat swimmer is not a sliver.
            pad_x = (x2 - x1) * 0.15
            pad_y = max((y2 - y1) * 0.5, ((x2 - x1) + 2 * pad_x) / 4 - (y2 - y1) / 2)
            image = image[int(max(0, y1 - pad_y)) : int(min(image.shape[0], y2 + pad_y)),
                          int(max(0, x1 - pad_x)) : int(min(image.shape[1], x2 + pad_x))]
        column.image(image, caption=f"{peak / track['fps']:.2f} s")


def moment_pictures(video: Path, track: dict, settings: dict) -> None:
    """The frames picked as the start, each wall and the finish, to check at a glance."""
    times = [settings["start"]] + settings["touches"]
    names = ["Start"] + [f"Wall {n}" for n in range(1, len(times) - 1)] + ["Finish"]
    st.markdown("**The moments used for the times.** If one is wrong, fix it under Adjust below.")
    per_row = 4
    for row in range(0, len(times), per_row):
        for column, name, time_s in zip(st.columns(per_row), names[row : row + per_row], times[row : row + per_row]):
            frame = min(round(time_s * track["fps"]), len(track["frames"]) - 1)
            image = read_frame(str(video), frame)
            scale = 360 / max(image.shape[:2])
            column.image(cv2.resize(image, None, fx=scale, fy=scale), caption=f"{name}: {time_s:.2f} s")


def adjust(video: Path, output_dir: Path, track: dict, stats: dict, settings: dict) -> None:
    analysis = settings["analysis"]
    with st.expander("Adjust: if a time or the stroke count looks wrong"):
        st.markdown(
            "**Stroke type.** These are the first strokes found. If the **same arm** is up in "
            "each picture, each is a full arm cycle (two strokes); if the **arms alternate**, "
            "each is one stroke."
        )
        peak_pictures(video, track, stats)
        options = {2: "Same arm each time", 1: "Arms alternate"}
        modes = {"lengths": "Start to finish, wall to wall", "window": "Part of a swim (no walls)"}
        with st.form("adjust"):
            choice = st.radio("Pictures show", list(options), format_func=options.get,
                              index=list(options).index(settings["strokes_per_peak"]), horizontal=True)
            mode = st.radio("The video shows", list(modes), format_func=modes.get,
                            index=list(modes).index(settings["mode"]), horizontal=True)
            start = st.number_input("Start (s)", value=float(settings["start"]), step=0.05, format="%.2f")
            touches = st.text_input("Wall touches, then the finish, or the end of a part swim "
                                    "(s, comma separated)",
                                    ", ".join(f"{t:.2f}" for t in settings["touches"]))
            left, right = st.columns(2)
            apply = left.form_submit_button("Apply", type="primary")
            reset = right.form_submit_button("Back to the automatic values")
        if apply:
            try:
                values = sorted(float(v) for v in touches.replace(" ", "").split(",") if v)
            except ValueError:
                st.error("Times must be numbers in seconds.")
                return
            if not values or values[0] <= start:
                st.error("Each wall time must come after the start.")
                return
            (output_dir / "adjusted.json").write_text(json.dumps({
                "mode": mode, "start": start,
                "touches": values if mode == "lengths" else values[-1:],
                "strokes_per_peak": choice,
            }))
            st.rerun()
        if reset:
            (output_dir / "adjusted.json").unlink(missing_ok=True)
            st.rerun()
        source = {"beep": "the start beep", "motion": "when the swimmer starts moving"}.get(
            analysis.get("start_source"), "the first frame the swimmer is followed")
        st.caption(
            f"Found automatically: start from {source}; walls from where the swimmer turns, "
            f"and the finish from the last stroke. The swimmer was followed in "
            f"{analysis['found_frames']} of {analysis['frames']} frames and found again "
            f"{analysis['reseeds']} times after being lost."
        )


def results(video: Path, output_dir: Path, pool: float, unit: str) -> None:
    track = json.loads((output_dir / "sam2_track.json").read_text())
    settings = load_settings(output_dir)
    whole_lengths = settings["mode"] == "lengths"
    try:
        stats = rs.race_stats(track, pool, unit, settings["start"], settings["touches"],
                              settings["strokes_per_peak"])
    except ValueError as error:
        st.error(f"No stats: {error}")
        return

    overlay = output_dir / "sam2_track.mp4"
    if overlay.exists():
        with st.spinner("Preparing the video…"):
            path = web_video(overlay)
        portrait = track["height"] > track["width"]
        (st.columns([2, 3])[0] if portrait else st.container()).video(str(path))
        st.caption("The tinted outline is the swimmer as the analysis saw them. "
                   "Where it sits on something else, the numbers there are off.")

    counted = [length for length in stats["lengths"] if length["strokes"]]
    if whole_lengths:
        if not all(SPEED_RANGE[0] <= length["speed_mps"] <= SPEED_RANGE[1] for length in stats["lengths"]):
            st.warning("Some lengths give an unlikely speed, so a wall was probably missed or "
                       "misplaced. Check the pool length and the times under Adjust.")
        columns = st.columns(4)
        columns[0].metric("Time", f"{stats['total_s']:.2f} s", help=f"{stats['distance']:g} {unit}")
        columns[1].metric("Average speed", f"{stats['average_speed']:.2f} {unit}/s")
        all_counted = counted and len(counted) == len(stats["lengths"])
        columns[2].metric("Strokes", stats["total_strokes"] if all_counted else "–",
                          help="Each arm counts as one stroke")
        rates = [length["stroke_rate_spm"] for length in counted]
        columns[3].metric("Stroke rate", f"{np.mean(rates):.0f} /min" if rates else "–")
        table = pd.DataFrame([{
            "Length": length["length"],
            "Split (s)": length["split_s"],
            f"Speed ({unit}/s)": length["speed"],
            "Strokes": length["strokes"],
            "Rate (/min)": length["stroke_rate_spm"],
            f"Per stroke ({unit})": length["distance_per_stroke"],
            "First stroke (s)": length["first_stroke_s"],
        } for length in stats["lengths"]])
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f")
                                    for c in table.columns if c not in ("Length", "Strokes")})
        st.caption("Distance per stroke uses the whole length, underwater part included; "
                   "first stroke is the time from the start or wall to it.")
        moment_pictures(video, track, settings)
    else:
        st.info("This video shows part of a swim (it starts or ends mid-pool), so there are "
                "no splits or speed; the stroke rhythm is measured instead.")
        length = stats["lengths"][0]
        columns = st.columns(4)
        columns[0].metric("Stroke rate", f"{length['stroke_rate_spm']:.0f} /min" if length["strokes"] else "–")
        columns[1].metric("Arm cycle", f"{120 / length['stroke_rate_spm']:.2f} s" if length["strokes"] else "–",
                          help="Time for both arms to take one stroke each")
        columns[2].metric("Strokes counted", length["strokes"] or "–")
        variation = rhythm_variation(length["peak_frames"] or [])
        columns[3].metric("Rhythm variation", f"{variation:.0%}" if variation is not None else "–",
                          help="How much the time per arm cycle varies; lower is steadier")

    st.subheader("Suggestions")
    notes = suggestions(stats, whole_lengths)
    for note in notes:
        show = {"good": st.success, "watch": st.warning, "info": st.info}[note["kind"]]
        show(f"**{note['title']}**  \n{note['detail']}")
    if not notes:
        st.success("**Steady rhythm**  \nNothing stands out in the stroke timing.")
    st.caption("Suggestions come from timing and stroke counts only; technique is not judged.")

    if stats["peak_s"]:
        charts(track, stats, whole_lengths)
    adjust(video, output_dir, track, stats, settings)
    st.download_button("Download the numbers (JSON)", json.dumps(stats, indent=1),
                       file_name=f"{video.stem}_stats.json", mime="application/json")


# ----------------------------------------------------------------------- page


def main() -> None:
    st.set_page_config(page_title="Swim analyzer", layout="centered")
    st.title("Swim analyzer")
    st.caption("Drop a side-view freestyle video. It takes a few minutes to analyze "
               "(about 3 frames a second), then shows speed, strokes, timings and suggestions.")
    left, right = st.columns(2)
    pool = left.radio("Pool length", [25, 50], horizontal=True)
    unit = right.radio("Unit", ["m", "yd"], horizontal=True)
    upload = st.file_uploader("Video", type=VIDEO_TYPES, label_visibility="collapsed")
    if upload is None:
        return
    video = save_upload(upload)
    output_dir = ROOT / "outputs" / f"upload_{video.stem}"
    if not (output_dir / "analysis.json").exists():
        if not run_analysis(video, output_dir):
            return
    results(video, output_dir, pool, unit)


main()
