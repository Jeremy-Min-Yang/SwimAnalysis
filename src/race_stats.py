"""Race stats from a swim video: splits, speed, stroke count, stroke rate, and
distance per stroke for each pool length.

Needs outputs/<video>/sam2_track.json from track_sam2.py, the pool length,
and when the swimmer started and touched each wall (--start, --touch). Those
moments are hard to find automatically in a panning phone video, so for now
they are entered by hand; everything else is measured.

Strokes come from the shape of the swimmer's SAM 2 mask, not from a pose
skeleton: the recovering arm lifts out of the water once per arm cycle, so
how far the mask reaches above its own center (scaled by the swimmer's size)
rises and falls with the stroke. Box width is left out on purpose: feet
drop in and out of the mask in the splash.

The typical cycle time comes from autocorrelation over the race; each cycle
is one peak, counted only where the signal is clearly periodic, so the glide
after the start and turns does not count. A cycle hidden by splash is filled
in from the rhythm.

What one peak is depends on the camera: usually only the near arm clears the
swimmer's outline, so a peak is one arm cycle, two strokes (--peak cycle).
Filmed close and side-on at water level, both arms do, and a peak is one
stroke (--peak arm). The signal alone cannot tell these apart.

When a length's signal is not periodic enough, its stroke numbers are left
out (null) rather than guessed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METERS_PER_YARD = 0.9144
# Freestyle arm cycles last between these many seconds (sprinters about 0.9,
# long-glide swimmers about 3).
MIN_CYCLE_S = 0.6
MAX_CYCLE_S = 3.5
# Below this autocorrelation at the cycle time, the race has no usable rhythm.
MIN_PERIODICITY = 0.3
# Remove slow changes (camera zoom, swimmer turning) with a moving average this
# many cycles long, and smooth splash noise over this fraction of a cycle.
DETREND_CYCLES = 1.0
SMOOTH_CYCLES = 0.15
# A frame is "stroking" when the signal over the surrounding LOCAL_CYCLES
# cycles correlates with itself one cycle later at least this much.
LOCAL_CYCLES = 2.0
LOCAL_PERIODICITY = 0.3
# Peaks: at least this many cycles apart, rising this many standard
# deviations above the surrounding lows.
PEAK_SPACING = 0.6
PEAK_PROMINENCE = 0.5
# A peak this close (in cycles) to one cycle from a counted peak also counts,
# if the rhythm there is at least CHAIN_PERIODICITY (the edges of a stretch
# of strokes score lower, a glide or push-off scores about 0).
CHAIN_TOLERANCE = 0.25
CHAIN_PERIODICITY = 0.1
# Peaks within this many cycles before a touch are the reach to the wall.
TOUCH_CYCLES = 0.5
# Peaks within this many frames of a frame without a mask are made by filling
# the gap, not by the swimmer.
GAP_MARGIN = 2


def fill_gaps(signal: np.ndarray) -> np.ndarray | None:
    """Fill NaN gaps by linear interpolation."""
    known = np.flatnonzero(np.isfinite(signal))
    if len(known) < 2:
        return None
    return np.interp(np.arange(len(signal)), known, signal[known])


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average, padding the ends by repeating end values."""
    half = window // 2
    if half < 1:
        return values
    padded = np.concatenate(([values[0]] * half, values, [values[-1]] * half))
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.convolve(padded, kernel, mode="valid")


def find_peaks(
    signal: np.ndarray, distance: int, base_distance: int, prominence: float
) -> list[int]:
    """Local maxima that are the highest within +/- distance samples and rise
    at least prominence above the lowest point within +/- base_distance
    samples on both sides."""
    # Pad with the minimum so a peak at either end of the signal can count.
    padded = np.concatenate(([signal.min()], signal, [signal.min()]))
    peaks = []
    for index in range(1, len(padded) - 1):
        value = padded[index]
        if value < padded[index - 1] or value <= padded[index + 1]:
            continue
        if value < padded[max(0, index - distance) : index + distance + 1].max():
            continue
        left_low = padded[max(0, index - base_distance) : index].min()
        right_low = padded[index + 1 : index + base_distance + 1].min()
        if value - max(left_low, right_low) < prominence:
            continue
        peaks.append(index - 1)
    return peaks


def mask_shape_signal(frames: list[dict]) -> np.ndarray:
    """Per frame, how far the swimmer mask reaches above its centroid, over the
    square root of its area; NaN when lost."""
    values = np.full(len(frames), np.nan)
    for index, frame in enumerate(frames):
        mask = frame["mask"]
        if mask is None:
            continue
        values[index] = (mask["centroid"][1] - mask["box"][1]) / np.sqrt(mask["area"])
    return values


def autocorrelation(signal: np.ndarray) -> np.ndarray:
    centered = signal - signal.mean()
    correlation = np.correlate(centered, centered, "full")[len(signal) - 1 :]
    return correlation / correlation[0] if correlation[0] > 0 else correlation


def cycle_frames(signal: np.ndarray, fps: float) -> tuple[float | None, float]:
    """Typical arm-cycle length in frames, and the autocorrelation there."""
    lowest, highest = int(MIN_CYCLE_S * fps), int(MAX_CYCLE_S * fps)
    if len(signal) <= highest:
        return None, 0.0
    correlation = autocorrelation(signal)
    lag = lowest + int(np.argmax(correlation[lowest:highest]))
    # Refine to a fraction of a frame with a parabola through the peak.
    a, b, c = correlation[lag - 1 : lag + 2]
    offset = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c < 0 else 0.0
    return lag + offset, float(correlation[lag])


def local_periodicity(signal: np.ndarray, period: int) -> np.ndarray:
    """Per frame, correlation of the surrounding window with itself one cycle later."""
    half = int(LOCAL_CYCLES * period / 2)
    result = np.zeros(len(signal))
    for index in range(len(signal)):
        start, end = max(0, index - half), min(len(signal) - period, index + half)
        if end - start < period:
            continue
        first, second = signal[start:end], signal[start + period : end + period]
        if first.std() > 0 and second.std() > 0:
            result[index] = np.corrcoef(first, second)[0, 1]
    return result


def extend_chain(
    accepted: list[int], candidates: list[int], cycle: int, edges: list[int]
) -> list[int]:
    """Add candidate peaks one or two cycles (CHAIN_TOLERANCE) before or after
    an accepted peak in the same length, repeating until none are added. Two
    cycles skip over one cycle hidden by splash or a lost mask.

    The rhythm check needs a few cycles on both sides, so it misses the first
    stroke after a breakout and the last before a wall; a peak right where
    the next cycle is due is taken anyway.
    """
    def length_of(frame: int) -> int:
        return sum(frame > edge for edge in edges)

    chain = set(accepted)
    added = True
    while added:
        added = False
        for candidate in candidates:
            if candidate in chain:
                continue
            for peak in chain:
                if length_of(peak) == length_of(candidate) and any(
                    abs(abs(candidate - peak) - steps * cycle) <= CHAIN_TOLERANCE * cycle
                    for steps in (1, 2)
                ):
                    chain.add(candidate)
                    added = True
                    break
    return sorted(chain)


def race_stats(
    track: dict,
    pool_length: float,
    unit: str,
    start_s: float,
    touches_s: list[float],
    strokes_per_peak: int = 2,
) -> dict:
    fps = track["fps"]
    frames = track["frames"]
    start, touches = round(start_s * fps), [round(t * fps) for t in touches_s]
    race = slice(start, touches[-1] + 1)

    measured = mask_shape_signal(frames)
    raw = fill_gaps(measured)
    if raw is None:
        raise ValueError("the swimmer mask is missing in almost every frame")
    period, periodicity = cycle_frames(
        raw[race] - moving_average(raw[race], int(fps)), fps
    )

    lengths = []
    edges = [start] + touches
    meters = pool_length * (METERS_PER_YARD if unit == "yd" else 1.0)
    peaks: list[int] = []
    if period is not None and periodicity >= MIN_PERIODICITY:
        cycle = int(round(period))
        signal = raw - moving_average(raw, max(3, int(DETREND_CYCLES * cycle)))
        signal = moving_average(signal, max(1, int(SMOOTH_CYCLES * cycle)))
        signal = (signal - signal[race].mean()) / signal[race].std()
        rhythm = local_periodicity(signal, cycle)
        candidates = [
            peak
            for peak in find_peaks(signal, int(PEAK_SPACING * cycle), cycle, PEAK_PROMINENCE)
            if start <= peak <= touches[-1]
            and not any(0 <= touch - peak < TOUCH_CYCLES * cycle for touch in touches)
            and np.isfinite(measured[max(0, peak - GAP_MARGIN) : peak + GAP_MARGIN + 1]).all()
        ]
        peaks = extend_chain(
            [peak for peak in candidates if rhythm[peak] >= LOCAL_PERIODICITY],
            [peak for peak in candidates if rhythm[peak] >= CHAIN_PERIODICITY],
            cycle,
            edges,
        )

    for number, (begin, end) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        split = (end - begin) / fps
        length = {
            "length": number,
            "start_s": begin / fps,
            "touch_s": end / fps,
            "split_s": split,
            "speed": pool_length / split,
            "speed_mps": meters / split,
        }
        length_peaks = [peak for peak in peaks if begin < peak < end]
        if len(length_peaks) >= 2:
            # Count cycles across the span, so one hidden by splash or a lost
            # mask still counts.
            span = length_peaks[-1] - length_peaks[0]
            count = max(len(length_peaks), round(span / period) + 1)
            peak_s = span / fps / (count - 1)
            strokes = strokes_per_peak * count
            length.update(
                {
                    "strokes": strokes,
                    "stroke_rate_spm": 60 * strokes_per_peak / peak_s,
                    "distance_per_stroke": pool_length / strokes,
                    "first_stroke_s": (length_peaks[0] - begin) / fps,
                    "peak_frames": length_peaks,
                }
            )
        else:
            length.update(
                {
                    key: None
                    for key in (
                        "strokes",
                        "stroke_rate_spm",
                        "distance_per_stroke",
                        "first_stroke_s",
                        "peak_frames",
                    )
                }
            )
        lengths.append(length)

    total_s = (touches[-1] - start) / fps
    distance = pool_length * len(touches)
    counted = [length for length in lengths if length["strokes"]]
    return {
        "pool_length": pool_length,
        "unit": unit,
        "distance": distance,
        "total_s": total_s,
        "average_speed": distance / total_s,
        "average_speed_mps": meters * len(touches) / total_s,
        "strokes_per_peak": strokes_per_peak,
        "peak_s": period / fps if period else None,
        "periodicity": periodicity,
        "total_strokes": sum(length["strokes"] for length in counted) if counted else None,
        "lengths": lengths,
    }


def print_report(stats: dict) -> None:
    unit = stats["unit"]
    print(
        f"{stats['distance']:g} {unit} in {stats['total_s']:.2f} s, "
        f"average {stats['average_speed']:.2f} {unit}/s"
    )
    if stats["peak_s"]:
        what = "arm cycle" if stats["strokes_per_peak"] == 2 else "stroke"
        print(f"One {what} every {stats['peak_s']:.2f} s (rhythm strength {stats['periodicity']:.2f})")
    print(
        f"\n{'length':>6} {'split':>7} {'speed':>7} {'strokes':>8} {'rate':>9} "
        f"{'dist/strk':>10} {'1st strk':>9}"
    )
    print(f"{'':>6} {'s':>7} {unit + '/s':>7} {'':>8} {'strk/min':>9} {unit:>10} {'s':>9}")
    for length in stats["lengths"]:
        def show(key, width, digits=2):
            value = length[key]
            return f"{value:{width}.{digits}f}" if value is not None else f"{'-':>{width}}"

        strokes = f"{length['strokes']:>8}" if length["strokes"] is not None else f"{'-':>8}"
        print(
            f"{length['length']:>6} {show('split_s', 7)} {show('speed', 7)} {strokes} "
            f"{show('stroke_rate_spm', 9, 1)} {show('distance_per_stroke', 10)} "
            f"{show('first_stroke_s', 9)}"
        )
    print(
        "\nStrokes count each arm. Distance per stroke uses the whole length,"
        "\nincluding the underwater part after the start or turn."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("video", type=Path, help="The video, or its outputs folder")
    parser.add_argument("--pool", type=float, required=True, choices=(25, 50), help="Pool length")
    parser.add_argument("--unit", choices=("m", "yd"), default="m")
    parser.add_argument("--start", type=float, required=True, help="Seconds when the swimmer leaves the block or wall")
    parser.add_argument(
        "--touch",
        required=True,
        help="Seconds of each wall touch, comma-separated: turns then the finish",
    )
    parser.add_argument(
        "--peak",
        choices=("cycle", "arm"),
        default="cycle",
        help="What one rise of the outline is: a full arm cycle (far arm hidden, "
        "the usual case) or one arm (close side-on view, both arms visible)",
    )
    parser.add_argument("-o", "--output", type=Path, help="Default: race_stats.json next to the track")
    args = parser.parse_args()

    folder = args.video if args.video.is_dir() else Path("outputs") / args.video.stem
    track_path = folder / "sam2_track.json"
    if not track_path.is_file():
        parser.error(f"{track_path} not found; run track_sam2.py first")
    try:
        touches = [float(value) for value in args.touch.split(",")]
    except ValueError:
        parser.error("--touch must be comma-separated seconds")
    if any(b <= a for a, b in zip([args.start] + touches, touches)):
        parser.error("--start and --touch times must increase")

    track = json.loads(track_path.read_text(encoding="utf-8"))
    strokes_per_peak = 2 if args.peak == "cycle" else 1
    stats = race_stats(track, args.pool, args.unit, args.start, touches, strokes_per_peak)
    output = args.output or folder / "race_stats.json"
    output.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print_report(stats)
    print(f"\nStats: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
