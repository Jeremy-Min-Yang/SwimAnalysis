"""Coaching notes from race stats.

Only numbers the video measures reliably are used: splits, stroke rate,
distance per stroke, and the rhythm of the detected strokes. Each note
states the numbers behind it. Technique (catch, body line, kick) is not
judged, since nothing here sees it.
"""

from __future__ import annotations

import numpy as np


# A change of at least this fraction between the first and last counted
# lengths is worth a note.
TREND = 0.08
# Peak-to-peak times varying more than this (coefficient of variation) is an
# uneven rhythm.
UNEVEN_RHYTHM = 0.12
# A turn whose first stroke comes this many seconds later than the other
# turns' is a slow breakout.
SLOW_BREAKOUT_S = 0.7


def cycle_times(peak_frames: list[int]) -> list[tuple[int, int, float]]:
    """(from, to, frames per cycle) for each pair of neighbouring peaks. A gap
    spanning a stroke hidden by splash is split into the cycles it spans."""
    if not peak_frames or len(peak_frames) < 2:
        return []
    gaps = np.diff(peak_frames)
    typical = float(np.median(gaps))
    return [
        (int(a), int(b), float(gap / max(1, round(gap / typical))))
        for a, b, gap in zip(peak_frames[:-1], peak_frames[1:], gaps)
    ]


def rhythm_variation(peak_frames: list[int]) -> float | None:
    """Coefficient of variation of the time per stroke cycle."""
    cycles = np.array([cycle for _, _, cycle in cycle_times(peak_frames)])
    if len(cycles) < 3:
        return None
    return float(cycles.std() / cycles.mean())


def suggestions(stats: dict, whole_lengths: bool = True) -> list[dict]:
    """Notes as {"kind": "info" | "watch" | "good", "title", "detail"}."""
    notes = []
    lengths = stats["lengths"]
    unit = stats["unit"]
    counted = [length for length in lengths if length["strokes"]]

    for length in lengths:
        if not length["strokes"]:
            where = f"Length {length['length']}" if whole_lengths else "This clip"
            notes.append(
                {
                    "kind": "info",
                    "title": f"{where}: no steady stroke rhythm found",
                    "detail": "The track probably lost the swimmer. Watch the overlay video, "
                    "add a mark where it drifts, and track again.",
                }
            )

    for length in counted:
        variation = rhythm_variation(length["peak_frames"])
        if variation is not None and variation > UNEVEN_RHYTHM:
            where = f"Length {length['length']}" if whole_lengths else "The clip"
            notes.append(
                {
                    "kind": "watch",
                    "title": f"{where}: uneven stroke rhythm",
                    "detail": f"Time between strokes varies by {variation:.0%}. A pause at the "
                    "breath or a catch-up glide at the front often causes this; "
                    "check the frames where the gap is longest.",
                }
            )

    if whole_lengths and len(counted) >= 2:
        first, last = counted[0], counted[-1]
        rate_change = last["stroke_rate_spm"] / first["stroke_rate_spm"] - 1
        # Length 1 includes the dive, which carries the swimmer much further
        # than a push-off, so distance per stroke compares turn lengths only.
        turn_lengths = [length for length in counted if length["length"] > 1]
        if len(turn_lengths) >= 2:
            dps_first, dps_last = turn_lengths[0], turn_lengths[-1]
            dps_change = dps_last["distance_per_stroke"] / dps_first["distance_per_stroke"] - 1
        else:
            dps_first = dps_last = None
            dps_change = 0.0
        span = f"length {first['length']} to {last['length']}"
        if rate_change <= -TREND and dps_change <= -TREND / 2:
            notes.append(
                {
                    "kind": "watch",
                    "title": "Tempo and grip both fade",
                    "detail": f"Stroke rate fell {-rate_change:.0%} ({first['stroke_rate_spm']:.0f} to "
                    f"{last['stroke_rate_spm']:.0f}/min) from {span} and distance per stroke "
                    f"{-dps_change:.0%} from length {dps_first['length']} to {dps_last['length']}: "
                    "classic fatigue. Race-pace repeats with short rest build "
                    "the ability to hold both.",
                }
            )
        elif rate_change <= -TREND:
            notes.append(
                {
                    "kind": "watch",
                    "title": "Stroke rate drops",
                    "detail": f"Stroke rate fell {-rate_change:.0%} ({first['stroke_rate_spm']:.0f} to "
                    f"{last['stroke_rate_spm']:.0f}/min) from {span}. Holding tempo late "
                    "(a tempo trainer helps) keeps speed up.",
                }
            )
        elif rate_change >= TREND and dps_change <= -TREND:
            notes.append(
                {
                    "kind": "watch",
                    "title": "Spinning: faster arms, shorter strokes",
                    "detail": f"Stroke rate rose {rate_change:.0%} from {span} but distance per "
                    f"stroke fell {-dps_change:.0%} from length {dps_first['length']} to "
                    f"{dps_last['length']}. Faster turnover is buying less per "
                    "stroke; focus on a full pull to the hip as the rate climbs.",
                }
            )
        elif dps_change <= -TREND:
            notes.append(
                {
                    "kind": "watch",
                    "title": "Distance per stroke drops",
                    "detail": f"Each stroke covered {-dps_change:.0%} less water from length "
                    f"{dps_first['length']} to {dps_last['length']} "
                    f"({dps_first['distance_per_stroke']:.2f} to {dps_last['distance_per_stroke']:.2f} {unit}).",
                }
            )
        else:
            notes.append(
                {
                    "kind": "good",
                    "title": "Tempo and stroke length held" if dps_first else "Tempo held",
                    "detail": f"Stroke rate stayed within {TREND:.0%} from {span}"
                    + (" and so did distance per stroke over the turn lengths" if dps_first else "")
                    + ": even pacing.",
                }
            )

    if whole_lengths and len(lengths) >= 3:
        # Length 1 includes the dive or push-off, so compare later lengths.
        later = lengths[1:]
        fade = later[-1]["split_s"] / later[0]["split_s"] - 1
        if fade >= 0.05:
            notes.append(
                {
                    "kind": "watch",
                    "title": "Back-half fade",
                    "detail": f"Length {later[-1]['length']} took {fade:.0%} longer than length "
                    f"{later[0]['length']} ({later[-1]['split_s']:.2f} s against "
                    f"{later[0]['split_s']:.2f} s).",
                }
            )

        turns = [length for length in later if length["first_stroke_s"] is not None]
        if len(turns) >= 2:
            typical = float(np.median([length["first_stroke_s"] for length in turns]))
            for length in turns:
                if length["first_stroke_s"] - typical >= SLOW_BREAKOUT_S:
                    notes.append(
                        {
                            "kind": "watch",
                            "title": f"Slow breakout on length {length['length']}",
                            "detail": f"First stroke {length['first_stroke_s']:.1f} s after the "
                            f"wall, against {typical:.1f} s on the other turns. A long glide "
                            "loses speed once it drops below swimming pace.",
                        }
                    )
    return notes
