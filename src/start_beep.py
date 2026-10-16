"""Find candidate start signals in a swim video's sound track.

Race starts use an electronic beep: a loud tone around 1 kHz whose pitch
does not move. Stretches where that band is loud and the pitch holds steady
are candidates, ranked by loudness. Splashes are too short and voices and
whistles slide in pitch, but a call to the blocks or hum from the pool hall
can rank too, so the candidates are suggestions to check against the video,
not the answer. A video without a beep (a push off the wall, or no sound)
gives none, and the start is marked by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BAND_HZ = (900.0, 1500.0)
HOP_S = 0.01
WINDOW_S = 0.04
# Band energy this many times its median over the clip counts as loud.
LOUDNESS = 8.0
# Quiet gaps up to this long inside a tone do not split it.
MAX_GAP_S = 0.03
MIN_TONE_S = 0.1
# Steady pitch: this share of the tone's first MIN_TONE_S peaks within
# PITCH_HZ of its median pitch.
PITCH_HZ = 40.0
MIN_STEADY_SHARE = 0.7


def read_audio(video_path: Path) -> tuple[np.ndarray, int] | None:
    """Mono samples and sample rate, or None when the video has no sound."""
    import av

    with av.open(str(video_path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        chunks = []
        for frame in container.decode(stream):
            samples = frame.to_ndarray()
            if frame.format.is_planar:
                chunks.append(samples.mean(axis=0))
            else:
                chunks.append(samples.reshape(-1, stream.channels).mean(axis=1))
        if not chunks:
            return None
        return np.concatenate(chunks).astype(np.float64), stream.rate


def find_start_beeps(video_path: Path, limit: int = 3) -> list[float]:
    """Seconds into the video where a start beep may begin, loudest first."""
    audio = read_audio(video_path)
    if audio is None:
        return []
    samples, rate = audio
    hop, window = int(HOP_S * rate), int(WINDOW_S * rate)
    if len(samples) < window:
        return []
    frequencies = np.fft.rfftfreq(window, 1 / rate)
    band = (frequencies >= BAND_HZ[0]) & (frequencies <= BAND_HZ[1])
    taper = np.hanning(window)
    starts = range(0, len(samples) - window, hop)
    energy = np.empty(len(starts))
    pitch = np.empty(len(starts))
    for i, start in enumerate(starts):
        spectrum = np.abs(np.fft.rfft(samples[start : start + window] * taper))[band] ** 2
        energy[i] = spectrum.sum()
        pitch[i] = frequencies[band][np.argmax(spectrum)]

    typical = np.median(energy)
    if typical <= 0:
        return []
    strength = energy / typical
    loud = strength > LOUDNESS
    max_gap = int(round(MAX_GAP_S / HOP_S))
    tones = []
    index = 0
    while index < len(loud):
        if not loud[index]:
            index += 1
            continue
        end = index
        while end + 1 < len(loud) and loud[end + 1 : end + 2 + max_gap].any():
            end += 1
        # Judge the pitch over the tone's start only: a beep can ring on into
        # crowd noise.
        head = slice(index, min(end + 1, index + int(round(MIN_TONE_S / HOP_S))))
        steady = np.mean(np.abs(pitch[head] - np.median(pitch[head])) <= PITCH_HZ)
        if (end - index + 1) * HOP_S >= MIN_TONE_S and steady >= MIN_STEADY_SHARE:
            tones.append((float(strength[head].max()), index * HOP_S))
        index = end + 1
    return [onset for _, onset in sorted(tones, reverse=True)[:limit]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--json", action="store_true", help="Print {video: [seconds, ...]}")
    args = parser.parse_args()
    if args.json:
        print(json.dumps({str(video): find_start_beeps(video) for video in args.videos}))
        return 0
    for video in args.videos:
        beeps = find_start_beeps(video)
        found = ", ".join(f"{onset:.2f} s" for onset in beeps) if beeps else "none"
        print(f"{video}: possible start beeps {found}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
