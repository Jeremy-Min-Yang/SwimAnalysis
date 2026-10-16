"""Re-encode a video as H.264 so a browser can play it.

OpenCV writes mp4v, which browsers do not play. Frames are scaled down so
the longer side is at most MAX_SIZE.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import av


MAX_SIZE = 960


def convert(source: Path, target: Path) -> None:
    with av.open(str(source)) as reader:
        video = reader.streams.video[0]
        scale = min(1.0, MAX_SIZE / max(video.width, video.height))
        width, height = int(video.width * scale) // 2 * 2, int(video.height * scale) // 2 * 2
        rate = Fraction(video.average_rate or 30).limit_denominator(1001)
        with av.open(str(target), "w", options={"movflags": "faststart"}) as writer:
            stream = writer.add_stream("libx264", rate=rate)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            stream.options = {"crf": "26", "preset": "veryfast"}
            for index, frame in enumerate(reader.decode(video)):
                frame = frame.reformat(width=width, height=height, format="yuv420p")
                frame.pts, frame.time_base = index, 1 / rate
                for packet in stream.encode(frame):
                    writer.mux(packet)
            for packet in stream.encode():
                writer.mux(packet)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    convert(args.source, args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
