# Swimline diagrams

How a dropped video becomes race stats. Read in order; each file zooms into
one box of the one before.

1. [System overview](01-system-overview.md): the app, the analysis process,
   and the files passed between them.
2. [Finding the swimmer: YOLO-World](02-yolo-world.md): what the detector
   does inside, and how its boxes become one swimmer path.
3. [Following the swimmer: SAM 2](03-sam2-tracking.md): what SAM 2 does
   inside, the per-frame loop, mask checks, memory and re-seeding.
4. [Times and strokes](04-timing-and-strokes.md): from a mask per frame to
   start, walls, finish, stroke count and rate.

The diagrams are Mermaid. GitHub renders them; in VS Code, install the
"Markdown Preview Mermaid Support" extension and open the Markdown preview.

Constant names in the diagrams (`MAX_TALL`, `LOST_FRAMES`, ...) are the
ones in the code, so they can be searched for.
