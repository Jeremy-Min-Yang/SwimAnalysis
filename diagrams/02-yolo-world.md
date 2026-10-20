# 2. Finding the swimmer: YOLO-World

Code: `detect` and `swimmer_path` in `src/analyze.py`.
Weights: `models/yolov8s-worldv2.pt`.

## What YOLO-World is

A normal YOLO is trained on a fixed list of classes (COCO's 80: person,
car, ...). "Swimmer" is not one of them. YOLO-World is *open-vocabulary*:
you give it class names as text, and it finds boxes that match the text.

```mermaid
flowchart LR
    subgraph once["Once: model.set_classes"]
        text["'swimmer'<br/>'person swimming'"] --> clip["CLIP text encoder"] --> temb[("text embeddings<br/>one vector per class")]
    end

    subgraph perframe["Every 5th frame or so"]
        img["video frame"] --> backbone["YOLOv8 backbone<br/>image features at<br/>3 scales"]
        backbone --> pan["RepVL-PAN<br/>mixes text embeddings<br/>into the image features"]
        temb -.-> pan
        pan --> head["detection head<br/>per anchor: a box +<br/>an embedding"]
        head --> sim["similarity of each box embedding<br/>to each text embedding<br/>= confidence per class"]
        temb -.-> sim
        sim --> nms["keep conf ≥ 0.08,<br/>non-max suppression"]
    end

    nms --> boxes[("boxes + confidences")]
```

Key idea: the class is not a fixed output neuron; it is "how close is this
region's embedding to the embedding of the word". That is why text prompts
work without retraining, and also why confidences are low (0.1–0.4 is
normal here) and why "swimmer" sometimes fires on a person on the deck.

## From many boxes to one swimmer path

`detect` runs YOLO-World `DETECTIONS_PER_S` (6) times a second and drops
boxes taller than `MAX_TALL` × their width: a swimmer lies flat.
`swimmer_path` then picks *one* swimmer across the video.

```mermaid
flowchart TD
    dets[("boxes per detection frame")] --> score["score every box:<br/>confidence × √area × closeness to picture center"]
    score --> anchor["anchor = best box in the whole video<br/>(the swimmer the camera follows)"]
    anchor --> fwd["walk forward frame by frame"]
    anchor --> back["walk backward frame by frame"]

    fwd & back --> step{"any box within reach<br/>of the last chosen box?"}
    step -- "horizontal: box width × (0.5 + MAX_PATH_SPEED × seconds)<br/>vertical: box height × (0.5 + MAX_PATH_VERTICAL_SPEED × seconds)" --> yes["take the nearest,<br/>favouring confident ones"]
    step -- none --> skip["no box on this frame;<br/>reach grows with time"]
    yes --> step
    skip --> step

    yes --> path[("path: frame → (box, confidence)")]
    path --> seeds["boxes with conf ≥ SEED_CONFIDENCE (0.25)<br/>may seed SAM 2"]
```

Why the vertical limit is tight: the camera pans *along* the pool, so the
swimmer moves left–right in the picture but hardly up–down. A big vertical
step is another lane. Without the limit the path hopped to a far-lane
swimmer whenever the near one was under water and undetected.

## What this step hands on

- The **first** seed-worthy box → SAM 2's starting prompt.
- The **whole path** → `make_reseeder`, used in [3](03-sam2-tracking.md)
  to put SAM 2 back on the swimmer.
