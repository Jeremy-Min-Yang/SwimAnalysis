"""Run ViTPose-SwimXYZ on the boxes the tracker already found and compare it
with the saved RTMPose keypoints.

Both models see the same swimmer box each frame (from detect_pose.py's
landmarks.json), so differences come from the pose model alone, not from
detection. There is no ground truth, so two proxies stand in for accuracy:

- jitter: median keypoint acceleration |p[t+1] - 2 p[t] + p[t-1]| in torso
  lengths. A real body moves smoothly; a model that jumps between joints, or
  between arms and legs, jitters.
- limb spread: coefficient of variation of thigh and shin length over the
  video. Real limbs keep their length (thighs and shins barely foreshorten
  from the side), so a high spread means wrong keypoints.

Writes landmarks_vitpose.json (same format as landmarks.json, so
analyze_strokes.py can read it), vitpose_compare.mp4, and
vitpose_compare.json next to the landmarks.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

from detect_pose import COCO_LANDMARK_NAMES, draw_swimmer, pose_from_record
from vitpose import ViTPoseSwim


BODY = list(range(5, 17))
SEGMENTS = {"thigh": ((11, 13), (12, 14)), "shin": ((13, 15), (14, 16))}
TILE = (640, 360)


def torso_length(keypoints: np.ndarray) -> float:
    shoulders = (keypoints[5] + keypoints[6]) / 2
    hips = (keypoints[11] + keypoints[12]) / 2
    return float(np.linalg.norm(shoulders - hips))


def pose_stats(keypoints: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Jitter and limb spread for one model; keypoints is (frames, 17, 2)."""
    torso = np.array([torso_length(k) for k in keypoints])
    torso_scale = np.nanmedian(np.where(torso > 1, torso, np.nan))
    sure = scores >= threshold

    acceleration = np.linalg.norm(
        keypoints[2:] - 2 * keypoints[1:-1] + keypoints[:-2], axis=2
    ) / torso_scale
    all_sure = sure[2:] & sure[1:-1] & sure[:-2]
    body_accel = acceleration[:, BODY][all_sure[:, BODY]]

    spreads = {}
    for name, pairs in SEGMENTS.items():
        lengths = []
        for a, b in pairs:
            ok = sure[:, a] & sure[:, b]
            lengths.extend(
                (np.linalg.norm(keypoints[ok, a] - keypoints[ok, b], axis=1) / torso_scale).tolist()
            )
        lengths = np.array(lengths)
        spreads[name] = float(lengths.std() / lengths.mean()) if len(lengths) > 10 else None

    return {
        "mean_body_confidence": float(scores[:, BODY].mean()),
        "sure_body_keypoints": float(sure[:, BODY].mean()),
        "jitter_median_torso": float(np.median(body_accel)) if body_accel.size else None,
        "jitter_p90_torso": float(np.percentile(body_accel, 90)) if body_accel.size else None,
        "limb_length_spread": spreads,
    }


def crop_tile(frame, box):
    """Cut a 16:9 window around box and resize it to one tile."""
    height, width = frame.shape[:2]
    center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    crop_width = min(max(box[2] - box[0], (box[3] - box[1]) * 16 / 9) * 1.3, width)
    crop_height = min(crop_width * 9 / 16, height)
    crop_width = crop_height * 16 / 9
    x1 = int(min(max(center_x - crop_width / 2, 0), width - crop_width))
    y1 = int(min(max(center_y - crop_height / 2, 0), height - crop_height))
    tile = frame[y1 : y1 + int(crop_height), x1 : x1 + int(crop_width)]
    return cv2.resize(tile, TILE)


def label(tile, text: str) -> None:
    cv2.putText(tile, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(tile, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def compare(landmarks_path: Path, device: str, threshold: float, limit: int | None) -> int:
    data = json.loads(landmarks_path.read_text(encoding="utf-8"))
    out_dir = landmarks_path.parent
    video_path = (out_dir / data["video"]).resolve()
    records = {record["frame"]: record for record in data["frames"] if record.get("box")}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Error: OpenCV could not open: {video_path}")
        return 1
    writer = cv2.VideoWriter(
        str(out_dir / "vitpose_compare.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        data["fps"],
        (TILE[0], TILE[1] * 2),
    )

    print("Loading ViTPose-SwimXYZ...")
    model = ViTPoseSwim(device=device)

    new_frames = []
    rtm_points, rtm_scores, vit_points, vit_scores = [], [], [], []
    frame_number = 0
    while limit is None or frame_number < limit:
        success, frame = capture.read()
        if not success:
            break
        record = records.get(frame_number)
        frame_number += 1
        if record is None:
            continue

        box, rtm_kp, rtm_sc = pose_from_record(record)
        keypoints, scores = model(frame, bboxes=[box])
        keypoints, scores = keypoints[0], scores[0]

        new_record = copy.deepcopy(record)
        new_record["model"] = "vitpose-swimxyz"
        new_record["keypoints"] = {
            name: {"x": float(x), "y": float(y), "confidence": float(s)}
            for name, (x, y), s in zip(COCO_LANDMARK_NAMES, keypoints, scores)
        }
        new_frames.append(new_record)
        rtm_points.append(rtm_kp)
        rtm_scores.append(rtm_sc)
        vit_points.append(keypoints)
        vit_scores.append(scores)

        view_box = record.get("display_box") or box
        top, bottom = frame.copy(), frame.copy()
        draw_swimmer(top, (box, rtm_kp, rtm_sc), threshold)
        draw_swimmer(bottom, (box, keypoints.tolist(), scores.tolist()), threshold)
        top, bottom = crop_tile(top, view_box), crop_tile(bottom, view_box)
        label(top, f"RTMPose-x (saved)  frame {record['frame']}")
        label(bottom, "ViTPose-L SwimXYZ")
        writer.write(np.vstack([top, bottom]))
        if len(new_frames) % 100 == 0:
            print(f"  {len(new_frames)}/{len(records)} frames")
    capture.release()
    writer.release()

    output = {key: value for key, value in data.items() if key != "frames"}
    output["annotated_video"] = "vitpose_compare.mp4"
    output["frames"] = new_frames
    (out_dir / "landmarks_vitpose.json").write_text(json.dumps(output), encoding="utf-8")

    summary = {
        "frames": len(new_frames),
        "keypoint_threshold": threshold,
        "rtmpose-x": pose_stats(np.array(rtm_points), np.array(rtm_scores), threshold),
        "vitpose-swimxyz": pose_stats(np.array(vit_points), np.array(vit_scores), threshold),
    }
    (out_dir / "vitpose_compare.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nOutputs in {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("landmarks", type=Path, help="landmarks.json from detect_pose.py")
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    parser.add_argument("--keypoint-confidence", type=float, default=0.3)
    parser.add_argument("--limit", type=int, help="Stop after this many video frames")
    args = parser.parse_args()
    return compare(args.landmarks, args.device, args.keypoint_confidence, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
