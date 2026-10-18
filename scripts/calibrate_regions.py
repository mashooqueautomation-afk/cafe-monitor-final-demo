#!/usr/bin/env python3
"""Draw a normalized (0..1) grid, table rectangles and/or an entry line on one video frame.

Why: cameras.yaml uses NORMALIZED coordinates (x1,y1,x2,y2 and line_ratio are fractions of width/height),
so they work at any resolution. Read coordinates off the grid, put them in config/cameras.yaml, re-run
this script with --regions/--line to check them, THEN restart the backend.

Examples (inside Docker use:  docker compose exec -w /workspace backend python scripts/calibrate_regions.py ...
          because the container's default folder is /workspace/models):
  python scripts/calibrate_regions.py data/client_footage/camera_04_seating_wide/Sample.mp4 --t 5
  python scripts/calibrate_regions.py data/client_footage/camera_04_seating_wide/Sample.mp4 \
      --regions "T1:0.10,0.55,0.25,0.75;T2:0.40,0.40,0.52,0.55"
  python scripts/calibrate_regions.py data/client_footage/camera_07_entrance/sample.mp4 --t 4 \
      --line horizontal:0.35

Output goes to runtime/calibration/ (never commit it: it contains client footage).
"""
import argparse
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--t", type=float, default=None, help="seconds into the clip (default: middle)")
    ap.add_argument("--regions", default="", help='"name:x1,y1,x2,y2;name2:x1,y1,x2,y2" (normalized)')
    ap.add_argument("--line", default="", help="vertical:0.72 or horizontal:0.35 (normalized)")
    ap.add_argument("--out", default="runtime/calibration")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {a.video} (check path AND capitalisation: Sample.mp4 vs sample.mp4)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.t * fps) if a.t is not None else n // 2)
    ok, img = cap.read()
    if not ok:
        raise SystemExit("Could not read that frame")
    h, w = img.shape[:2]

    for i in range(1, 10):                       # 10% grid with labels
        x, y = int(w * i / 10), int(h * i / 10)
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(img, f"{i / 10:.1f}", (x + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"{i / 10:.1f}", (3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    yaml_lines = []
    for item in filter(None, a.regions.split(";")):
        name, coords = item.split(":")
        x1, y1, x2, y2 = (float(v) for v in coords.split(","))
        assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1, f"{name}: need 0<=x1<x2<=1 and 0<=y1<y2<=1"
        cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 200, 255), 2)
        cv2.putText(img, name, (int(x1 * w) + 4, int(y1 * h) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        yaml_lines.append(f'      - {{ id: {name.lower()}, name: "{name}", x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2} }}')
    if a.line:
        orient, ratio = a.line.split(":"); r = float(ratio)
        if orient == "vertical":
            cv2.line(img, (int(r * w), 0), (int(r * w), h), (0, 210, 255), 3)
        else:
            cv2.line(img, (0, int(r * h)), (w, int(r * h)), (0, 210, 255), 3)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{Path(a.video).parent.name}_calibration.png"
    cv2.imwrite(str(dest), img)
    print(f"Frame {w}x{h} written to {dest}")
    if yaml_lines:
        print("\nPaste under the camera in config/cameras.yaml:\n    table_regions:\n" + "\n".join(yaml_lines))
    if a.line:
        print(f"\nPaste in config/cameras.yaml:  line_orientation: {orient}   line_ratio: {r}   entry_direction: <see note>")
        print("  vertical -> left_to_right | right_to_left     horizontal -> top_to_bottom | bottom_to_top")


if __name__ == "__main__":
    main()
