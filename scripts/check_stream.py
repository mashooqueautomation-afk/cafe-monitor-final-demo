"""Check a camera stream and save a snapshot with a 0..1 coordinate grid.

Use it for TWO things:
  1. Connection test: does the stream open? What resolution / FPS do we really get?
  2. Calibration: open the saved picture and read where the door line and the table
     rectangles should go (values between 0.0 and 1.0) for config/cameras.yaml.

Examples (run from the project root):
  python scripts/check_stream.py --source "rtsp://USER:PASS@192.168.1.64:554/Streaming/Channels/102"
  python scripts/check_stream.py --source webcam:0
  python scripts/check_stream.py --source "http://192.168.1.20:8080/video"     # phone app
  python scripts/check_stream.py --camera 1          # reads CAMERA_01_RTSP from your .env file

The password is never printed. Needs: pip install opencv-python-headless
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import cv2

ROOT = Path(__file__).resolve().parents[1]


def read_env_file() -> dict[str, str]:
    env = {}
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def mask(url: str) -> str:
    if url.startswith("webcam:") or "://" not in url:
        return url
    p = urlsplit(url)
    return f"{p.scheme}://{'***@' if (p.username or p.password) else ''}{p.hostname}{f':{p.port}' if p.port else ''}"


def draw_grid(frame):
    h, w = frame.shape[:2]
    out = frame.copy()
    for i in range(1, 10):
        x, y = int(w * i / 10), int(h * i / 10)
        cv2.line(out, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.line(out, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(out, f"{i / 10:.1f}", (x + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"{i / 10:.1f}", (3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="rtsp://..., http://..., webcam:0 or a video file")
    ap.add_argument("--camera", type=int, help="use CAMERA_0N_RTSP from .env instead of --source")
    ap.add_argument("--seconds", type=float, default=10.0, help="how long to measure (default 10)")
    ap.add_argument("--out", default=str(ROOT / "runtime" / "snapshot_grid.jpg"))
    a = ap.parse_args()

    source = a.source
    if a.camera:
        source = os.environ.get(f"CAMERA_{a.camera:02d}_RTSP") or read_env_file().get(f"CAMERA_{a.camera:02d}_RTSP", "")
    if not source:
        print("No source. Use --source or --camera N (with CAMERA_0N_RTSP set in .env).")
        return 2

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")
    print(f"Opening {mask(source)} ...")
    t0 = time.monotonic()
    if source.lower().startswith("webcam:"):
        idx = source.split(":", 1)[1] or "0"
        cap = cv2.VideoCapture(int(idx) if idx.isdigit() else idx)
    elif "://" in source:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("FAILED to open the stream. Check: IP/port reachable (VPN up?), stream path, username/password "
              "(special characters like @ # : must be URL-encoded), and that RTSP/ONVIF is enabled on the NVR.")
        return 1

    frames, first_at, last = 0, None, None
    deadline = time.monotonic() + a.seconds
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        if first_at is None:
            first_at = time.monotonic() - t0
        frames, last = frames + 1, frame
    cap.release()
    if last is None:
        print("Stream opened but NO frames arrived. Possible: unsupported codec (try the sub-stream / H.264), "
              "bandwidth too low, or the NVR limits the number of viewers.")
        return 1

    h, w = last.shape[:2]
    secs = max(0.001, a.seconds)
    print(f"OK  resolution={w}x{h}  measured_fps={frames / secs:.1f}  first_frame_after={first_at:.1f}s  frames={frames}")
    if w >= 1920:
        print("Tip: this looks like a MAIN stream. For AI analytics, ask IT for the SUB stream (e.g. 640x360 or 1280x720): "
              "far less bandwidth and CPU.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, draw_grid(last))
    print(f"Saved snapshot with 0.0-1.0 grid: {a.out}")
    print("Open it and note where the door line / table rectangles go, then edit config/cameras.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
