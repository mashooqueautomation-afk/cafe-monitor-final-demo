"""Create one-time low-rate, low-resolution demo preview copies.

This script never modifies or deletes any source recording. Outputs are cached in runtime/demo_proxy,
which is intentionally separate from data/client_footage.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import Settings, load_cameras  # noqa: E402


def main() -> int:
    settings = Settings()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ERROR: ffmpeg is unavailable in the backend image; originals will not be modified.", flush=True)
        return 2

    out_dir = Path(settings.demo_proxy_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = max(1, min(8, int(round(settings.demo_proxy_fps))))
    width = max(320, min(960, int(settings.demo_proxy_width)))
    rows = []
    found = 0
    errors = 0

    print(f"Preparing cached preview videos at {width}px wide / {fps} FPS.", flush=True)
    print("Original recordings in data/ are read-only inputs for this step and will not be changed.", flush=True)

    for cam in load_cameras(settings):
        source = cam.demo_file
        if not source.is_file():
            print(f"Camera {cam.id:02d}: skipped (no recording found; optional channels are allowed).", flush=True)
            rows.append({"camera_id": cam.id, "status": "missing", "source": str(source)})
            continue
        found += 1
        target = out_dir / f"camera_{cam.id:02d}.mp4"
        if target.is_file() and target.stat().st_size > 0 and target.stat().st_mtime >= source.stat().st_mtime:
            print(f"Camera {cam.id:02d}: cached proxy is current; reusing {target.name}.", flush=True)
            rows.append({"camera_id": cam.id, "status": "reused", "proxy": target.name})
            continue

        temp = out_dir / f"camera_{cam.id:02d}.tmp.mp4"
        temp.unlink(missing_ok=True)
        vf = f"fps={fps},scale={width}:-2"
        command = [
            ffmpeg, "-hide_banner", "-y", "-i", str(source),
            "-vf", vf, "-an", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(temp),
        ]
        print(f"\nCamera {cam.id:02d}: creating proxy from {source.name} ...", flush=True)
        try:
            subprocess.run(command, check=True)
            if not temp.is_file() or temp.stat().st_size < 1024:
                raise RuntimeError("ffmpeg produced an empty/invalid proxy")
            temp.replace(target)
            print(f"Camera {cam.id:02d}: ready ({target.stat().st_size / (1024 * 1024):.1f} MB).", flush=True)
            rows.append({"camera_id": cam.id, "status": "created", "proxy": target.name,
                         "proxy_bytes": target.stat().st_size})
        except Exception as exc:
            errors += 1
            temp.unlink(missing_ok=True)
            print(f"Camera {cam.id:02d}: ERROR creating proxy: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"camera_id": cam.id, "status": "error", "error": f"{type(exc).__name__}: {exc}"})

    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "fps": fps, "width": width, "cameras": rows}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if found == 0:
        print("No source recordings were found. Add recordings to the configured data/client_footage paths.", flush=True)
        return 1
    if errors:
        print(f"WARNING: {errors} proxy conversion(s) failed. Those cameras will fall back to original recordings.", flush=True)
        return 0
    print("\nPreview cache ready. Source footage was not changed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
