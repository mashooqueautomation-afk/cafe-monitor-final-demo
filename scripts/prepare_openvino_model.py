"""Prepare a reusable Intel OpenVINO export for the COCO YOLO nano model.

The export lives in the named Docker model volume, not in the project data directory.
If conversion/runtime setup fails, the app remains able to fall back to PyTorch.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def main() -> int:
    from app.config import Settings

    settings = Settings()
    os.environ.setdefault("TORCH_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_INTEROP_THREADS", "1")
    raw_path = Path(settings.yolo_model).expanduser()
    weights = raw_path if raw_path.is_absolute() else Path.cwd() / raw_path
    export_dir = weights.with_name(f"{weights.stem}_openvino_model")
    xml_files = list(export_dir.glob("*.xml")) if export_dir.is_dir() else []
    exported_recently = bool(xml_files) and weights.is_file() and max(x.stat().st_mtime for x in xml_files) >= weights.stat().st_mtime
    if exported_recently:
        print(f"OpenVINO export already exists: {export_dir}", flush=True)
        return 0

    try:
        from ultralytics import YOLO
        import numpy as np

        if weights.is_file():
            model = YOLO(str(weights))
        elif settings.yolo_auto_download and weights.suffix.lower() == ".pt":
            # The container working directory is /workspace/models, backed by the persistent yolo_models volume.
            model = YOLO(weights.name)
            weights = Path.cwd() / weights.name
        else:
            raise FileNotFoundError(f"YOLO weights are not present: {weights}")

        names = getattr(model, "names", {})
        person_label = names.get(0) if isinstance(names, dict) else (names[0] if names else None)
        if str(person_label).lower() != "person":
            raise RuntimeError(f"Expected COCO class 0='person', found {person_label!r}; use COCO yolov8n.pt weights")

        print(f"Exporting {weights.name} to OpenVINO at image size {settings.person_detection_imgsz} ...", flush=True)
        model.export(format="openvino", imgsz=settings.person_detection_imgsz, dynamic=False, device="cpu")
        if not export_dir.is_dir() or not list(export_dir.glob("*.xml")):
            raise RuntimeError(f"Exporter did not create the expected directory: {export_dir}")

        print("Validating the exported model with a warm-up frame ...", flush=True)
        optimized = YOLO(str(export_dir))
        sample = np.zeros((640, 640, 3), dtype=np.uint8)
        optimized.predict(sample, imgsz=settings.person_detection_imgsz, conf=0.12,
                          device="intel:cpu", verbose=False)
        print(f"OpenVINO model is ready: {export_dir}", flush=True)
        return 0
    except Exception as exc:
        message = f"OpenVINO preparation failed; app will use PyTorch fallback. {type(exc).__name__}: {exc}"
        print("WARNING: " + message, flush=True)
        status_path = ROOT / "runtime" / "openvino_export_status.txt"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(message + "\n", encoding="utf-8")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
