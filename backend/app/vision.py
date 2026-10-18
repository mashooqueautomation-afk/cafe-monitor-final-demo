"""Optional YOLO-based inference helpers.

The app remains usable when model weights are unavailable; analytics then stay
unavailable (None) rather than inventing detections.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import cv2

log = logging.getLogger("cafe.vision")
_model = None
_model_name: str | None = None
_model_backend: str | None = None
_model_lock = threading.Lock()
_load_lock = threading.Lock()


def _openvino_dir(model_name: str) -> Path:
    p = Path(model_name)
    return p.with_name(f"{p.stem}_openvino_model") if p.suffix.lower() == ".pt" else p.with_name(f"{p.stem}_openvino_model")


def _load_model(model_name: str, auto_download: bool = False, backend: str = "auto"):
    """Load an Intel-optimized OpenVINO model when available; safely fall back to PyTorch."""
    global _model, _model_name, _model_backend
    preferred = (backend or "auto").lower()
    ov_dir = _openvino_dir(model_name)
    ov_xml = next(ov_dir.glob("*.xml"), None) if ov_dir.is_dir() else None
    chosen_path = str(ov_dir) if preferred in {"auto", "openvino"} and ov_xml else model_name
    chosen_backend = "openvino" if chosen_path != model_name else "pytorch"
    cache_key = f"{model_name}|{chosen_path}|{chosen_backend}"
    with _load_lock:
        if _model is not None and _model_name == cache_key:
            return _model, None
        try:
            from ultralytics import YOLO
            if chosen_backend == "pytorch":
                # Keep the legacy CPU path predictable if OpenVINO export was unavailable.
                try:
                    import os
                    import torch
                    torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))
                    try:
                        torch.set_num_interop_threads(max(1, int(os.getenv("TORCH_INTEROP_THREADS", "1"))))
                    except RuntimeError:
                        pass
                except Exception as thread_exc:
                    log.debug("Could not set PyTorch CPU thread limits: %s", thread_exc)
            if chosen_backend == "openvino":
                try:
                    model = YOLO(chosen_path)
                    log.info("Loaded OpenVINO model from %s", chosen_path)
                except Exception:
                    if preferred != "auto" or not Path(model_name).is_file():
                        raise
                    # A stale or partially exported OpenVINO directory must not block the known .pt model.
                    log.exception("OpenVINO model could not be loaded; falling back to PyTorch weights %s", model_name)
                    chosen_backend, chosen_path = "pytorch", model_name
                    cache_key = f"{model_name}|{chosen_path}|{chosen_backend}"
                    import os
                    import torch
                    torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))
                    try:
                        torch.set_num_interop_threads(max(1, int(os.getenv("TORCH_INTEROP_THREADS", "1"))))
                    except RuntimeError:
                        pass
                    model = YOLO(model_name)
            elif Path(model_name).is_file():
                model = YOLO(model_name)
                if preferred == "openvino":
                    log.warning("OpenVINO model directory %s is missing; using PyTorch fallback", ov_dir)
            elif auto_download and Path(model_name).name.lower().endswith(".pt"):
                model = YOLO(Path(model_name).name)
            else:
                return None, f"Model weights not found: {model_name}. Provide weights or enable YOLO_AUTO_DOWNLOAD."
            # Person counting expects COCO class 0 to mean person. Flag mismatched custom weights clearly.
            names = getattr(model, "names", {})
            class_zero = names.get(0) if isinstance(names, dict) else (names[0] if isinstance(names, (list, tuple)) and names else None)
            if class_zero is not None and str(class_zero).strip().lower() not in {"person", "people", "human"}:
                log.warning("Model class 0 is '%s', not 'person'; person/COCO analytics may be invalid. Use COCO yolov8n.pt weights.", class_zero)
            _model, _model_name, _model_backend = model, cache_key, chosen_backend
            return model, None
        except Exception as exc:
            log.warning("Could not load YOLO model %s using %s: %s", model_name, chosen_backend, exc)
            return None, f"YOLO model unavailable ({chosen_backend}): {type(exc).__name__}: {exc}"


def _tile_regions(frame, divisions: int = 2, overlap: float = 0.08):
    """Generate overlapping image tiles so distant seated people occupy more model pixels."""
    h, w = frame.shape[:2]
    divisions = max(2, int(divisions))
    tile_w = min(w, int(round(w / divisions * (1.0 + overlap))))
    tile_h = min(h, int(round(h / divisions * (1.0 + overlap))))
    xs = sorted(set([0] + [min(max(0, int(round(i * w / divisions - (tile_w - w / divisions) / 2))), w - tile_w)
                           for i in range(1, divisions)] + [max(0, w - tile_w)]))
    ys = sorted(set([0] + [min(max(0, int(round(i * h / divisions - (tile_h - h / divisions) / 2))), h - tile_h)
                           for i in range(1, divisions)] + [max(0, h - tile_h)]))
    return [(x, y, x + tile_w, y + tile_h, frame[y:y + tile_h, x:x + tile_w])
            for y in ys for x in xs]


def _suppress_duplicate_boxes(items: list[dict[str, Any]], iou_threshold: float = 0.45) -> list[dict[str, Any]]:
    """Class-aware non-maximum suppression for objects appearing in overlapping tiles."""
    ordered = sorted(items, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        x1, y1, x2, y2 = candidate["xyxy"]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        duplicate = False
        for prior in kept:
            if int(prior["class_id"]) != int(candidate["class_id"]):
                continue
            a1, b1, a2, b2 = prior["xyxy"]
            iw = max(0, min(x2, a2) - max(x1, a1))
            ih = max(0, min(y2, b2) - max(y1, b1))
            inter = iw * ih
            prior_area = max(0, a2 - a1) * max(0, b2 - b1)
            union = area + prior_area - inter
            if union > 0 and inter / union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def detect(frame, model_name: str, confidence: float = 0.35, auto_download: bool = False,
           device: str = "cpu", imgsz: int = 640, classes: list[int] | None = None,
           backend: str = "auto", tiled: bool = False) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Detect objects; optional overlapping tiles improve small-object recall on wide CCTV views.

    Tiled mode costs roughly four single-frame predictions. It is intended for the Main Hall
    only, at a low analysis rate. OpenVINO is preferred on Intel CPU; PyTorch is the fallback.
    """
    model, error = _load_model(model_name, auto_download, backend=backend)
    if model is None:
        return None, error or "YOLO model unavailable"
    try:
        with _model_lock:
            predict_device = "intel:cpu" if _model_backend == "openvino" and device == "cpu" else device
            regions = _tile_regions(frame) if tiled else [(0, 0, frame.shape[1], frame.shape[0], frame)]
            output: list[dict[str, Any]] = []
            for x_off, y_off, _x2, _y2, image in regions:
                results = model.predict(source=image, conf=confidence, verbose=False, imgsz=imgsz,
                                        device=predict_device, classes=classes, max_det=100)
                for result in results:
                    names = result.names
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    for box in boxes:
                        cls = int(box.cls.item())
                        local = [float(x) for x in box.xyxy[0].tolist()]
                        x1, y1, x2, y2 = local
                        output.append({
                            "class_id": cls,
                            "class_name": str(names.get(cls, cls) if isinstance(names, dict) else names[cls]),
                            "confidence": float(box.conf.item()),
                            "xyxy": [int(max(0, min(frame.shape[1] if i in (0, 2) else frame.shape[0], v + (x_off if i in (0, 2) else y_off)))) for i, v in enumerate(local)],
                        })
            if tiled:
                output = _suppress_duplicate_boxes(output, iou_threshold=0.45)
            return output, None
    except Exception as exc:
        log.exception("YOLO inference failed")
        return None, f"Inference failed: {type(exc).__name__}: {exc}"


def draw_detections(frame, detections: list[dict[str, Any]], extras: list[dict[str, Any]] | None = None):
    """Draw relevant detections and optional configured region/line overlays."""
    annotated = frame.copy()
    for det in detections:
        # All COCO boxes are shown; non-target classes may be noisy on busy scenes.
        x1, y1, x2, y2 = det["xyxy"]
        is_person = det["class_id"] == 0
        is_cup = det["class_name"].lower() == "cup"
        color = (55, 220, 160) if is_person else ((235, 190, 50) if is_cup else (50, 160, 220))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        if det.get("track_id") is not None:
            label += f" #{det['track_id']}"
        cv2.putText(annotated, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    for extra in extras or []:
        if extra.get("type") == "line_x":
            x = int(frame.shape[1] * float(extra["ratio"]))
            cv2.line(annotated, (x, 0), (x, frame.shape[0]), (0, 210, 255), 2)
            cv2.putText(annotated, "ENTRY / EXIT LINE", (max(8, x - 90), 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1, cv2.LINE_AA)
        elif extra.get("type") == "line_y":
            y = int(frame.shape[0] * float(extra["ratio"]))
            cv2.line(annotated, (0, y), (frame.shape[1], y), (0, 210, 255), 2)
            cv2.putText(annotated, "ENTRY / EXIT LINE", (8, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1, cv2.LINE_AA)
        elif extra.get("type") == "region":
            x1, y1, x2, y2 = extra["xyxy"]
            color = {"occupied": (80, 80, 255), "needs_check": (0, 165, 255), "empty": (90, 200, 90)}.get(extra.get("state"), (255, 185, 50))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = extra.get("label", "TABLE") + (f": {extra['state'].replace('_', ' ').upper()}" if extra.get("state") else "")
            cv2.putText(annotated, text, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return annotated
