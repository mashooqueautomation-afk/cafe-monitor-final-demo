"""Offline, CPU-conscious CCTV evidence analyzer for the configured demo recordings.

Run inside the backend container after stopping monitoring:
  python /workspace/scripts/analyze_recordings.py --interval-seconds 2

This is a review aid, not a ground-truth accuracy certificate. Tracking IDs are camera-local
and may change when people are occluded or move too far between sampled frames.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import Settings, load_cameras  # noqa: E402
from app.tracking import PersonTracker  # noqa: E402
from app.vision import detect, draw_detections  # noqa: E402
from app.workers import COCO_CUP_LIKE, ROLE_CLASSES  # noqa: E402


def make_report(out_dir: Path, camera_rows: list[dict], frame_rows: list[dict], image_paths: dict[int, list[Path]]) -> None:
    thumb_w, thumb_h, header_h = 240, 135, 25
    tiles: list[np.ndarray] = []
    for camera_id, paths in image_paths.items():
        if not paths:
            continue
        # A bounded visual overview; every analyzed sampled image is still saved individually.
        picks = sorted(set([0, len(paths) // 3, (2 * len(paths)) // 3, len(paths) - 1]))
        for idx in picks:
            path = paths[idx]
            image = cv2.imread(str(path))
            if image is None:
                continue
            image = cv2.resize(image, (thumb_w, thumb_h))
            tile = np.zeros((thumb_h + header_h, thumb_w, 3), dtype=np.uint8)
            tile[header_h:, :, :] = image
            title = f"Camera {camera_id:02d} - {path.stem.split('_t_')[-1]}s"
            cv2.putText(tile, title[:34], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 245, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
    sheet_path = out_dir / "contact_sheet.jpg"
    if tiles:
        cols = min(4, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.zeros((rows * (thumb_h + header_h), cols * thumb_w, 3), dtype=np.uint8)
        for i, tile in enumerate(tiles):
            y, x = divmod(i, cols)
            sheet[y * tile.shape[0]:(y + 1) * tile.shape[0], x * tile.shape[1]:(x + 1) * tile.shape[1]] = tile
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 75])

    table_rows = []
    for row in camera_rows:
        table_rows.append(
            "<tr>"
            f"<td>{row['camera_id']:02d} · {html.escape(row['camera_name'])}</td>"
            f"<td>{html.escape(row['status'])}</td>"
            f"<td>{row['samples']}</td><td>{row['person_detections']}</td>"
            f"<td>{row['cup_candidates']}</td><td>{row['table_zone_hits']}</td>"
            f"<td>{row['inference_seconds']:.1f}s</td>"
            "</tr>"
        )
    sheet_html = '<p>No screenshots were generated.</p>'
    if (out_dir / "contact_sheet.jpg").exists():
        sheet_html = '<img class="sheet" src="contact_sheet.jpg" alt="Representative annotated frames">'
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Café Monitor Offline Analysis</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#0b1220;color:#e7eef8;margin:24px}}h1,h2{{color:#80dff0}}p,small{{color:#b7c5d8}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #32435a;padding:8px;text-align:left}}th{{background:#15253a}}.sheet{{max-width:100%;height:auto;border:1px solid #32435a}}code{{color:#b8edb0}}.warning{{padding:12px;border:1px solid #6d5a2a;background:#292315}}
</style></head><body>
<h1>Café Monitor · Offline Recording Analysis</h1>
<p>Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}</p>
<div class="warning"><b>Interpretation:</b> these are sampled-frame model outputs, not verified accuracy. A frame-level person count is not the number of unique customers. Tracking IDs are camera-local and can switch under occlusion. Validate entry/exit and table states against a manual count before presenting them as measured results.</div>
<h2>Camera Summary</h2><table><thead><tr><th>Camera</th><th>Status</th><th>Frames sampled</th><th>Person detections</th><th>Cup-like detections</th><th>Table-zone hits</th><th>Inference time</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Annotated Contact Sheet</h2>{sheet_html}
<h2>Files</h2><p><code>frames.csv</code> contains one row per sampled frame. <code>detections.csv</code> contains each detected object and its box/track ID/center position. Annotated images include short trails joining sampled positions for the same temporary track ID. Individual images are under <code>screenshots/</code>.</p>
</body></html>"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=float, default=2.0, help="Video-time gap between analyzed frames; minimum 0.5 seconds")
    parser.add_argument("--confidence", type=float, default=None, help="Override YOLO confidence threshold")
    parser.add_argument("--max-width", type=int, default=640, help="Maximum width of saved evidence images")
    args = parser.parse_args()
    if args.interval_seconds < 0.5:
        parser.error("--interval-seconds must be at least 0.5 on the low-spec profile")

    settings = Settings()
    confidence = settings.yolo_confidence if args.confidence is None else args.confidence
    cams = load_cameras(settings)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "runtime" / "offline_analysis" / run_id
    screenshots = out_dir / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    all_frames: list[dict] = []
    all_detections: list[dict] = []
    camera_rows: list[dict] = []
    image_paths: dict[int, list[Path]] = {}

    print(f"Output directory: {out_dir}", flush=True)
    print(f"Sampling every {args.interval_seconds:g}s of video; image size <= {args.max_width}px; device={settings.yolo_device}", flush=True)
    print("Stop dashboard monitoring before starting this job to avoid competing for CPU.", flush=True)

    for spec in cams:
        path = spec.demo_file
        if not path.is_file():
            print(f"Camera {spec.id:02d}: SKIPPED (recording not found: {path})", flush=True)
            camera_rows.append({"camera_id": spec.id, "camera_name": spec.name, "status": "missing recording", "samples": 0,
                                "person_detections": 0, "cup_candidates": 0, "table_zone_hits": 0, "inference_seconds": 0.0})
            continue
        if spec.role not in ROLE_CLASSES:
            print(f"Camera {spec.id:02d}: SKIPPED (analytics role is not configured)", flush=True)
            camera_rows.append({"camera_id": spec.id, "camera_name": spec.name, "status": "analytics not configured", "samples": 0,
                                "person_detections": 0, "cup_candidates": 0, "table_zone_hits": 0, "inference_seconds": 0.0})
            continue

        folder = screenshots / f"camera_{spec.id:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        image_paths[spec.id] = []
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"Camera {spec.id:02d}: ERROR (video could not be opened)", flush=True)
            camera_rows.append({"camera_id": spec.id, "camera_name": spec.name, "status": "video open error", "samples": 0,
                                "person_detections": 0, "cup_candidates": 0, "table_zone_hits": 0, "inference_seconds": 0.0})
            cap.release()
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fps = min(max(fps, 1.0), 60.0)
        frame_step = max(1, int(round(fps * args.interval_seconds)))
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        tracker = PersonTracker(gate=1.8, max_age_seconds=max(5.0, args.interval_seconds * 3.0))
        frame_no = 0
        sample_no = 0
        person_total = cup_total = table_hits = 0
        inference_seconds = 0.0
        status = "completed"
        # Keep a short, camera-local trajectory tail for each temporary tracker ID.
        # This illustrates movement across sampled frames; it is not persistent identity.
        track_trails: dict[int, list[tuple[int, int]]] = {}
        print(f"Camera {spec.id:02d}: analyzing {path.name} ({total_frames / fps / 60:.1f} min)", flush=True)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_no % frame_step == 0:
                    video_sec = frame_no / fps
                    t0 = time.monotonic()
                    person_related = 0 in ROLE_CLASSES[spec.role]
                    frame_confidence = min(confidence, settings.person_detection_confidence) if person_related else confidence
                    frame_imgsz = max(settings.yolo_imgsz, settings.person_detection_imgsz) if person_related else settings.yolo_imgsz
                    detections, error = detect(frame, settings.yolo_model, frame_confidence,
                                               settings.yolo_auto_download, device=settings.yolo_device,
                                               imgsz=frame_imgsz, classes=ROLE_CLASSES[spec.role], backend=settings.yolo_backend,
                                               tiled=(spec.role == "people_count" and settings.person_tiled_inference))
                    inference_seconds += time.monotonic() - t0
                    if detections is None:
                        status = f"model error: {error}"
                        print(f"  Inference error: {error}", flush=True)
                        break

                    people = [d for d in detections if d["class_id"] == 0 or d["class_name"].lower() == "person"]
                    points = [((d["xyxy"][0] + d["xyxy"][2]) / 2.0, float(d["xyxy"][3]),
                               max(d["xyxy"][2] - d["xyxy"][0], d["xyxy"][3] - d["xyxy"][1])) for d in people]
                    tracks = tracker.update(points, video_sec)
                    active_track_ids: set[int] = set()
                    for det, track in zip(people, tracks):
                        det["track_id"] = track.id
                        active_track_ids.add(track.id)
                        cx = int(round((det["xyxy"][0] + det["xyxy"][2]) / 2))
                        cy = int(round((det["xyxy"][1] + det["xyxy"][3]) / 2))
                        trail = track_trails.setdefault(track.id, [])
                        trail.append((cx, cy))
                        if len(trail) > 12:
                            del trail[:-12]
                    person_total += len(people)
                    cup_total += sum(1 for d in detections if d["class_id"] in COCO_CUP_LIKE or d["class_name"].lower() in {"cup", "wine glass"})
                    frame_extras: list[dict] = []
                    if spec.role == "table_status":
                        h, w = frame.shape[:2]
                        for index, region in enumerate(spec.table_regions):
                            x1, y1, x2, y2 = (int(float(region[k]) * dim) for k, dim in (("x1", w), ("y1", h), ("x2", w), ("y2", h)))
                            occupied = any(x1 <= (d["xyxy"][0] + d["xyxy"][2]) / 2 <= x2 and
                                           y1 <= (d["xyxy"][1] + d["xyxy"][3]) / 2 <= y2 for d in people)
                            if occupied:
                                table_hits += 1
                            frame_extras.append({"type": "region", "xyxy": (x1, y1, x2, y2),
                                                 "label": region.get("name", f"Table {index + 1}"),
                                                 "state": "occupied" if occupied else "empty"})
                    annotated = draw_detections(frame, detections, frame_extras)
                    # Draw movement paths after boxes, joining the sampled positions of the same
                    # temporary tracker ID. Short trails avoid retaining misleading old positions.
                    for track_id, trail in track_trails.items():
                        # Only draw a trail when that ID is detected in this sampled frame;
                        # never leave a marker behind for a person who has vanished.
                        if track_id not in active_track_ids:
                            continue
                        if len(trail) >= 2:
                            cv2.polylines(annotated, [np.asarray(trail, dtype=np.int32)], False,
                                          (255, 180, 40), 2, cv2.LINE_AA)
                        if trail:
                            cv2.circle(annotated, trail[-1], 4, (255, 180, 40), -1, cv2.LINE_AA)
                    if annotated.shape[1] > args.max_width:
                        annotated = cv2.resize(annotated, (args.max_width, int(annotated.shape[0] * args.max_width / annotated.shape[1])))
                    name = f"cam_{spec.id:02d}_t_{video_sec:08.2f}.jpg"
                    img_path = folder / name
                    if not cv2.imwrite(str(img_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]):
                        print(f"  Warning: could not save {img_path.name}", flush=True)
                    else:
                        image_paths[spec.id].append(img_path)
                    frame_record = {"camera_id": spec.id, "camera_name": spec.name, "video_time_seconds": round(video_sec, 3),
                                    "frame_number": frame_no, "screenshot": str(img_path.relative_to(out_dir)).replace('\\\\', '/'),
                                    "detected_people": len(people), "cup_candidates": sum(1 for d in detections if d["class_id"] in COCO_CUP_LIKE or d["class_name"].lower() in {"cup", "wine glass"}),
                                    "table_zones_with_people": sum(1 for x in frame_extras if x.get("state") == "occupied")}
                    all_frames.append(frame_record)
                    for det in detections:
                        x1, y1, x2, y2 = det["xyxy"]
                        all_detections.append({"camera_id": spec.id, "camera_name": spec.name, "video_time_seconds": round(video_sec, 3),
                            "frame_number": frame_no, "class_id": det["class_id"], "class_name": det["class_name"],
                            "confidence": round(det["confidence"], 4), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "center_x": round((x1 + x2) / 2, 2), "center_y": round((y1 + y2) / 2, 2),
                            "track_id": det.get("track_id", ""), "screenshot": frame_record["screenshot"]})
                    sample_no += 1
                    if sample_no % 10 == 0:
                        print(f"  {sample_no} frames sampled; latest video time {video_sec:.1f}s", flush=True)

                frame_no += 1
                for _ in range(frame_step - 1):
                    if not cap.grab():
                        frame_no = int(total_frames) if total_frames else frame_no
                        break
                    frame_no += 1
                else:
                    continue
                break
        finally:
            cap.release()

        camera_rows.append({"camera_id": spec.id, "camera_name": spec.name, "status": status, "samples": sample_no,
                            "person_detections": person_total, "cup_candidates": cup_total, "table_zone_hits": table_hits,
                            "inference_seconds": round(inference_seconds, 2)})
        print(f"Camera {spec.id:02d}: {status}; {sample_no} samples; {person_total} person detections; {cup_total} cup-like detections", flush=True)

    def write_csv(name: str, rows: list[dict]) -> None:
        file_path = out_dir / name
        fields = list(rows[0].keys()) if rows else []
        with file_path.open("w", newline="", encoding="utf-8-sig") as f:
            if fields:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

    write_csv("frames.csv", all_frames)
    write_csv("detections.csv", all_detections)
    summary = {"created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "interval_seconds": args.interval_seconds, "model": settings.yolo_model,
               "imgsz": settings.person_detection_imgsz, "confidence": confidence, "backend": settings.yolo_backend, "device": settings.yolo_device,
               "cameras": camera_rows, "sampled_frames_total": len(all_frames), "detections_total": len(all_detections),
               "notes": ["Sampled-frame detections are not ground truth.", "Person IDs are camera-local and can switch.",
                         "Cup candidates use generic COCO classes and are not a café-specific cup classifier.",
                         "Table hits are geometric frame-level estimates; no multi-frame table-state smoothing is applied in this offline report."]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_report(out_dir, camera_rows, all_frames, image_paths)
    print("\nAnalysis finished.", flush=True)
    print(f"Report: {out_dir / 'index.html'}", flush=True)
    print(f"Frames CSV: {out_dir / 'frames.csv'}", flush=True)
    print(f"Detections CSV: {out_dir / 'detections.csv'}", flush=True)
    if not any(r["samples"] for r in camera_rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
