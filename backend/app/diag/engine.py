"""Diagnostics engine: finds out WHY the dashboard is slow / shows 0, using the same code paths as the monitor.

Nothing here changes the running monitoring session. Heavy work runs as one background job at a time.
All "detect" calls go through `Engine.detect`, so tests can inject a fake detector.
"""
from __future__ import annotations

import base64
import inspect
import os
import statistics
import threading
import time
import traceback
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .. import vision
from ..config import CameraSpec, Settings
from ..tracking import LineCounter

CUP_LIKE = {"cup", "wine glass", "bottle", "bowl", "vase"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _b64_jpeg(img, width: int = 420, quality: int = 70) -> str:
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode() if ok else ""


def _check(level: str, title: str, detail: str, fix: str = "") -> dict:
    return {"level": level, "title": title, "detail": detail, "fix": fix}   # level: ok | warn | fail | info


def _read_cgroup() -> dict:
    out: dict[str, Any] = {}
    try:
        raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        out["mem_limit_mb"] = None if raw == "max" else round(int(raw) / 1048576)
    except Exception:
        pass
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        out["cpu_limit_cores"] = None if quota == "max" else round(int(quota) / int(period), 2)
    except Exception:
        pass
    return out


class JobBusy(Exception):
    pass


class Engine:
    def __init__(self, settings: Settings, cameras: list[CameraSpec], detect_fn: Callable | None = None,
                 backend_url: str | None = None):
        self.settings, self.cameras = settings, cameras
        self._detect = detect_fn or vision.detect
        self.backend_url = backend_url if backend_url is not None else os.environ.get("DIAG_BACKEND_URL", "http://backend:8000")
        self.jobs: dict[str, dict] = {}
        self._job_lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def model_path(self) -> str | None:
        name = self.settings.yolo_model
        ov_dir = Path(name).with_name(f"{Path(name).stem}_openvino_model")
        for cand in (name, f"/workspace/models/{Path(name).name}"):
            if Path(cand).is_file():
                return cand
        if self.settings.yolo_backend in {"auto", "openvino"} and ov_dir.is_dir() and next(ov_dir.glob("*.xml"), None):
            return str(ov_dir)
        return None

    def detect(self, frame, imgsz: int, classes: list[int] | None, conf: float | None = None,
               tiled: bool = False):
        path = self.model_path()
        if path is None:
            return None, f"Model weights not found ({self.settings.yolo_model}); diagnostics never downloads weights"
        kwargs = {"device": self.settings.yolo_device, "imgsz": imgsz, "classes": classes,
                  "backend": self.settings.yolo_backend, "tiled": tiled}
        # Test fakes and third-party detectors may implement only the legacy signature.
        try:
            sig = inspect.signature(self._detect)
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not accepts_kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        except (TypeError, ValueError):
            pass
        return self._detect(frame, path, self.settings.person_detection_confidence if conf is None else conf,
                            False, **kwargs)

    def cam(self, cam_id: int) -> CameraSpec:
        for c in self.cameras:
            if c.id == cam_id:
                return c
        raise KeyError(f"camera {cam_id} not found")

    @staticmethod
    def _sample_frames(path: Path, n: int) -> tuple[list[tuple[float, np.ndarray]], dict]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Video cannot be opened: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(fps, 2), "frames": total, "duration_sec": round(total / fps, 1) if fps else None}
        frames = []
        for k in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / n))
            ok, fr = cap.read()
            if ok:
                frames.append((round((k + 0.5) * total / n / fps, 1), fr))
        cap.release()
        return frames, info

    # ------------------------------------------------------------------ health checks (instant)
    def health_checks(self) -> list[dict]:
        s, out = self.settings, []
        # model + ultralytics
        mp = self.model_path()
        out.append(_check("ok", "YOLO weights available", mp) if mp else
                   _check("fail", "YOLO weights missing", f"'{s.yolo_model}' was not found in /workspace/models.",
                          "Start the main dashboard once with internet access to download yolov8n.pt, or copy the weights file into the model volume."))
        try:
            import ultralytics  # noqa: F401
            out.append(_check("ok", "Ultralytics installed", getattr(ultralytics, "__version__", "")))
        except Exception as e:  # pragma: no cover - depends on image
            out.append(_check("fail", "Ultralytics import failed", str(e), "docker compose up -d --build backend diagnostics"))
        # machine
        cg, cores = _read_cgroup(), os.cpu_count() or 1
        try:
            import psutil
            vm = psutil.virtual_memory()
            ram = f"host RAM {round(vm.total / 2**30, 1)} GB, free {round(vm.available / 2**30, 1)} GB"
        except Exception:
            ram = "RAM info unavailable"
        out.append(_check("info", "Machine / container", f"{cores} CPU threads visible; {ram}; container limits: {cg or 'unknown'}"))
        if cg.get("mem_limit_mb") and cg["mem_limit_mb"] < 2500:
            out.append(_check("warn", "Container memory is low", f"Limit {cg['mem_limit_mb']} MB.", "CAFE_MEMORY_LIMIT=4g (.env)"))
        # fps / size settings
        if s.entrance_fps < 4:
            out.append(_check("warn", f"ENTRANCE_FPS={s.entrance_fps} is low",
                              "People may cross the doorway in about 2–3 seconds; at 1–2 AI checks per second, tracking may miss crossings or count them incorrectly.",
                              f"Select only Camera 07 (focus mode uses {s.focus_entrance_fps} FPS), or set ENTRANCE_FPS to 4 or higher only if CPU capacity permits."))
        else:
            out.append(_check("ok", f"ENTRANCE_FPS={s.entrance_fps}", "This is a workable range for counting; validate it against a manual count."))
        if s.yolo_imgsz <= 320:
            out.append(_check("warn", f"YOLO_IMGSZ={s.yolo_imgsz} is small",
                              "Frames are 960x1088; at 320 input size, distant people and cups occupy fewer pixels and may be missed.",
                              "Use the Camera tab test to compare 320 with 640, then try YOLO_IMGSZ=416 or 480 if CPU capacity allows."))
        # mode / live sources
        configured = [c.id for c in self.cameras if s.rtsp_url(c.id)]
        if s.default_mode == "live" and not configured:
            out.append(_check("fail", "DEFAULT_MODE=live but no RTSP URL is configured",
                              "In Live Mode, every camera without a configured RTSP URL will remain offline.",
                              "For recorded-file testing, set DEFAULT_MODE=demo in .env."))
        else:
            out.append(_check("info", f"DEFAULT_MODE={s.default_mode}", f"Live sources set: {configured or 'none'}"))
        # per camera
        for c in self.cameras:
            title = f"Cam {c.id:02d} {c.name}"
            p = c.demo_file
            if not str(p) or str(p) == "." or not p.is_file():
                out.append(_check("fail", f"{title}: demo video file is missing", f"{p}", "Check the folder and exact filename. Linux paths are case-sensitive."))
                continue
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                out.append(_check("fail", f"{title}: video could not be opened", f"{p}", "The video may be corrupt or use an unsupported codec. Convert it to MP4 with H.264 video using FFmpeg."))
                continue
            w, h, fps, n = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 0, int(cap.get(7))
            cap.release()
            out.append(_check("ok", f"{title}: video OK", f"{w}x{h} @ {fps:.1f} fps, {n} frames (~{n / fps:.0f}s) - {p.name}" if fps else f"{w}x{h}, {n} frames"))
            if c.role == "none":
                out.append(_check("info", f"{title}: analytics role not configured",
                                  "This camera is intentionally configured for video only. AI analytics will not run on it; this is not an application error."))
            if c.role == "table_status" and not c.table_regions:
                out.append(_check("warn", f"{title}: table regions are empty", "An empty table_regions list prevents occupied/empty table estimates.",
                                  "Create regions with scripts/calibrate_regions.py, then add them to config/cameras.yaml."))
            if c.role == "entry_exit":
                out.append(_check("info", f"{title}: counting line", f"Configured line: {c.line_orientation} at {c.line_x_ratio}; entry direction = {c.entry_direction}. Compare alternatives in the Entrance Test tab."))
        # is the main backend busy?
        if self.backend_url:
            try:
                import json
                with urllib.request.urlopen(self.backend_url.rstrip("/") + "/api/monitoring/status", timeout=2) as r:
                    st = json.loads(r.read().decode())
                if st.get("running"):
                    out.append(_check("warn", "The main dashboard is currently monitoring",
                                      f"Cameras {st.get('selected_camera_ids') or 'all'} are being analyzed. Diagnostics timing results may be slower because CPU resources are shared.",
                                      "Stop monitoring on the main dashboard before running the benchmark."))
                else:
                    out.append(_check("ok", "Main dashboard is idle", "Benchmark results will be less affected by concurrent processing."))
            except Exception:
                out.append(_check("info", "The main backend did not respond", f"{self.backend_url} - this is expected if the diagnostics page is running without the main backend."))
        return out

    # ------------------------------------------------------------------ jobs
    def start_job(self, kind: str, params: dict) -> str:
        runners = {"benchmark": self.run_benchmark, "camera": self.run_camera, "cups": self.run_cups, "entrance": self.run_entrance}
        if kind not in runners:
            raise KeyError(kind)
        if not self._job_lock.acquire(blocking=False):
            raise JobBusy("Another diagnostics job is running; wait for it to finish.")
        jid = uuid.uuid4().hex[:10]
        job = {"id": jid, "kind": kind, "status": "running", "progress": 0.0, "message": "Starting...", "result": None,
               "error": None, "started": time.time(), "finished": None}
        self.jobs[jid] = job

        def target():
            try:
                job["result"] = runners[kind](job, **params)
                job["status"] = "done"
            except Exception as e:
                job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
                job["trace"] = traceback.format_exc()[-1500:]
            finally:
                job["progress"], job["finished"] = 1.0, time.time()
                self._job_lock.release()

        threading.Thread(target=target, daemon=True, name=f"diag-{kind}").start()
        return jid

    @staticmethod
    def _prog(job: dict, frac: float, msg: str) -> None:
        job["progress"], job["message"] = round(min(0.99, frac), 3), msg

    # ------------------------------------------------------------------ benchmark: can this machine cope?
    def run_benchmark(self, job: dict, sizes: tuple[int, ...] = (320, 480, 640)) -> dict:
        s = self.settings
        cams = [c for c in self.cameras if c.demo_file.is_file()]
        if not cams:
            raise RuntimeError("No demo video files were found. Add sample.mp4 to a camera folder under data/client_footage/.")
        ref = next((c for c in cams if c.role == "people_count"), cams[0])
        # decode + dashboard-encode cost
        self._prog(job, 0.05, "Measuring video decoding speed...")
        dec = []
        for c in cams[:3]:
            cap = cv2.VideoCapture(str(c.demo_file)); n = 0; t0 = time.perf_counter()
            while n < 60:
                ok, _ = cap.read()
                if not ok:
                    break
                n += 1
            if n:
                dec.append((time.perf_counter() - t0) * 1000 / n)
            cap.release()
        decode_ms = statistics.median(dec) if dec else 0.0
        frames, info = self._sample_frames(ref.demo_file, 3)
        t0 = time.perf_counter()
        for _, fr in frames * 5:
            small = cv2.resize(fr, (960, int(fr.shape[0] * 960 / fr.shape[1]))) if fr.shape[1] > 960 else fr
            cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, s.jpeg_quality])
        encode_ms = (time.perf_counter() - t0) * 1000 / max(1, len(frames) * 5)
        # inference speed per image size
        results = {}
        _, probe = frames[len(frames) // 2]
        for i, size in enumerate(sizes):
            self._prog(job, 0.15 + 0.7 * i / len(sizes), f"Measuring YOLO speed at image size {size}; the model may load on the first run...")
            dets, err = self.detect(probe, size, [0], tiled=(ref.role == "people_count" and self.settings.person_tiled_inference))
            if dets is None:
                raise RuntimeError(err)
            times = []
            for _ in range(5):
                t0 = time.perf_counter(); self.detect(probe, size, [0], tiled=(ref.role == "people_count" and self.settings.person_tiled_inference)); times.append((time.perf_counter() - t0) * 1000)
            results[size] = round(statistics.median(times), 1)
        cores = max(1, (_read_cgroup().get("cpu_limit_cores") or os.cpu_count() or 1))
        base_size = s.yolo_imgsz if s.yolo_imgsz in results else min(results, key=lambda k: abs(k - s.yolo_imgsz))

        def fps_for(c: CameraSpec, focus: bool, entrance_override: float | None = None) -> float:
            if c.role == "none" or (c.role == "table_status" and not c.table_regions):
                return 0.0
            if c.role == "entry_exit":
                return entrance_override or (max(s.entrance_fps, s.focus_entrance_fps) if focus else s.entrance_fps)
            return max(s.analytics_fps, s.focus_analytics_fps) if focus else s.analytics_fps

        def scenario(name: str, ids: list[int], size: int, entrance_override: float | None = None) -> dict:
            sel = [c for c in self.cameras if c.id in ids]
            focus = len(sel) <= s.focus_max_cameras < len(self.cameras)
            inf_per_sec = sum(fps_for(c, focus, entrance_override) for c in sel)
            clip_fps = info["fps"] or 25.0
            decode_cpu_s = len(sel) * (clip_fps * decode_ms + s.display_fps * encode_ms) / 1000.0
            inf_cpu_s = inf_per_sec * results[size] / 1000.0 * cores        # inference keeps all cores busy while it runs
            util = (decode_cpu_s + inf_cpu_s) / cores
            return {"name": name, "cameras": ids, "imgsz": size, "ai_checks_per_sec": round(inf_per_sec, 2),
                    "cpu_use_percent": round(util * 100), "verdict": "OK" if util < 0.7 else ("TIGHT" if util <= 1.0 else "OVERLOAD"),
                    "inference_busy_percent": round(min(999, inf_per_sec * results[size] / 10))}

        every = [c.id for c in self.cameras]
        ent = next((c.id for c in self.cameras if c.role == "entry_exit"), every[-1])
        ppl = next((c.id for c in self.cameras if c.role == "people_count"), every[0])
        cup = next((c.id for c in self.cameras if c.role == "cup_detection"), every[0])
        scenarios = [scenario(f"All {len(every)} cameras (current configuration)", every, base_size),
                     scenario(f"Camera {ent:02d} entrance only (focus mode)", [ent], base_size),
                     scenario(f"Camera {ent:02d} entrance only, larger image size ({max(results)})", [ent], max(results)),
                     scenario(f"Camera {ppl:02d} main hall only (focus mode)", [ppl], max(results)),
                     scenario("Cameras %02d + %02d + %02d together" % (ppl, cup, ent), [ppl, cup, ent], base_size)]
        return {"cpu_threads": cores, "decode_ms_per_frame": round(decode_ms, 2), "encode_ms_per_frame": round(encode_ms, 2),
                "clip": info, "inference_ms_by_imgsz": results, "scenarios": scenarios,
                "note": "This is an estimate based on video decoding and AI CPU time. Actual load may differ; use OK/TIGHT/OVERLOAD as a guide and validate on this PC."}

    # ------------------------------------------------------------------ crowd / video analysis for one camera
    def run_camera(self, job: dict, camera_id: int, samples: int = 8, sizes: tuple[int, ...] = (320, 640)) -> dict:
        c = self.cam(camera_id)
        if not c.demo_file.is_file():
            raise RuntimeError(f"Demo video file is missing: {c.demo_file}")
        frames, info = self._sample_frames(c.demo_file, samples)
        per_size: dict[int, dict] = {}
        thumbs: list[dict] = []
        cup_counts: list[int] = []
        step, total = 0, len(frames) * (len(sizes) + (1 if c.role == "cup_detection" else 0))
        heights: list[float] = []
        for size in sizes:
            counts = []
            for i, (t, fr) in enumerate(frames):
                self._prog(job, step / total, f"Cam {c.id}: imgsz={size}, frame {i + 1}/{len(frames)}")
                step += 1
                dets, err = self.detect(fr, size, [0], conf=self.settings.person_detection_confidence,
                                        tiled=(c.role == "people_count" and self.settings.person_tiled_inference))
                if dets is None:
                    raise RuntimeError(err)
                counts.append(len(dets))
                if size == max(sizes):
                    heights += [(d["xyxy"][3] - d["xyxy"][1]) / fr.shape[0] for d in dets]
                    if i % max(1, len(frames) // 4) == 0 and len(thumbs) < 4:
                        thumbs.append({"t": t, "persons": len(dets), "image": _b64_jpeg(vision.draw_detections(fr, dets))})
            per_size[size] = {"counts": counts, "mean": round(statistics.mean(counts), 1) if counts else 0, "max": max(counts or [0])}
        if c.role == "cup_detection":
            for i, (t, fr) in enumerate(frames):
                self._prog(job, step / total, f"Camera {c.id}: checking cup/glass candidates {i + 1}/{len(frames)}")
                step += 1
                dets, err = self.detect(fr, max(sizes), [40, 41])
                if dets is None:
                    raise RuntimeError(err)
                cup_counts.append(len(dets))
                if i == len(frames) // 2:
                    thumbs.append({"t": t, "persons": None, "cups": len(dets), "image": _b64_jpeg(vision.draw_detections(fr, dets), 520)})
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _, f in frames]
        brightness = round(float(np.mean([g.mean() for g in gray])), 1)
        sharp = round(float(np.mean([cv2.Laplacian(cv2.resize(g, (480, int(g.shape[0] * 480 / g.shape[1]))), cv2.CV_64F).var() for g in gray])), 1)
        lo, hi = min(sizes), max(sizes)
        mx = per_size[hi]["max"]
        density = "low (mostly empty)" if mx <= 2 else ("moderate" if mx <= 6 else "high (crowded)")
        med_h = round(statistics.median(heights) * 100, 1) if heights else None
        findings = [f"Maximum people visible at one time in this clip: {mx} (imgsz={hi}) -> {density}. "
                    "YOLO processing time depends mainly on frame size rather than the number of people, so crowd size alone is not usually the main cause of CPU slowdown."]
        gap = per_size[hi]["mean"] - per_size[lo]["mean"]
        if gap >= 1:
            findings.append(f"At imgsz={lo}, the model detects {gap:.1f} fewer people on average than at imgsz={hi}. Smaller inputs can miss people; increasing YOLO_IMGSZ may improve recall but uses more CPU.")
        elif per_size[hi]["mean"] == 0:
            findings.append("No people were detected at any tested input size. If people are visible, check confidence, object size, lighting, and camera angle before attributing the issue to CPU performance.")
        if med_h is not None and med_h < 12:
            findings.append(f"People occupy little image area (median box height is about {med_h}% of the frame). Distant people may require a larger input size or a better camera view.")
        if brightness < 70:
            findings.append(f"The video is dark (brightness {brightness}/255); detections may be less reliable.")
        if sharp < 40:
            findings.append(f"The video appears blurry (sharpness {sharp}); detections may be less reliable.")
        if c.role == "cup_detection":
            findings.append(f"Cup/glass candidate boxes across {len(cup_counts)} sampled frames: {cup_counts}. Compare these counts with a manual review; "
                            "reliable classification of the client’s takeaway and ceramic cups requires a custom model.")
        return {"camera": {"id": c.id, "name": c.name, "role": c.role}, "video": info, "sizes": per_size, "sample_times": [t for t, _ in frames],
                "density": density, "median_person_height_percent": med_h, "brightness": brightness, "sharpness": sharp,
                "cup_boxes_per_frame": cup_counts, "findings": findings, "thumbnails": thumbs}

    # ------------------------------------------------------------------ the client's cup photos
    def run_cups(self, job: dict, imgsz: int = 640, conf: float = 0.25) -> dict:
        raw = Path(os.environ.get("DIAG_CUP_DIR", "/workspace/data/cup_dataset/raw"))
        if not raw.is_dir():
            raw = Path(__file__).resolve().parents[3] / "data" / "cup_dataset" / "raw"
        files = sorted(p for p in raw.glob("*") if p.suffix.lower() in IMG_EXT)
        if not files:
            raise RuntimeError(f"No image files were found in {raw}. Add JPG, JPEG, PNG, or WEBP images to this folder.")
        rows, found_any = [], 0
        for i, f in enumerate(files):
            self._prog(job, i / len(files), f"Photo {i + 1}/{len(files)}")
            img = cv2.imread(str(f))
            if img is None:
                rows.append({"file": f.name, "error": "The image could not be opened"}); continue
            dets, err = self.detect(img, imgsz, None, conf)
            if dets is None:
                raise RuntimeError(err)
            classes: dict[str, int] = {}
            for d in dets:
                classes[d["class_name"]] = classes.get(d["class_name"], 0) + 1
            cups = sum(v for k, v in classes.items() if k in CUP_LIKE)
            found_any += 1 if cups else 0
            rows.append({"file": f.name, "size": f"{img.shape[1]}x{img.shape[0]}", "cup_like_boxes": cups, "classes": classes,
                         "image": _b64_jpeg(vision.draw_detections(img, [d for d in dets if d["class_name"] in CUP_LIKE]), 360)})
        return {"photos": len(files), "photos_with_cup_like_detection": found_any, "imgsz": imgsz, "conf": conf, "rows": rows,
                "note": "This is a general COCO model. It detects classes such as cup, wine glass, bottle, and bowl; it does not distinguish ceramic vs glass vs takeaway, size, or brand. "
                        "Reference photos differ significantly from real CCTV viewpoints. A custom model requires annotated frames from the actual cup camera."}

    # ------------------------------------------------------------------ entrance counting simulator
    def run_entrance(self, job: dict, imgsz: int = 480, sim_fps: float = 5.0, expected_in: int | None = None,
                     expected_out: int | None = None) -> dict:
        c = next((x for x in self.cameras if x.role == "entry_exit"), None)
        if c is None or not c.demo_file.is_file():
            raise RuntimeError("No configured entrance camera or entrance demo video was found. Add the Camera 07 sample.mp4 file.")
        cap = cv2.VideoCapture(str(c.demo_file))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(fps / sim_fps))
        samples: list[tuple[float, list[list[float]]]] = []
        mid_frame, w, h = None, int(cap.get(3)), int(cap.get(4))
        idx = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if idx % step == 0:
                self._prog(job, 0.9 * idx / max(1, total), f"Running entrance inference ({idx}/{total} frames)")
                dets, err = self.detect(fr, imgsz, [0], conf=self.settings.person_detection_confidence)
                if dets is None:
                    raise RuntimeError(err)
                samples.append((idx / fps, [d["xyxy"] for d in dets]))
                if mid_frame is None and idx >= total // 2:
                    mid_frame = fr.copy()
            idx += 1
        cap.release()
        eff_fps = fps / step
        s = self.settings
        candidates = [("CURRENT config", c.line_orientation, c.entry_direction, c.line_x_ratio)]
        for r in (0.30, 0.35, 0.40, 0.45):
            candidates.append(("horizontal", "horizontal", "top_to_bottom", r))
        for r in (0.60, 0.85):
            candidates.append(("vertical", "vertical", "right_to_left", r))
        rates = sorted({r for r in (1.0, 2.0, 4.0, round(eff_fps, 1)) if r <= eff_fps + 1e-6})
        table = []
        trails: dict[int, list[tuple[float, float]]] = {}
        for label, orient, direction, ratio in candidates:
            for rate in rates:
                k = max(1, round(eff_fps / rate))
                lc = LineCounter(orient, direction, ratio, band_ratio=s.line_band_ratio, min_hits=s.counter_min_hits,
                                 cooldown_seconds=s.counter_cooldown_seconds)
                for j, (t, boxes) in enumerate(samples):
                    if j % k:
                        continue
                    _, ids = lc.update(boxes, w, h, t)
                    if rate == rates[-1] and label == "CURRENT config":
                        for b, tid in zip(boxes, ids):
                            trails.setdefault(tid, []).append(((b[0] + b[2]) / 2, b[3]))
                row = {"line": f"{orient} {ratio} ({direction})", "label": label, "ai_checks_per_sec": rate,
                       "entries": lc.entries, "exits": lc.exits}
                if expected_in is not None and expected_out is not None:
                    row["error"] = abs(lc.entries - expected_in) + abs(lc.exits - expected_out)
                table.append(row)
        preview = None
        if mid_frame is not None:
            img = mid_frame.copy()
            for _, orient, _, ratio in candidates:
                col = (0, 210, 255) if ratio == c.line_x_ratio and orient == c.line_orientation else (160, 160, 160)
                if orient == "vertical":
                    cv2.line(img, (int(ratio * w), 0), (int(ratio * w), h), col, 2)
                else:
                    cv2.line(img, (0, int(ratio * h)), (w, int(ratio * h)), col, 2)
            for tid, pts in trails.items():
                p = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [p], False, (60 + (tid * 53) % 195, 255 - (tid * 37) % 200, 80 + (tid * 91) % 175), 3)
            preview = _b64_jpeg(img, 560)
        persons_per_sample = [len(b) for _, b in samples]
        return {"camera": {"id": c.id, "name": c.name}, "imgsz": imgsz, "samples": len(samples), "sample_fps": round(eff_fps, 2),
                "clip_seconds": round(total / fps, 1), "samples_with_nobody": sum(1 for n in persons_per_sample if n == 0),
                "max_people_in_a_sample": max(persons_per_sample or [0]), "expected": {"in": expected_in, "out": expected_out},
                "table": table, "preview": preview,
                "note": "Each row runs the dashboard LineCounter with a different line position and AI-check rate. "
                        "Enter manually counted entries and exits to identify the closest match (error=0). The yellow line marks the current configuration; colored trails show tracked people."}
