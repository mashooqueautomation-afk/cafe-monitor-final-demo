"""Per-camera workers (reader + processor threads) and the monitoring manager.

Reader thread  = playback: decodes frames (file or RTSP), publishes the latest
                 frame + JPEG into a FrameBuffer.
Processor thread = processing: consumes the latest frame and applies role-specific
                 YOLO inference when analytics is selected and the model is available.
Unselected feeds continue playback but skip inference to conserve CPU.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import cv2

from .config import CameraSpec, Mode, Settings
from .db import AnalyticsEvent, CameraSessionStat, MonitoringSession
from .tables import TableMonitor
from .tracking import LineCounter, PersonTracker
from .vision import detect as yolo_detect, draw_detections

log = logging.getLogger("cafe.workers")

# Keep OpenCV from creating large internal thread pools on the target two-core laptop.
try:
    cv2.setNumThreads(max(1, int(os.getenv("OPENCV_NUM_THREADS", "1"))))
except Exception:
    pass

# COCO class ids (yolov8n.pt). Restricting inference to the classes a role needs saves CPU and
# removes unrelated boxes (chairs, bottles, TVs...) that were being drawn on every camera.
COCO_PERSON = 0
COCO_CUP_LIKE = [40, 41]                      # 40 = wine glass, 41 = cup; both are generic drinkware candidates
COCO_TABLE_CLUTTER = [39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 53, 54, 55]  # bottle..bowl, apple, sandwich, orange, pizza, donut, cake
ROLE_CLASSES = {
    "people_count": [COCO_PERSON],
    "people_presence": [COCO_PERSON],
    "entry_exit": [COCO_PERSON],
    "cup_detection": COCO_CUP_LIKE,
    "table_status": [COCO_PERSON] + COCO_TABLE_CLUTTER,
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def mask_url(url: str) -> str:
    """rtsp://user:pw@host:554/path?x -> rtsp://***@host:554 (never log secrets)."""
    if url.startswith("webcam:"):
        return url  # "webcam:0" holds no secret
    try:
        p = urlsplit(url)
        host = p.hostname or "?"
        port = f":{p.port}" if p.port else ""
        auth = "***@" if (p.username or p.password) else ""
        return f"{p.scheme}://{auth}{host}{port}"
    except ValueError:
        return "rtsp://<unparseable>"


class FrameBuffer:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.frame = None
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.display_id = 0
        self.frame_position_sec: float | None = None
        self.frame_captured_monotonic = 0.0

    def put(self, frame, jpeg: bytes | None, position_sec: float | None = None) -> None:
        """Publish the latest frame only; an unprocessed older frame is intentionally replaced."""
        with self._cond:
            self.frame = frame
            self.frame_id += 1
            self.frame_position_sec = position_sec
            self.frame_captured_monotonic = time.monotonic()
            if jpeg is not None:
                self.jpeg = jpeg
                self.display_id += 1
            self._cond.notify_all()

    def latest(self) -> tuple[int, bytes | None]:
        with self._cond:
            return self.display_id, self.jpeg

    def set_display_jpeg(self, jpeg: bytes) -> None:
        """Replace the visible JPEG with annotated output without advancing frame_id."""
        with self._cond:
            self.jpeg = jpeg
            self.display_id += 1
            self._cond.notify_all()

    def wait_new(self, last_id: int, timeout: float):
        with self._cond:
            if self.frame_id == last_id:
                self._cond.wait(timeout)
            if self.frame_id == last_id:
                return None
            return self.frame_id, self.frame, self.frame_position_sec, self.frame_captured_monotonic

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()


class Source:
    def __init__(self, kind: str, target: str, error: str | None, configured: bool):
        self.kind, self.target, self.error, self.configured = kind, target, error, configured


def live_kind(url: str) -> str:
    """What kind of live source is this? rtsp://..., http(s)://... (phone MJPEG apps) or webcam:0 (laptop camera)."""
    low = url.lower()
    if low.startswith("webcam:"):
        return "webcam"
    if low.startswith(("http://", "https://")):
        return "http"
    return "rtsp"


def open_live_capture(target: str):
    if target.lower().startswith("webcam:"):
        idx = target.split(":", 1)[1].strip() or "0"
        cap = cv2.VideoCapture(int(idx) if idx.isdigit() else idx)
    else:
        cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best-effort: prefer fresh RTSP frames over a growing decode buffer
    except Exception:
        pass
    return cap


def resolve_source(spec: CameraSpec, mode: Mode, settings: Settings) -> Source:
    if mode == "demo":
        path = spec.demo_file
        if not str(path) or str(path) == ".":
            return Source("file", "", "No demo video file configured", False)
        if not path.exists():
            return Source("file", str(path), f"Video file not found: {path}", False)
        if not path.is_file():
            return Source("file", str(path), f"Video path is not a file: {path}", False)
        # Demo playback prefers a cached low-FPS proxy so seven 25-FPS source files are
        # not all decoded at full rate in parallel on a dual-core laptop. The original
        # recording remains untouched and is still used to generate the proxy.
        if settings.demo_use_proxies:
            proxy_dir = Path(settings.demo_proxy_dir).expanduser()
            if not proxy_dir.is_absolute():
                proxy_dir = Path(__file__).resolve().parents[2] / proxy_dir
            proxy = proxy_dir / f"camera_{spec.id:02d}.mp4"
            if proxy.is_file() and proxy.stat().st_size > 0:
                return Source("file", str(proxy), None, True)
        return Source("file", str(path), None, True)
    url = settings.rtsp_url(spec.id)
    if not url:
        return Source("rtsp", "", f"RTSP URL not configured (set CAMERA_{spec.id:02d}_RTSP)", False)
    return Source(live_kind(url), url, None, True)


class CameraWorker:
    def __init__(self, spec: CameraSpec, mode: Mode, settings: Settings, event_callback=None,
                 focus: bool = False, analytics_selected: bool = True,
                 opening_occupancy: int | None = None):
        self.spec, self.mode, self.settings = spec, mode, settings
        self.focus = focus
        self.analytics_selected = analytics_selected
        self.event_callback = event_callback
        self._last_inference_at = 0.0
        self._last_encode_at = 0.0
        self._counter: LineCounter | None = None
        if spec.role == "entry_exit":  # role is assigned in cameras.yaml
            self._counter = LineCounter(spec.line_orientation, spec.entry_direction, spec.line_x_ratio,
                                        band_ratio=settings.line_band_ratio, min_hits=settings.counter_min_hits,
                                        cooldown_seconds=settings.counter_cooldown_seconds)
        # A staff-verified baseline is installed before the worker starts, so early crossings
        # in the first frames are not missed while an operator races to enter the value.
        self._opening_occupancy: int | None = int(opening_occupancy) if opening_occupancy is not None and self._counter is not None else None
        self._occupancy_offset = self._opening_occupancy if self._opening_occupancy is not None else 0
        self._presence = PersonTracker(gate=1.8, max_age_seconds=settings.person_track_max_age)   # stabilize missed detections for a short interval
        self._tables: TableMonitor | None = None
        if spec.role == "table_status" and spec.table_regions:
            self._tables = TableMonitor(spec.table_regions, settings.table_occupy_seconds,
                                        settings.table_leave_seconds, settings.table_clutter_seconds)
        self._availability_event_sent = False
        # Live playback never reuses detector boxes from an older frame. The latest
        # annotated detector frame is retained separately as explicitly timestamped evidence.
        self._display_detections: list[dict] = []
        self._display_extras: list[dict] = self._line_extras()
        self._evidence_jpeg: bytes | None = None
        self.source = resolve_source(spec, mode, settings)
        self.buffer = FrameBuffer()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._s = dict(
            playback_state="starting", processing_state="starting", error=self.source.error,
            connected=False, position_sec=None, duration_sec=None, fps=None,
            frames_played=0, frames_processed=0, processed_frame_id=0,
            frames_seen=0, inference_errors=0, last_inference_ms=None, loops_completed=0,
            last_frame_at=None, reconnect_attempts=0,
            evidence_available=False, evidence_updated_at=None, evidence_position_sec=None,
            result_frame_age_ms=None,
            entries=0, exits=0,
            analytics={"available": False, "status": "waiting", "note": "Waiting for first inference", "people_count": None,
                       "entries": 0, "exits": 0, "people_inside": self._occupancy_offset if self._counter is not None else None, "messy_tables": None, "clean_tables": None,
                       "occupied_tables": None, "empty_tables": None, "cup_count": None, "detection_count": None, "zone_dwell_avg_seconds": None,
                       "zone_dwell_max_seconds": None, "tracked_people": None, "updated_at": None},
        )

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> bool:
        if self._started:
            return False  # a worker object is single-use: no duplicate threads
        self._started = True
        target = self._read_demo if self.mode == "demo" else self._read_live
        for name, fn in (("reader", self._guard(target)), ("processor", self._guard_processor)):
            t = threading.Thread(target=fn, name=f"cam{self.spec.id:02d}-{name}", daemon=True)
            self._threads.append(t)
            t.start()
        return True

    def request_stop(self) -> None:
        self._stop.set()
        self.buffer.wake()

    def join(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        for t in self._threads:
            t.join(max(0.0, deadline - time.monotonic()))
        alive = self.is_running()
        if alive:
            log.warning("camera %s threads did not stop within %.1fs", self.spec.id, timeout)
        return not alive

    def is_running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def _set(self, **kw) -> None:
        with self._lock:
            self._s.update(kw)

    def _line_extras(self) -> list[dict]:
        if self.spec.role != "entry_exit":
            return []
        kind = "line_x" if self.spec.line_orientation == "vertical" else "line_y"
        return [{"type": kind, "ratio": self.spec.line_x_ratio}]

    def _static_overlay_extras(self, width: int, height: int) -> list[dict]:
        """Return only static geometry for the live feed, never stale occupancy labels."""
        extras = self._line_extras()
        if self.spec.role == "table_status":
            for i, region in enumerate(self.spec.table_regions):
                x1, y1, x2, y2 = (int(float(region[k]) * dim) for k, dim in (
                    ("x1", width), ("y1", height), ("x2", width), ("y2", height)))
                extras.append({"type": "region", "xyxy": (x1, y1, x2, y2),
                               "label": region.get("name", f"TABLE {i + 1}")})
        return extras

    def evidence_snapshot(self) -> tuple[bytes | None, str | None, float | None]:
        """Return an annotated JPEG of the exact frame used for the last successful inference."""
        with self._lock:
            return (self._evidence_jpeg, self._s.get("evidence_updated_at"),
                    self._s.get("evidence_position_sec"))

    def _store_evidence(self, frame, detections: list[dict], extras: list[dict],
                        position_sec: float | None, frame_age_ms: float | None) -> None:
        """Keep detector marks paired with the exact source frame; never burn them into newer video."""
        out = draw_detections(frame, detections, extras)
        width = max(320, self.settings.max_stream_width)
        if out.shape[1] > width:
            out = cv2.resize(out, (width, int(out.shape[0] * width / out.shape[1])))
        ok, encoded = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality])
        if not ok:
            return
        updated = utcnow().isoformat()
        with self._lock:
            self._evidence_jpeg = encoded.tobytes()
            self._s["evidence_available"] = True
            self._s["evidence_updated_at"] = updated
            self._s["evidence_position_sec"] = position_sec
            self._s["result_frame_age_ms"] = round(frame_age_ms, 1) if frame_age_ms is not None else None

    def _assign_presence_track_ids(self, detections: list[dict], now: float | None = None) -> list:
        """Attach camera-local track IDs to person boxes for evidence snapshots."""
        people = self._person_boxes(detections)
        points = [((d["xyxy"][0] + d["xyxy"][2]) / 2.0, float(d["xyxy"][3]),
                   max(d["xyxy"][2] - d["xyxy"][0], d["xyxy"][3] - d["xyxy"][1])) for d in people]
        tracks = self._presence.update(points, time.monotonic() if now is None else now)
        for det, track in zip(people, tracks):
            det["track_id"] = track.id
        return tracks

    def people_inside(self) -> int | None:
        if self._counter is None:
            return None
        with self._lock:
            return max(0, self._counter.entries - self._counter.exits + self._occupancy_offset)

    def set_people_inside(self, n: int) -> int:
        """Staff correction: 'there are really N people inside right now'."""
        if self._counter is None:
            raise ValueError("Only the entrance camera tracks people inside")
        with self._lock:
            self._occupancy_offset = int(n) - (self._counter.entries - self._counter.exits)
            if self.mode == "demo":
                self._opening_occupancy = int(n)
            self._s["analytics"] = {**self._s["analytics"], "people_inside": int(n)}
        self._emit_event("occupancy_correction", {"people_inside": int(n)})
        return int(n)

    def confirm_table_cleaned(self, table_id: str) -> None:
        if self._tables is not None:
            self._tables.confirm_cleaned(table_id)

    def target_fps(self) -> float:
        """AI checks/second for this camera (higher when only a few cameras are selected: 'focus mode')."""
        s = self.settings
        if self.spec.role == "entry_exit":
            return max(s.entrance_fps, s.focus_entrance_fps) if self.focus else s.entrance_fps
        return max(s.analytics_fps, s.focus_analytics_fps) if self.focus else s.analytics_fps

    def inference_enabled(self) -> tuple[bool, str]:
        """Does this camera have an analytics role that actually needs the model? (saves CPU on the rest)"""
        role = self.spec.role
        if role not in ROLE_CLASSES:
            return False, "Analytics not configured for this camera (video monitoring only)"
        if role == "table_status" and not self.spec.table_regions:
            return False, "No table regions configured; calibrate normalized x1/y1/x2/y2 regions in config/cameras.yaml"
        return True, ""

    def _on_demo_loop(self) -> None:
        """Start a clean analytics cycle when recorded demo footage rewinds.

        Playback keeps looping, but counts are per playback cycle so the same
        prerecorded crossings do not accumulate as if they were new visitors.
        Historical AnalyticsEvent rows are retained and tagged with their cycle.
        Live RTSP workers never call this method.
        """
        if self._counter is not None:
            self._counter.tracker.reset()
            self._counter.entries = 0
            self._counter.exits = 0
        self._presence.reset()
        # Replays are independent cycles. Preserve the staff-verified opening occupancy
        # baseline but reset crossings so recorded entries are not accumulated repeatedly.
        self._occupancy_offset = self._opening_occupancy if self._opening_occupancy is not None else 0
        with self._lock:
            # Do not leave an evidence image from the previous loop presented as current.
            self._evidence_jpeg = None
            self._s["evidence_available"] = False
            self._s["evidence_updated_at"] = None
            self._s["evidence_position_sec"] = None
            self._s["result_frame_age_ms"] = None
            self._s["loops_completed"] += 1
            self._s["entries"] = 0
            self._s["exits"] = 0
            self._s["analytics"] = {
                **self._s["analytics"],
                "entries": 0,
                "exits": 0,
                "people_inside": self._occupancy_offset if self._counter is not None else None,
                "tracked_people": 0 if self.spec.role == "people_count" else self._s["analytics"].get("tracked_people"),
                "zone_dwell_avg_seconds": 0.0 if self.spec.role == "people_count" else self._s["analytics"].get("zone_dwell_avg_seconds"),
                "zone_dwell_max_seconds": 0.0 if self.spec.role == "people_count" else self._s["analytics"].get("zone_dwell_max_seconds"),
                "updated_at": utcnow().isoformat(),
                "note": "New demo playback cycle; counts reset to avoid replay totals accumulating",
            }

    def _guard(self, fn):
        def run():
            try:
                fn()
            except Exception as exc:  # one camera failing must never crash others
                log.exception("camera %s reader crashed", self.spec.id)
                self._set(playback_state="error", error=f"Worker error: {type(exc).__name__}", connected=False)
            finally:
                if self._stop.is_set():
                    st = self._s["playback_state"]
                    if st not in ("error", "ended", "offline"):
                        self._set(playback_state="stopped", connected=False)
        return run

    def _guard_processor(self) -> None:
        try:
            self._process_loop()
        except Exception as exc:
            log.exception("camera %s processor crashed", self.spec.id)
            with self._lock:
                self._s["processing_state"] = "error"
                self._s["analytics"] = {**self._s["analytics"], "available": False,
                    "note": f"Processing error: {type(exc).__name__}: {exc}"}
        finally:
            if self._stop.is_set():
                self._set(processing_state="stopped")

    # ---- publishing ------------------------------------------------------
    def _publish(self, frame, position: float | None, force_encode: bool = False) -> None:
        """Hand a decoded frame to the processor and (at most display_fps times/sec) refresh the dashboard JPEG.

        JPEG encoding is rate-limited to the dashboard refresh rate. Demo playback also samples frames
        with grab() between displayed frames; otherwise 8 x 25 FPS would trigger about 200 buffer updates/sec.
        frames_played still counts every decoded frame.
        """
        now = time.monotonic()
        due = force_encode or self.buffer.jpeg is None or (now - self._last_encode_at) >= 1.0 / max(1.0, self.settings.display_fps)
        jpeg = None
        if due:
            # The live video must stay synchronized to the current frame. Never draw
            # person boxes from a previous inference onto a newer frame: that creates
            # false boxes at the person's old location. Static door/table regions are OK.
            overlay_extras = self._static_overlay_extras(frame.shape[1], frame.shape[0])
            out = draw_detections(frame, [], overlay_extras) if overlay_extras else frame
            w = max(320, self.settings.max_stream_width)
            if out.shape[1] > w:
                out = cv2.resize(out, (w, int(out.shape[0] * w / out.shape[1])))
            ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality])
            if not ok:
                return
            jpeg = enc.tobytes()
            self._last_encode_at = now
        self.buffer.put(frame, jpeg, position)
        with self._lock:
            self._s["frames_played"] += 1
            self._s["position_sec"] = position
            self._s["last_frame_at"] = utcnow()

    # ---- Demo Mode reader --------------------------------------------------
    def _read_demo(self) -> None:
        src = self.source
        if src.error:
            self._set(playback_state="error", processing_state="idle")
            return
        cap = cv2.VideoCapture(src.target)
        try:
            if not cap.isOpened():
                return self._fail(f"Cannot open video (invalid or unsupported file): {Path(src.target).name}")
            ok, frame = cap.read()
            if not ok:
                return self._fail(f"Video has no readable frames: {Path(src.target).name}")
            source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            source_fps = min(max(source_fps, 1.0), 60.0)
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            target_fps = min(max(1.0, self.settings.demo_playback_fps), source_fps)
            frame_step = max(1, int(round(source_fps / target_fps)))
            # Keep clip time close to real time. Cached proxies are already encoded at only
            # four frames/second, so decoding skips very few frames instead of all 25 FPS.
            dt, idx, next_t = frame_step / source_fps, 0, time.monotonic()
            self._set(playback_state="playing", connected=True, fps=round(target_fps, 2),
                      duration_sec=round(total / source_fps, 2) if total > 0 else None, error=None)
            while not self._stop.is_set():
                self._publish(frame, idx / source_fps)
                next_t += dt
                wait = next_t - time.monotonic()
                if wait > 0:
                    if self._stop.wait(wait):
                        break
                elif wait < -1.0:
                    next_t = time.monotonic()  # fell behind: resync instead of racing

                # Skip source frames between dashboard/processor samples. Count skipped source frames
                # so frames_played remains a source-frame metric, not a sampled-frame metric.
                ok = True
                for _ in range(frame_step - 1):
                    ok = cap.grab()
                    if not ok:
                        break
                    with self._lock:
                        self._s["frames_played"] += 1
                if ok:
                    ok, frame = cap.read()
                if not ok:
                    if not self.spec.demo_loop:
                        self._set(playback_state="ended", connected=False)
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    idx = 0
                    self._on_demo_loop()
                    ok, frame = cap.read()
                    if not ok:
                        return self._fail("Video could not be looped (rewind failed)")
                else:
                    idx += frame_step
        finally:
            cap.release()

    def _fail(self, msg: str) -> None:
        log.warning("camera %s: %s", self.spec.id, msg)
        self._set(playback_state="error", error=msg, connected=False, processing_state="idle")

    # ---- Live Mode reader (NOT yet tested against the real café NVR) --------
    def _read_live(self) -> None:
        src = self.source
        if src.error:
            self._set(playback_state="offline", processing_state="idle", connected=False)
            return
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")
        masked = mask_url(src.target)
        lo, hi = self.settings.live_reconnect_min_seconds, self.settings.live_reconnect_max_seconds
        backoff, attempts = lo, 0
        # Show at most display_fps frames per second (saves CPU); the entrance camera needs enough frames for counting.
        pub_fps = max(self.settings.display_fps, self.target_fps() if self.spec.role == "entry_exit" else 0.0, 1.0)
        min_gap = 1.0 / pub_fps
        while not self._stop.is_set():
            self._set(playback_state="connecting" if attempts == 0 else "reconnecting")
            log.info("camera %s connecting to %s", self.spec.id, masked)
            cap = open_live_capture(src.target)
            try:
                if cap.isOpened():
                    self._set(playback_state="playing", connected=True, error=None)
                    backoff, bad, last_pub = lo, 0, 0.0
                    while not self._stop.is_set():
                        if not cap.grab():          # grab = cheap; only decode the frames we will actually use
                            bad += 1
                            if bad >= 30:
                                break
                            time.sleep(0.05)
                            continue
                        bad = 0
                        now = time.monotonic()
                        if now - last_pub < min_gap:
                            continue
                        ok, frame = cap.retrieve()
                        if not ok or frame is None:
                            continue
                        last_pub = now
                        self._publish(frame, None, force_encode=True)
            finally:
                cap.release()
            if self._stop.is_set():
                break
            attempts += 1
            self._set(playback_state="offline", connected=False, reconnect_attempts=attempts,
                      error=f"Camera unreachable ({masked}); retrying in {backoff:.0f}s")
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, hi)

    # ---- processing ------------------------------------------------------
    def _process_loop(self) -> None:
        last = 0
        self._set(processing_state="waiting")
        while not self._stop.is_set():
            got = self.buffer.wait_new(last, timeout=0.5)
            if got is None:
                st = self._s["playback_state"]
                if st in ("error", "offline") and self._s["processing_state"] != "idle":
                    self._set(processing_state="idle")
                elif st not in ("error", "offline") and self._s["processing_state"] == "running":
                    self._set(processing_state="waiting")
                continue
            last, frame, frame_position_sec, frame_captured_at = got
            outcome = self.process_frame(frame, frame_position_sec=frame_position_sec,
                                         frame_captured_at=frame_captured_at)
            with self._lock:
                self._s["frames_seen"] += 1          # frames handed to the processor (NOT inferences)
                self._s["processed_frame_id"] = last
                if outcome == "ok":
                    self._s["processing_state"] = "running"
                elif outcome == "error":
                    self._s["processing_state"] = "inference_error"
                elif outcome == "not_configured":
                    self._s["processing_state"] = "not_configured"
                elif outcome == "not_selected":
                    self._s["processing_state"] = "not_selected"
                elif outcome == "stale":
                    self._s["processing_state"] = "stale"
                # "skipped" (rate limit) keeps the previous state
        self._set(processing_state="stopped")

    def _emit_event(self, event_type: str, payload: dict) -> None:
        if not self.event_callback:
            return
        try:
            self.event_callback(self.spec.id, event_type, payload)
        except Exception:
            log.exception("failed to persist analytics event for camera %s", self.spec.id)

    def _person_boxes(self, detections: list[dict]) -> list[dict]:
        return [d for d in detections if d["class_id"] == 0 or d["class_name"].lower() == "person"]

    def _update_entrance(self, detections: list[dict], width: int, height: int) -> None:
        """Door-line counting with a real tracker (see tracking.py)."""
        assert self._counter is not None
        people = self._person_boxes(detections)
        events, track_ids = self._counter.update([p["xyxy"] for p in people], width, height, time.monotonic())
        for det, tid in zip(people, track_ids):
            det["track_id"] = tid
        with self._lock:
            self._s["entries"], self._s["exits"] = self._counter.entries, self._counter.exits
        with self._lock:
            demo_cycle = self._s["loops_completed"]
        for ev in events:
            payload = {"track_id": ev.track_id, "direction": ev.direction,
                       "line_ratio": self.spec.line_x_ratio, "orientation": self.spec.line_orientation}
            if self.mode == "demo":
                payload["demo_cycle"] = demo_cycle
            self._emit_event(ev.kind, payload)

    def _update_zone_presence(self, detections: list[dict]) -> dict:
        """Approximate dwell time inside the configured people-count camera; not total café visit time."""
        now = time.monotonic()
        people = self._person_boxes(detections)
        pts = [((d["xyxy"][0] + d["xyxy"][2]) / 2.0, float(d["xyxy"][3]),
                max(d["xyxy"][2] - d["xyxy"][0], d["xyxy"][3] - d["xyxy"][1])) for d in people]
        tracks = self._presence.update(pts, now)
        for det, tr in zip(people, tracks):
            det["track_id"] = tr.id
        durations = [max(0.0, now - tr.first_seen) for tr in tracks]
        # A short-lived tracker count smooths one-off missed detections. Tracks expire after
        # PERSON_TRACK_MAX_AGE seconds; this is an estimate, not a permanent customer identity.
        return {"tracked_people": len(self._presence.tracks),
                "detected_people_current_frame": len(people),
                "zone_dwell_avg_seconds": round(sum(durations) / len(durations), 1) if durations else 0.0,
                "zone_dwell_max_seconds": round(max(durations), 1) if durations else 0.0}

    def _update_tables(self, detections: list[dict], width: int, height: int) -> tuple[list[dict], dict]:
        """Occupied / empty / needs-check per configured table region."""
        assert self._tables is not None
        clutter_names = {"cup", "bowl", "bottle", "fork", "knife", "spoon", "pizza", "sandwich", "donut", "cake", "apple", "orange", "wine glass"}
        people = self._person_boxes(detections)
        now = time.monotonic()
        extras: list[dict] = []
        for i, region in enumerate(self.spec.table_regions):
            tid = str(region.get("id", region.get("name", f"table_{i + 1}")))
            x1, y1, x2, y2 = (int(float(region[k]) * dim) for k, dim in (("x1", width), ("y1", height), ("x2", width), ("y2", height)))

            def inside(d: dict) -> bool:
                cx, cy = (d["xyxy"][0] + d["xyxy"][2]) / 2, (d["xyxy"][1] + d["xyxy"][3]) / 2
                return x1 <= cx <= x2 and y1 <= cy <= y2

            person = any(inside(p) for p in people)
            clutter = any(d["class_name"].lower() in clutter_names and inside(d) for d in detections)
            state = self._tables.update(tid, person, clutter, now)
            extras.append({"type": "region", "xyxy": (x1, y1, x2, y2), "label": region.get("name", f"TABLE {i + 1}"), "state": state})
        return extras, self._tables.counts()

    def process_frame(self, frame, frame_position_sec: float | None = None,
                      frame_captured_at: float | None = None) -> str:
        """Rate-limited YOLO inference. Metrics stay unavailable when model inference fails.

        Returns "ok", "error", "skipped", "not_selected" (video only in focus mode), or
        "not_configured" (no analytics role / missing table regions). Only successful inferences
        increment frames_processed. All camera readers continue displaying video in focus mode.
        """
        if not self.analytics_selected:
            with self._lock:
                if self._s["analytics"].get("status") != "not_selected":
                    self._s["analytics"] = {**self._s["analytics"], "available": False, "status": "not_selected",
                                            "note": "Video is displayed; AI analytics are paused for this camera in focus mode",
                                            "updated_at": utcnow().isoformat()}
                self._display_detections = []
                self._display_extras = self._line_extras()
            return "not_selected"
        enabled, why = self.inference_enabled()
        if not enabled:
            with self._lock:
                if self._s["analytics"].get("status") != "not_configured":
                    self._s["analytics"] = {**self._s["analytics"], "available": False, "status": "not_configured",
                                            "note": why, "updated_at": utcnow().isoformat()}
            return "not_configured"
        now = time.monotonic()
        inference_period = 1.0 / max(0.1, self.target_fps())
        if now - self._last_inference_at < inference_period:
            return "skipped"
        # Cooldown is based on the previous inference completion, not its start.
        # Otherwise a 17-second inference on a 0.2-FPS profile would immediately
        # launch another one and permanently saturate the CPU.
        classes = ROLE_CLASSES[self.spec.role]
        t0 = time.monotonic()
        person_related = COCO_PERSON in classes
        confidence = min(self.settings.yolo_confidence, self.settings.person_detection_confidence) if person_related else self.settings.yolo_confidence
        imgsz = max(self.settings.yolo_imgsz, self.settings.person_detection_imgsz) if person_related else self.settings.yolo_imgsz
        tiled_people = (self.spec.role == "people_count" and self.settings.person_tiled_inference)
        detections, error = yolo_detect(frame, self.settings.yolo_model, confidence,
                                        self.settings.yolo_auto_download, device=self.settings.yolo_device,
                                        imgsz=imgsz, classes=classes, backend=self.settings.yolo_backend,
                                        tiled=tiled_people)
        inferred_at = time.monotonic()
        self._last_inference_at = inferred_at
        infer_ms = round((inferred_at - t0) * 1000.0, 1)
        frame_age_ms = ((inferred_at - frame_captured_at) * 1000.0) if frame_captured_at else None
        if detections is None:
            with self._lock:
                self._s["inference_errors"] += 1
                self._s["last_inference_ms"] = infer_ms
                self._s["analytics"] = {**self._s["analytics"], "available": False, "status": "error",
                    "note": error or "Detection model unavailable", "updated_at": utcnow().isoformat()}
                self._display_detections = []
                self._display_extras = self._line_extras()
            return "error"

        # Do not update the visible count or crossing counters from an inference whose source
        # frame is already too old. Keep the exact frame as evidence but mark the analytics stale.
        # Latest-frame buffering drops queued frames; this final guard prevents a slow result
        # from being mistaken for a measurement of the current scene.
        if frame_age_ms is not None and frame_age_ms > self.settings.max_result_frame_age_ms:
            self._store_evidence(frame, detections, [], frame_position_sec, frame_age_ms)
            # Keep the count that belongs to this exact old frame as evidence, but do not
            # update tracks, table state or entrance counters from delayed inference. The UI
            # labels this value HISTORICAL / STALE rather than presenting it as current.
            evidence_people = len(self._person_boxes(detections)) if self.spec.role == "people_count" else None
            note = (f"HISTORICAL EVIDENCE ONLY: detector found {evidence_people} people in a source frame "
                    f"that was {int(frame_age_ms)} ms old at completion. Not a current live count; "
                    "tracking and counters were not updated.")
            with self._lock:
                old = self._s["analytics"]
                self._s["analytics"] = {**old, "available": False, "status": "stale",
                    "people_count": evidence_people, "tracked_people": None,
                    "detection_count": len(detections), "note": note,
                    "updated_at": utcnow().isoformat()}
                self._s["last_inference_ms"] = infer_ms
                self._s["frames_processed"] += 1
                self._s["processing_state"] = "stale"
                self._display_detections = []
                self._display_extras = self._line_extras()
            return "stale"

        if not self._availability_event_sent:
            self._availability_event_sent = True
            self._emit_event("inference_available", {"model": self.settings.yolo_model, "mode": self.mode})
        height, width = frame.shape[:2]
        extras: list[dict] = []
        analytics = {
            "available": True, "status": "ok", "note": "COCO pretrained model; validate on actual café footage",
            "people_count": None, "entries": self._s["analytics"].get("entries", 0),
            "exits": self._s["analytics"].get("exits", 0), "people_inside": None,
            "messy_tables": None, "clean_tables": None, "occupied_tables": None, "empty_tables": None,
            "cup_count": None, "detection_count": len(detections),
            "zone_dwell_avg_seconds": None, "zone_dwell_max_seconds": None, "tracked_people": None,
            "updated_at": utcnow().isoformat(),
        }
        if self.spec.role == "people_presence":
            self._assign_presence_track_ids(detections)
            analytics["people_count"] = len(self._person_boxes(detections))
            analytics["note"] = "Current-frame people visible in this camera only; not staff identity or cross-camera customer tracking"
        elif self.spec.role == "people_count":
            zone = self._update_zone_presence(detections)
            # The visible count comes from this exact processed frame. Recent tracks are a
            # separate smoothing aid and must not be mislabeled as a current-frame count.
            analytics["people_count"] = zone["detected_people_current_frame"]
            analytics.update(zone)
            if analytics["people_count"] == 0:
                analytics["note"] = "No people detected in this processed frame. Main Hall uses overlapping high-recall tiles; inspect LAST AI EVIDENCE and compare with a manual frame count."
            else:
                analytics["note"] = "Current-frame people count from overlapping Main Hall tiles; active tracks are a separate short-term estimate. Check the evidence frame against a manual count."
        elif self.spec.role == "entry_exit":
            extras.extend(self._line_extras())
            self._update_entrance(detections, width, height)
            analytics["note"] = "Door-line counter with tracking; calibrate the line on real footage and compare with a manual count"
        elif self.spec.role == "table_status":
            regions = self.spec.table_regions
            self._assign_presence_track_ids(detections)
            if self._tables is not None:
                table_extras, counts = self._update_tables(detections, width, height)
                extras.extend(table_extras)
                analytics["occupied_tables"] = counts["occupied"]
                analytics["empty_tables"] = counts["empty"]
                # needs_check = cups/plates left on a table nobody is sitting at. A REVIEW PROMPT, not proof of dirt.
                analytics["messy_tables"] = counts["needs_check"]
                analytics["clean_tables"] = None   # cleanliness cannot be proven by this model
                analytics["note"] = "Occupied = person seated. Needs-check = items left with nobody there (staff must confirm). Clean/dirty is NOT classified."
            elif regions:
                analytics["note"] = "Table regions are configured but the table monitor did not start"
            else:
                analytics["note"] = "No table regions configured; define normalized x1/y1/x2/y2 regions in config/cameras.yaml"
        elif self.spec.role == "cup_detection":
            # The supplied reference set includes clear glass cups as well as ceramic/takeaway cups.
            # COCO's generic "wine glass" class is therefore counted as a drinkware candidate too;
            # this is deliberately NOT claimed to be a verified cup-type classifier.
            analytics["cup_count"] = sum(1 for d in detections if d["class_id"] in COCO_CUP_LIKE or d["class_name"].lower() in {"cup", "wine glass"})
            analytics["note"] = "Generic COCO cup/drinkware candidates (cup + wine glass); type/size/color and serve-count need a café-trained model and validation"

        with self._lock:
            # Counters are maintained by the entrance tracker; keep them in the published analytics.
            analytics["entries"] = self._s["entries"]
            analytics["exits"] = self._s["exits"]
            if self._counter is not None:
                analytics["people_inside"] = max(0, self._counter.entries - self._counter.exits + self._occupancy_offset)
            self._s["analytics"] = analytics
            self._s["frames_processed"] += 1          # successful inference only
            self._s["last_inference_ms"] = infer_ms
            # Retained for diagnostics/tests only; _publish never paints these on newer frames.
            self._display_detections = detections
            self._display_extras = extras
        self._store_evidence(frame, detections, extras, frame_position_sec, frame_age_ms)
        return "ok"

    # ---- status ----------------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            s = dict(self._s)
        s["sync_lag_frames"] = max(0, self.buffer.frame_id - s["processed_frame_id"])
        return _decorate(self.spec, self.mode, self.source, s)


def _decorate(spec: CameraSpec, mode: Mode, source: Source, s: dict) -> dict:
    live = mode == "live"
    if live:
        label = "LIVE" if (s["connected"] and s["playback_state"] == "playing") else "OFFLINE"
    else:
        label = "DEMO"
    last = s.get("last_frame_at")
    return {
        "id": spec.id, "name": spec.name, "mode": mode, "label": label, "recorded": not live,
        "source_kind": source.kind, "source_configured": source.configured,
        "source_path": source.target if (not live and source.target) else None,  # never RTSP
        **{k: s[k] for k in ("playback_state", "processing_state", "error", "connected", "position_sec",
                              "duration_sec", "fps", "frames_played", "frames_processed",
                              "reconnect_attempts", "sync_lag_frames")},
        "frames_seen": s.get("frames_seen", 0), "inference_errors": s.get("inference_errors", 0),
        "last_inference_ms": s.get("last_inference_ms"), "loops_completed": s.get("loops_completed", 0),
        "evidence_available": s.get("evidence_available", False),
        "evidence_updated_at": s.get("evidence_updated_at"),
        "evidence_position_sec": s.get("evidence_position_sec"),
        "result_frame_age_ms": s.get("result_frame_age_ms"),
        "last_frame_at": last.isoformat() if last else None,
        "analytics": s.get("analytics", {}),
    }


def idle_status(spec: CameraSpec, mode: Mode, settings: Settings) -> dict:
    src = resolve_source(spec, mode, settings)
    s = dict(playback_state="idle", processing_state="idle", error=src.error, connected=False,
             position_sec=None, duration_sec=None, fps=None, frames_played=0, frames_processed=0,
             reconnect_attempts=0, sync_lag_frames=0, last_frame_at=None,
             analytics={"available": False, "status": "idle", "note": "Not running", "people_count": None, "entries": None,
                        "exits": None, "people_inside": None, "occupied_tables": None, "empty_tables": None, "messy_tables": None, "clean_tables": None, "cup_count": None,
                        "detection_count": None, "zone_dwell_avg_seconds": None,
                        "zone_dwell_max_seconds": None, "tracked_people": None, "updated_at": None})
    return _decorate(spec, mode, src, s)


class ModeChangeError(Exception):
    pass


class MonitoringManager:
    def __init__(self, settings: Settings, cameras: list[CameraSpec], session_factory):
        self.settings, self.cameras, self._sf = settings, cameras, session_factory
        self._lock = threading.RLock()
        self.mode: Mode = settings.default_mode
        self.running = False
        self.session_id: int | None = None
        self.started_at: datetime | None = None
        self.workers: dict[int, CameraWorker] = {}
        self.selected: list[int] = [c.id for c in cameras]   # cameras chosen for the current/last run
        self.focus_mode = False
        self.opening_occupancy: int | None = None

    def persist_event(self, camera_id: int, event_type: str, payload: dict) -> None:
        with self._sf() as db:
            saved_payload = {"mode": self.mode, **payload}
            db.add(AnalyticsEvent(camera_id=camera_id, event_type=event_type, created_at=utcnow(), payload=saved_payload))
            db.commit()

    def start(self, mode: Mode | None = None, camera_ids: list[int] | None = None,
              opening_occupancy: int | None = None) -> bool:
        """Returns False (no-op) if already running -> no duplicate workers.

        camera_ids=None -> AI analytics on all cameras; otherwise only those cameras run AI.
        Video still plays on every configured camera so focus mode does not blank the other tiles.
        """
        with self._lock:
            if self.running:
                if mode and mode != self.mode:
                    raise ModeChangeError("Stop monitoring before switching mode")
                return False
            if mode:
                self.mode = mode
            if opening_occupancy is not None:
                if not isinstance(opening_occupancy, int) or opening_occupancy < 0 or opening_occupancy > 1000:
                    raise ModeChangeError("Opening occupancy must be a whole number from 0 to 1000")
                self.opening_occupancy = opening_occupancy
            if camera_ids is not None:
                known = {c.id for c in self.cameras}
                chosen = sorted(set(camera_ids))
                if not chosen or any(i not in known for i in chosen):
                    raise ModeChangeError(f"Choose at least one valid camera id from {sorted(known)}")
                self.selected = chosen
            with self._sf() as db:
                row = MonitoringSession(mode=self.mode, started_at=utcnow())
                db.add(row)
                db.commit()
                self.session_id = row.id
            self.started_at = utcnow()
            self.focus_mode = len(self.selected) <= self.settings.focus_max_cameras < len(self.cameras)
            self.workers = {
                c.id: CameraWorker(c, self.mode, self.settings, event_callback=self.persist_event,
                                   focus=self.focus_mode and c.id in self.selected,
                                   analytics_selected=c.id in self.selected,
                                   opening_occupancy=self.opening_occupancy if c.role == "entry_exit" else None)
                for c in self.cameras
            }
            for w in self.workers.values():
                w.start()
            self.running = True
            log.info("monitoring started: session=%s mode=%s", self.session_id, self.mode)
            return True

    def stop(self) -> bool:
        with self._lock:
            if not self.running:
                return False
            for w in self.workers.values():
                w.request_stop()
            for w in self.workers.values():
                w.join(timeout=5.0)
            with self._sf() as db:
                for cid, w in self.workers.items():
                    st = w.status()
                    db.add(CameraSessionStat(
                        session_id=self.session_id, camera_id=cid, frames_played=st["frames_played"],
                        frames_processed=st["frames_processed"], final_playback_state=st["playback_state"],
                        last_error=(st["error"] or None) and st["error"][:500]))
                row = db.get(MonitoringSession, self.session_id)
                if row:
                    row.stopped_at = utcnow()
                db.commit()
            self.workers, self.running = {}, False
            log.info("monitoring stopped: session=%s", self.session_id)
            return True

    def restart(self) -> None:
        with self._lock:
            self.stop()
            self.start()

    def set_mode(self, mode: Mode) -> None:
        with self._lock:
            if self.running and mode != self.mode:
                raise ModeChangeError("Stop monitoring before switching mode")
            self.mode = mode

    def camera_status(self, cam_id: int) -> dict | None:
        spec = next((c for c in self.cameras if c.id == cam_id), None)
        if spec is None:
            return None
        w = self.workers.get(cam_id)
        if w:
            return {**w.status(), "selected": cam_id in self.selected}
        st = idle_status(spec, self.mode, self.settings)
        picked = cam_id in self.selected
        if self.running and not picked:
            st["processing_state"] = "not_selected"
            st["error"] = None
            st["analytics"] = {**st["analytics"], "status": "not_selected", "note": "Camera not selected for this run"}
        return {**st, "selected": picked or not self.running}

    def snapshot(self) -> dict:
        with self._lock:
            cams = [self.camera_status(c.id) for c in self.cameras]
            return {"mode": self.mode, "running": self.running, "session_id": self.session_id if self.running else None,
                    "started_at": self.started_at.isoformat() if (self.running and self.started_at) else None,
                    "cameras": cams, "selected_camera_ids": list(self.selected), "focus_mode": self.running and self.focus_mode}

    def summary(self) -> dict:
        from sqlalchemy import func, select
        snap = self.snapshot()
        cams = snap["cameras"]
        event_totals = {"entry": 0, "exit": 0}
        inference_sessions_today = 0
        with self._sf() as db:
            sessions_total = db.scalar(select(func.count(MonitoringSession.id))) or 0
            lifetime = db.scalar(select(func.coalesce(func.sum(CameraSessionStat.frames_processed), 0))) or 0
            try:
                today = utcnow().date()
                today_start = datetime.combine(today, datetime.min.time())
                if snap["running"] and self.started_at:
                    # While monitoring runs, show this session only (tile counters are per session);
                    # otherwise earlier sessions of the day made the Overview disagree with the tile.
                    today_start = max(today_start, self.started_at.replace(tzinfo=None))
                rows = db.execute(select(AnalyticsEvent.camera_id, AnalyticsEvent.event_type, AnalyticsEvent.payload).where(
                    AnalyticsEvent.event_type.in_(["entry", "exit", "inference_available"]),
                    AnalyticsEvent.created_at >= today_start,
                )).all()
                current_demo_cycles = {c["id"]: c.get("loops_completed", 0) for c in cams}
                for camera_id, event_type, payload in rows:
                    # Never mix counts from prerecorded Demo Mode with actual Live Mode events.
                    event_mode = (payload or {}).get("mode", snap["mode"])
                    if event_mode != snap["mode"]:
                        continue
                    if event_type in event_totals:
                        # During demo playback, overview totals match the currently visible
                        # cycle. Prior-cycle rows remain in the database for audit/history.
                        if snap["running"] and snap["mode"] == "demo":
                            if (payload or {}).get("demo_cycle", 0) != current_demo_cycles.get(camera_id, 0):
                                continue
                        event_totals[event_type] += 1
                    elif event_type == "inference_available":
                        inference_sessions_today += 1
            except Exception:
                log.exception("Unable to query analytics events")
        # Analytics are keyed by semantic role, not physical NVR channel number.
        analytics_by_role: dict[str, dict] = {}
        for spec in self.cameras:
            if spec.role not in analytics_by_role:
                row = next((c for c in cams if c["id"] == spec.id), {})
                analytics_by_role[spec.role] = row.get("analytics", {})
        d_people = analytics_by_role.get("people_count", {})
        d_entry = analytics_by_role.get("entry_exit", {})
        d_tables = analytics_by_role.get("table_status", {})
        d_cups = analytics_by_role.get("cup_detection", {})
        relevant = (d_people, d_entry, d_tables, d_cups)
        detection_available = any(x.get("available") for x in relevant) or inference_sessions_today > 0
        notes = [x.get("note") for x in relevant if x.get("note")]
        return {
            "mode": snap["mode"], "monitoring_running": snap["running"], "session_id": snap["session_id"],
            "cameras_total": len(cams),
            "cameras_playing": sum(c["playback_state"] == "playing" for c in cams),
            "cameras_with_problems": sum(c["playback_state"] in ("error", "offline") for c in cams),
            "frames_played_session": sum(c["frames_played"] for c in cams),
            "frames_processed_session": sum(c["frames_processed"] for c in cams),
            "sessions_total": sessions_total, "frames_processed_lifetime": int(lifetime) + sum(c["frames_processed"] for c in cams),
            "detection": {"available": detection_available,
                          "note": "; ".join(dict.fromkeys(notes)) if notes else "Waiting for model inference",
                          "people_count": d_people.get("people_count"), "people_inside": d_entry.get("people_inside"),
                          "occupied_tables": d_tables.get("occupied_tables"), "empty_tables": d_tables.get("empty_tables"),
                          "entries": event_totals["entry"] if detection_available else None,
                          "exits": event_totals["exit"] if detection_available else None,
                          "messy_tables": d_tables.get("messy_tables"), "clean_tables": d_tables.get("clean_tables"),
                          "cup_count": d_cups.get("cup_count"), "zone_dwell_avg_seconds": d_people.get("zone_dwell_avg_seconds"),
                          "zone_dwell_max_seconds": d_people.get("zone_dwell_max_seconds"), "tracked_people": d_people.get("tracked_people")},
            "integrations": {"cashier_pos": {"available": False,
                "note": "Not integrated yet: requires confirmation of a dedicated cashier camera and/or authorized POS/API data."}},
        }

    def set_people_inside(self, n: int) -> int:
        """Staff correction of the 'people inside now' number (needs monitoring to be running)."""
        with self._lock:
            entry_id = next((c.id for c in self.cameras if c.role == "entry_exit"), None)
            w = self.workers.get(entry_id) if entry_id is not None else None
            if not self.running or w is None:
                raise ModeChangeError("Start monitoring first, then set the number of people inside")
            value = w.set_people_inside(n)
            self.opening_occupancy = value
            return value

    def confirm_table_cleaned(self, table_id: str) -> None:
        # Resolve the camera by semantic role, but forward the actual table ID unchanged.
        camera_id = next((c.id for c in self.cameras if c.role == "table_status"), None)
        w = self.workers.get(camera_id) if camera_id is not None else None
        if w is not None:
            w.confirm_table_cleaned(table_id)

    def shutdown(self) -> None:
        self.stop()
