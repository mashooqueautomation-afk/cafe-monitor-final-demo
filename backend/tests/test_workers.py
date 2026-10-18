import time

import cv2

import app.workers as workers
from app.config import CameraSpec
from app.workers import CameraWorker, MonitoringManager, mask_url

from .conftest import cam_threads, wait_for


def spec_for(path, loop):
    return CameraSpec(id=1, key="c", name="C", subtitle="", purpose="", demo_file=path, demo_loop=loop)


def test_mask_url_hides_credentials():
    m = mask_url("rtsp://admin:pw123@192.168.1.5:554/Streaming/Channels/101?token=abc")
    assert m == "rtsp://***@192.168.1.5:554" and "pw123" not in m and "token" not in m


def test_opening_occupancy_is_installed_before_first_frame(make_settings, videos):
    spec = CameraSpec(id=7, key="entrance", name="Entrance", subtitle="", purpose="",
                      role="entry_exit", demo_file=videos[0], line_orientation="vertical",
                      line_x_ratio=0.72, entry_direction="right_to_left")
    w = CameraWorker(spec, "demo", make_settings(), opening_occupancy=16)
    assert w.people_inside() == 16
    assert w.status()["analytics"]["people_inside"] == 16
    assert w._opening_occupancy == 16


def test_no_loop_plays_once_then_ends(make_settings, videos):
    w = CameraWorker(spec_for(videos[0], False), "demo", make_settings())
    w.start()
    assert wait_for(lambda: w.status()["playback_state"] == "ended", timeout=6)
    assert w.status()["frames_played"] == 20
    w.request_stop(); w.join()


def test_loop_keeps_playing_past_video_length(make_settings, videos):
    w = CameraWorker(spec_for(videos[0], True), "demo", make_settings())
    w.start()
    assert wait_for(lambda: w.status()["frames_played"] > 45, timeout=8)  # > 2 full loops of 20 frames
    assert w.status()["playback_state"] == "playing"
    w.request_stop(); assert w.join()


def test_processing_stays_in_sync_with_playback(make_settings, videos):
    # sync is measured on frames_seen (frames the processor consumed); frames_processed = real inferences
    w = CameraWorker(spec_for(videos[0], True), "demo", make_settings())
    w.start()
    assert wait_for(lambda: w.status()["frames_played"] > 30)
    st = w.status()
    assert st["frames_seen"] > 0 and st["sync_lag_frames"] <= 3
    w.request_stop(); w.join()


def test_start_twice_does_not_spawn_extra_threads(make_settings, videos):
    w = CameraWorker(spec_for(videos[0], True), "demo", make_settings())
    assert w.start() is True and w.start() is False
    assert len(cam_threads()) == 2
    w.request_stop(); w.join()
    assert cam_threads() == []


def test_capture_is_released_on_stop(make_settings, videos, monkeypatch):
    opened, released = [], []
    Real = cv2.VideoCapture

    class Proxy:
        def __init__(self, *a):
            self._c = Real(*a); opened.append(1)
        def release(self):
            released.append(1); self._c.release()
        def __getattr__(self, n):
            return getattr(self._c, n)

    monkeypatch.setattr(workers.cv2, "VideoCapture", Proxy)
    w = CameraWorker(spec_for(videos[0], True), "demo", make_settings())
    w.start()
    assert wait_for(lambda: w.status()["frames_played"] > 2)
    w.request_stop(); assert w.join()
    assert len(opened) == len(released) == 1


def test_manager_forwards_table_id_to_table_status_camera():
    from unittest.mock import Mock

    manager = MonitoringManager.__new__(MonitoringManager)
    manager.cameras = [CameraSpec(id=4, key="tables", name="Tables", subtitle="", purpose="", role="table_status")]
    worker = Mock()
    manager.workers = {4: worker}

    manager.confirm_table_cleaned("table_3")

    worker.confirm_table_cleaned.assert_called_once_with("table_3")


def test_live_video_never_draws_previous_person_detections(make_settings, videos, monkeypatch):
    """A person's previous bounding box must not be painted on a newer video frame."""
    import numpy as np
    calls = []
    monkeypatch.setattr(workers, "draw_detections", lambda frame, detections, extras=None: calls.append(list(detections)) or frame)
    spec = CameraSpec(id=1, key="c", name="C", subtitle="", purpose="t", role="people_presence", demo_file=videos[0])
    w = CameraWorker(spec, "demo", make_settings())
    w._display_detections = [{"class_id": 0, "class_name": "person", "confidence": 0.9, "xyxy": [5, 5, 30, 40]}]
    w._publish(np.zeros((48, 64, 3), dtype=np.uint8), 1.0, force_encode=True)
    assert calls == []


def test_successful_inference_creates_separate_exact_frame_evidence(make_settings, videos, monkeypatch):
    import numpy as np
    detections = [{"class_id": 0, "class_name": "person", "confidence": 0.91, "xyxy": [5, 5, 30, 40]}]
    seen = []
    monkeypatch.setattr(workers, "yolo_detect", lambda *args, **kwargs: (detections, None))
    monkeypatch.setattr(workers, "draw_detections", lambda frame, dets, extras=None: seen.append(list(dets)) or frame.copy())
    spec = CameraSpec(id=1, key="c", name="C", subtitle="", purpose="t", role="people_presence", demo_file=videos[0])
    w = CameraWorker(spec, "demo", make_settings())
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    assert w.process_frame(frame, frame_position_sec=2.5) == "ok"
    jpeg, updated_at, position = w.evidence_snapshot()
    assert jpeg and updated_at and position == 2.5
    assert seen == [detections]
    assert w.status()["evidence_available"] is True


def test_main_hall_tiled_mode_counts_only_people_in_current_frame(make_settings, videos, monkeypatch):
    import numpy as np

    responses = [
        ([{"class_id": 0, "class_name": "person", "confidence": 0.8, "xyxy": [5, 5, 30, 40]}], None),
        ([], None),
    ]
    calls = []
    def fake_detect(*args, **kwargs):
        calls.append(kwargs)
        return responses.pop(0)
    monkeypatch.setattr(workers, "yolo_detect", fake_detect)
    spec = CameraSpec(id=3, key="hall", name="Main Hall", subtitle="", purpose="t",
                      role="people_count", demo_file=videos[2])
    w = CameraWorker(spec, "demo", make_settings(analytics_fps=10, person_tiled_inference=True))
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    assert w.process_frame(frame) == "ok"
    assert w.status()["analytics"]["people_count"] == 1
    assert calls[-1]["tiled"] is True
    # Even if the tracker holds a recent ID, a frame with no visible boxes reports 0 current people.
    time.sleep(0.11)
    w._last_inference_at = 0
    assert w.process_frame(frame) == "ok"
    a = w.status()["analytics"]
    assert a["people_count"] == 0
    assert a["tracked_people"] == 1


def test_stale_inference_is_not_published_as_current_count(make_settings, videos, monkeypatch):
    import numpy as np
    person = {"class_id": 0, "class_name": "person", "confidence": 0.9, "xyxy": [5, 5, 30, 40]}
    monkeypatch.setattr(workers, "yolo_detect", lambda *a, **k: ([person], None))
    spec = CameraSpec(id=3, key="hall", name="Main Hall", subtitle="", purpose="t",
                      role="people_count", demo_file=videos[2])
    w = CameraWorker(spec, "demo", make_settings(analytics_fps=10, max_result_frame_age_ms=1000))
    result = w.process_frame(np.zeros((48, 64, 3), dtype=np.uint8), frame_captured_at=time.monotonic() - 2.0)
    assert result == "stale"
    a = w.status()["analytics"]
    assert a["status"] == "stale" and a["available"] is False
    assert a["people_count"] == 1 and a["detection_count"] == 1
    assert "HISTORICAL EVIDENCE ONLY" in a["note"]
    assert w.status()["evidence_available"] is True
    assert w._counter is None  # stale observations do not update counters or tracker state


def test_manual_opening_occupancy_survives_demo_loop(make_settings, videos):
    spec = CameraSpec(id=7, key="entrance", name="Entrance", subtitle="", purpose="t",
                      role="entry_exit", demo_file=videos[6], line_orientation="vertical")
    w = CameraWorker(spec, "demo", make_settings())
    assert w.set_people_inside(16) == 16
    w._counter.entries = 3
    w._s["entries"] = 3
    assert w.people_inside() == 19
    w._on_demo_loop()
    assert w.people_inside() == 16


def test_inference_cooldown_starts_after_completion(make_settings, videos, monkeypatch):
    import numpy as np
    calls = []
    person = {"class_id": 0, "class_name": "person", "confidence": 0.9, "xyxy": [5, 5, 30, 40]}
    def slow_detect(*args, **kwargs):
        calls.append(time.monotonic())
        time.sleep(0.06)
        return [person], None
    monkeypatch.setattr(workers, "yolo_detect", slow_detect)
    spec = CameraSpec(id=1, key="presence", name="Presence", subtitle="", purpose="t",
                      role="people_presence", demo_file=videos[0])
    w = CameraWorker(spec, "demo", make_settings(analytics_fps=10.0))
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    assert w.process_frame(frame) == "ok"
    # Even though the fake inference took longer than the configured 0.1 s interval,
    # no next inference is due immediately on the very next fresh frame.
    assert w.process_frame(frame) == "skipped"
    assert len(calls) == 1
