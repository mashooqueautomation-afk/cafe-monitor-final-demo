"""Regression tests for the audit findings: counters/states must reflect what really happened."""
import time

import cv2
import numpy as np
import yaml
from fastapi.testclient import TestClient

import app.workers as workers
from app.config import CameraSpec, Settings, load_cameras
from app.main import create_app
from app.workers import CameraWorker

from .conftest import wait_for, write_video

PERSON = {"class_id": 0, "class_name": "person", "confidence": 0.9, "xyxy": [5, 5, 30, 45]}


def spec(path, role="none", **kw):
    return CameraSpec(id=3, key="c", name="C", subtitle="", purpose="", role=role, demo_file=path, demo_loop=True, **kw)


def run(worker, seconds=2.0):
    worker.start(); time.sleep(seconds); st = worker.status(); worker.request_stop(); worker.join(); return st


def test_frames_processed_counts_only_successful_inferences(make_settings, videos, monkeypatch):
    calls = []
    monkeypatch.setattr(workers, "yolo_detect", lambda *a, **k: (calls.append(k.get("classes")) or [PERSON], None))
    st = run(CameraWorker(spec(videos[0], "people_count"), "demo", make_settings(analytics_fps=2.0, demo_playback_fps=8.0)), 3.0)
    assert st["frames_processed"] == len(calls) >= 2
    assert st["frames_seen"] > st["frames_processed"] * 3          # most frames are NOT inferred
    assert st["processing_state"] == "running" and st["analytics"]["status"] == "ok"
    assert st["analytics"]["people_count"] == 1 and st["last_inference_ms"] is not None


def test_inference_failure_is_reported_not_hidden(make_settings, videos, monkeypatch):
    monkeypatch.setattr(workers, "yolo_detect", lambda *a, **k: (None, "weights missing"))
    st = run(CameraWorker(spec(videos[0], "people_count"), "demo", make_settings(analytics_fps=5.0)), 2.0)
    assert st["frames_processed"] == 0 and st["inference_errors"] > 0
    assert st["processing_state"] == "inference_error"
    assert st["analytics"]["available"] is False and st["analytics"]["status"] == "error"
    assert st["analytics"]["people_count"] is None                  # unavailable is None, never 0


def test_roles_without_analytics_do_not_run_the_model(make_settings, videos, monkeypatch):
    calls = []
    monkeypatch.setattr(workers, "yolo_detect", lambda *a, **k: (calls.append(1) or [], None))
    for role in ("none", "table_status"):                           # table_status has no regions here
        st = run(CameraWorker(spec(videos[0], role), "demo", make_settings(analytics_fps=5.0)), 1.5)
        assert st["processing_state"] == "not_configured" and st["analytics"]["status"] == "not_configured"
        assert st["frames_processed"] == 0
    assert calls == []


def test_inference_is_restricted_to_role_classes(make_settings, videos, monkeypatch):
    seen = {}
    def fake(frame, *a, **k):
        seen.setdefault("classes", []).append(k.get("classes")); return [], None
    monkeypatch.setattr(workers, "yolo_detect", fake)
    for role, expected in (("people_count", [0]), ("people_presence", [0]), ("entry_exit", [0]), ("cup_detection", [40, 41])):
        seen.clear()
        run(CameraWorker(spec(videos[0], role, line_x_ratio=0.5), "demo", make_settings(analytics_fps=5.0, entrance_fps=5.0)), 1.2)
        assert seen["classes"] and all(c == expected for c in seen["classes"]), role


def test_demo_loop_is_counted(make_settings, videos, monkeypatch):
    monkeypatch.setattr(workers, "yolo_detect", lambda *a, **k: ([PERSON], None))
    w = CameraWorker(spec(videos[0], "entry_exit", line_x_ratio=0.5), "demo", make_settings(entrance_fps=10.0))
    w.start()
    assert wait_for(lambda: w.status()["loops_completed"] >= 2, timeout=6)
    w.request_stop(); w.join()


def test_demo_loop_resets_cycle_counters_and_preserves_looping(make_settings, videos, monkeypatch):
    monkeypatch.setattr(workers, "yolo_detect", lambda *a, **k: ([PERSON], None))
    w = CameraWorker(spec(videos[0], "entry_exit", line_x_ratio=0.5), "demo", make_settings(entrance_fps=10.0))
    # Exercise the seam reset deterministically: replay cycles should start at zero.
    w._counter.entries = 4
    w._counter.exits = 2
    w._occupancy_offset = 3
    w._s["entries"], w._s["exits"] = 4, 2
    w._s["analytics"].update(entries=4, exits=2, people_inside=5)
    w._on_demo_loop()
    st = w.status()
    assert st["loops_completed"] == 1
    assert (st["analytics"]["entries"], st["analytics"]["exits"], st["analytics"]["people_inside"]) == (0, 0, 0)
    assert w._occupancy_offset == 0


def test_demo_path_is_matched_case_insensitively(tmp_path, make_settings):
    real = write_video(tmp_path / "Sample.mp4")
    cams = [{"id": i, "name": f"C{i}", "role": "none", "demo": {"file": str(tmp_path / "sample.mp4")}} for i in range(1, 9)]
    cfg = tmp_path / "c.yaml"; cfg.write_text(yaml.safe_dump({"cameras": cams}))
    s = Settings(_env_file=None, cameras_config=str(cfg))
    assert all(c.demo_file == real for c in load_cameras(s))        # Windows 'Sample.mp4' vs yaml 'sample.mp4'
    # ambiguous or genuinely missing files are NOT guessed: the normal 'not found' error stays
    (tmp_path / "sub").mkdir()
    assert workers.resolve_source(CameraSpec(id=1, key="k", name="n", subtitle="", purpose="", demo_file=tmp_path / "sub" / "x.mp4"),
                                  "demo", s).error.startswith("Video file not found")


def test_snapshot_endpoint(make_settings):
    with TestClient(create_app(make_settings())) as c:
        assert c.get("/api/cameras/99/snapshot").status_code == 404
        assert c.get("/api/cameras/1/snapshot").status_code == 409          # monitoring not started
        c.post("/api/monitoring/start")
        assert wait_for(lambda: c.get("/api/cameras/7/snapshot").status_code == 200)
        r = c.get("/api/cameras/7/snapshot")
        assert r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"
        c.post("/api/monitoring/stop")


def test_dashboard_jpeg_is_refreshed_at_display_fps_not_every_frame(make_settings, videos, monkeypatch):
    monkeypatch.setattr(workers, "yolo_detect", lambda *a, **k: ([], None))
    w = CameraWorker(spec(videos[0], "none"), "demo", make_settings(display_fps=2.0))
    w.start(); time.sleep(3.0); st = w.status(); shown = w.buffer.display_id; w.request_stop(); w.join()
    assert st["frames_played"] > 40                      # every decoded frame is still counted/processed
    assert 3 <= shown <= 10 and shown < st["frames_played"] / 4   # but only ~2 JPEGs/sec were encoded


def test_overview_entries_exits_cover_current_session_only(make_settings, monkeypatch):
    monkeypatch.setattr(workers, "yolo_detect", lambda *a, **k: ([], None))
    with TestClient(create_app(make_settings())) as c:
        mgr = c.app.state.manager
        mgr.persist_event(2, "entry", {})                # an event from an EARLIER session today
        time.sleep(0.2)
        c.post("/api/monitoring/start")
        assert wait_for(lambda: c.get("/api/analytics/summary").json()["detection"]["available"])
        assert c.get("/api/analytics/summary").json()["detection"]["entries"] == 0   # old session not mixed in
        mgr.persist_event(2, "entry", {})                # event during the current session
        assert c.get("/api/analytics/summary").json()["detection"]["entries"] == 1
        c.post("/api/monitoring/stop")


def test_start_only_selected_cameras(make_settings, monkeypatch):
    monkeypatch.setattr(workers, "yolo_detect", lambda *a, **k: ([PERSON], None))
    with TestClient(create_app(make_settings())) as c:
        r = c.post("/api/monitoring/start", json={"cameras": [7]})
        assert r.status_code == 200
        body = r.json()
        assert body["selected_camera_ids"] == [7] and body["focus_mode"] is True
        assert wait_for(lambda: next(x for x in c.get("/api/monitoring/status").json()["cameras"] if x["id"] == 7)["frames_played"] > 3)
        cams = {x["id"]: x for x in c.get("/api/monitoring/status").json()["cameras"]}
        assert cams[7]["selected"] and cams[7]["playback_state"] == "playing"
        for i in (1, 2, 3, 4, 5, 6, 8):
            assert cams[i]["selected"] is False and cams[i]["processing_state"] == "not_selected"
            assert cams[i]["playback_state"] == "playing" and cams[i]["frames_played"] > 0
            assert c.get(f"/api/cameras/{i}/snapshot").status_code == 200  # video stays visible; AI is paused
        assert c.get("/api/analytics/summary").json()["cameras_total"] == 8
        assert set(c.app.state.manager.workers) == set(range(1, 9))
        c.post("/api/monitoring/restart")                                          # restart keeps the selection
        assert c.get("/api/monitoring/status").json()["selected_camera_ids"] == [7]
        c.post("/api/monitoring/stop")
        # after stop every camera is idle again (selection preview is a dashboard concern)
        assert all(x["selected"] for x in c.get("/api/monitoring/status").json()["cameras"] if x["id"] == 7)


def test_invalid_camera_selection_is_rejected(make_settings):
    with TestClient(create_app(make_settings())) as c:
        assert c.post("/api/monitoring/start", json={"cameras": []}).status_code >= 400
        assert c.post("/api/monitoring/start", json={"cameras": [99]}).status_code >= 400
        assert c.get("/api/monitoring/status").json()["running"] is False


def test_focus_mode_raises_ai_rate_only_for_few_cameras(make_settings, videos):
    s = make_settings(entrance_fps=1.0, analytics_fps=0.5, focus_entrance_fps=6.0, focus_analytics_fps=2.0)
    entr = CameraWorker(spec(videos[0], "entry_exit", line_x_ratio=0.5), "demo", s, focus=True)
    assert entr.target_fps() == 6.0
    assert CameraWorker(spec(videos[0], "entry_exit", line_x_ratio=0.5), "demo", s, focus=False).target_fps() == 1.0
    assert CameraWorker(spec(videos[0], "people_count"), "demo", s, focus=True).target_fps() == 2.0
    assert CameraWorker(spec(videos[0], "people_count"), "demo", s, focus=False).target_fps() == 0.5


def test_people_presence_role_reports_current_frame_people_without_dwell(make_settings, videos, monkeypatch):
    monkeypatch.setattr(workers, "yolo_detect", lambda *a, **k: ([PERSON], None))
    st = run(CameraWorker(spec(videos[0], "people_presence"), "demo", make_settings(analytics_fps=5.0)), 1.5)
    assert st["processing_state"] == "running"
    assert st["analytics"]["people_count"] == 1
    assert st["analytics"]["tracked_people"] is None
    assert "cross-camera" in st["analytics"]["note"]


def test_cup_candidates_include_coco_cup_and_wine_glass_classes(make_settings, videos, monkeypatch):
    detections = [
        {"class_id": 41, "class_name": "cup", "confidence": 0.8, "xyxy": [2, 2, 20, 30]},
        {"class_id": 40, "class_name": "wine glass", "confidence": 0.7, "xyxy": [30, 2, 45, 30]},
    ]
    monkeypatch.setattr(workers, "yolo_detect", lambda *a, **k: (detections, None))
    st = run(CameraWorker(spec(videos[0], "cup_detection"), "demo", make_settings(analytics_fps=5.0)), 1.5)
    assert st["analytics"]["cup_count"] == 2
    assert "type/size/color" in st["analytics"]["note"]


def test_cafe_config_has_eight_channels_and_expected_user_media_paths():
    from app.config import ROOT, Settings, load_cameras
    from pathlib import Path
    settings = Settings(_env_file=None)
    cameras = load_cameras(settings)
    assert [c.id for c in cameras] == list(range(1, 9))
    by_id = {c.id: c for c in cameras}
    assert {i: by_id[i].role for i in (1, 2, 5)} == {1: "people_presence", 2: "people_presence", 5: "people_presence"}
    expected_folders = {
        1: "camera_01_seating", 2: "camera_02_kitchen", 3: "camera_03_main_hall",
        4: "camera_04_seating_wide", 5: "camera_05_service_counter",
        6: "camera_06_beverage_counter", 7: "camera_07_entrance",
        8: "camera_08_unmapped",
    }
    for camera_id, folder in expected_folders.items():
        assert by_id[camera_id].demo_file.relative_to(ROOT).as_posix() == f"data/client_footage/{folder}/sample.mp4"
