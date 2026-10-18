"""End-to-end through CameraWorker.process_frame with a scripted fake detector (no YOLO needed)."""
import numpy as np
import pytest

import app.workers as workers
from app.config import CameraSpec
from app.workers import CameraWorker, live_kind, mask_url, resolve_source


def person(cx, y2=400):
    return {"class_id": 0, "class_name": "person", "confidence": 0.9, "xyxy": [cx - 40, y2 - 300, cx + 40, y2]}


@pytest.fixture()
def clock(monkeypatch):
    state = {"t": 1000.0}
    monkeypatch.setattr(workers.time, "monotonic", lambda: state["t"])
    return state


def entrance_worker(make_settings, events):
    spec = CameraSpec(id=2, key="c2", name="Entrance", subtitle="", purpose="", role="entry_exit", line_x_ratio=0.5, entry_direction="left_to_right")
    return CameraWorker(spec, "demo", make_settings(entrance_fps=8.0), event_callback=lambda cam, kind, payload: events.append((kind, payload)))


def drive(worker, monkeypatch, clock, xs, dt=0.2):
    frame = np.zeros((720, 1280, 3), np.uint8)
    seq = iter(xs)
    monkeypatch.setattr(workers, "yolo_detect", lambda *a, **k: ([person(next(seq))], None))
    for _ in xs:
        worker.process_frame(frame)
        clock["t"] += dt


def test_entrance_counts_people_inside_and_events(make_settings, monkeypatch, clock):
    events = []
    w = entrance_worker(make_settings, events)
    drive(w, monkeypatch, clock, [200 + 30 * i for i in range(30)])          # one person walks in
    a = w.status()["analytics"]
    assert (a["entries"], a["exits"], a["people_inside"]) == (1, 0, 1)
    assert [k for k, _ in events if k in ("entry", "exit")] == ["entry"]
    drive(w, monkeypatch, clock, [200 + 30 * i for i in range(30)])          # a second person walks in
    assert w.status()["analytics"]["people_inside"] == 2


def test_people_inside_never_goes_negative_and_staff_can_correct_it(make_settings, monkeypatch, clock):
    events = []
    w = entrance_worker(make_settings, events)
    drive(w, monkeypatch, clock, [1000 - 30 * i for i in range(30)])        # someone walks OUT first
    assert w.status()["analytics"]["people_inside"] == 0
    w.set_people_inside(5)
    drive(w, monkeypatch, clock, [200 + 30 * i for i in range(30)])        # one more enters
    assert w.status()["analytics"]["people_inside"] == 6
    assert any(k == "occupancy_correction" for k, _ in events)


def test_only_entrance_camera_can_set_occupancy(make_settings):
    spec = CameraSpec(id=1, key="c1", name="Hall", subtitle="", purpose="")
    with pytest.raises(ValueError):
        CameraWorker(spec, "demo", make_settings()).set_people_inside(3)


def test_occupancy_api_requires_running_monitoring(client):
    assert client.post("/api/occupancy/set", json={"people_inside": 3}).status_code == 409
    assert client.post("/api/occupancy/set", json={"people_inside": -1}).status_code == 422
    client.post("/api/monitoring/start")
    r = client.post("/api/occupancy/set", json={"people_inside": 4})
    assert r.status_code == 200 and r.json()["people_inside"] == 4
    client.post("/api/monitoring/stop")


def test_table_states_flow_through_worker(make_settings, monkeypatch, clock):
    regions = [{"id": "t1", "name": "Table 1", "x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 1.0}]
    spec = CameraSpec(id=3, key="c3", name="Seating", subtitle="", purpose="", role="table_status", table_regions=regions)
    w = CameraWorker(spec, "demo", make_settings(analytics_fps=100.0))
    frame = np.zeros((720, 1280, 3), np.uint8)
    cup = {"class_id": 41, "class_name": "cup", "confidence": 0.8, "xyxy": [100, 300, 160, 380]}

    def step(dets, seconds):
        monkeypatch.setattr(workers, "yolo_detect", lambda *a, **k: (dets, None))
        for _ in range(int(seconds)):
            w.process_frame(frame)
            clock["t"] += 1.0

    step([person(300), cup], 6)
    assert w.status()["analytics"]["occupied_tables"] == 1
    step([cup], 40)                                           # person left, cup stays
    a = w.status()["analytics"]
    assert (a["occupied_tables"], a["messy_tables"], a["empty_tables"]) == (0, 1, 0)
    assert a["clean_tables"] is None                          # never claims "clean"
    w.confirm_table_cleaned("t1")
    step([cup], 3)
    assert w.status()["analytics"]["empty_tables"] == 1


def test_live_source_kinds_and_masking():
    assert live_kind("rtsp://u:p@10.0.0.5:554/s") == "rtsp"
    assert live_kind("http://192.168.1.20:8080/video") == "http"
    assert live_kind("webcam:0") == "webcam"
    assert mask_url("webcam:0") == "webcam:0"
    assert "secret" not in mask_url("http://admin:secret@192.168.1.20:8080/video")


def test_webcam_url_is_a_valid_live_source(make_settings):
    s = make_settings(camera_01_rtsp="webcam:0")
    spec = CameraSpec(id=1, key="c1", name="Hall", subtitle="", purpose="")
    src = resolve_source(spec, "live", s)
    assert src.kind == "webcam" and src.configured and src.error is None


def test_bad_door_line_config_fails_early(make_settings, tmp_path):
    import yaml
    from app.config import load_cameras
    s = make_settings()
    raw = yaml.safe_load(open(s.cameras_config))
    next(c for c in raw["cameras"] if c["id"] == 2).update(line_orientation="vertical", entry_direction="top_to_bottom")
    open(s.cameras_config, "w").write(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="does not match"):
        load_cameras(s)
