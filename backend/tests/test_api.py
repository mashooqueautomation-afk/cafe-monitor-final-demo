import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app, mjpeg_frames

from .conftest import cam_threads, wait_for


def statuses(client):
    return client.get("/api/monitoring/status").json()["cameras"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_cameras_config_has_eight_and_hides_secrets(make_settings):
    s = make_settings(camera_01_rtsp="rtsp://admin:topsecret@127.0.0.1:1/s")
    with TestClient(create_app(s)) as c:
        body = c.get("/api/cameras").text
        data = json.loads(body)
        assert [x["id"] for x in data] == [1, 2, 3, 4, 5, 6, 7, 8]
        assert data[0]["live"] == {"configured": True}
        assert data[6]["id"] == 7 and data[6]["role"] == "people_count"  # test fixture role
        assert data[5]["role"] == "cup_detection"
        assert data[7]["role"] == "none" and data[7]["demo"]["exists"] is True
        assert "topsecret" not in body and "rtsp://" not in body


def test_camera_status_404_and_ok(client):
    assert client.get("/api/cameras/99/status").status_code == 404
    st = client.get("/api/cameras/1/status").json()
    assert st["playback_state"] == "idle" and st["label"] == "DEMO" and st["recorded"] is True


def test_start_validation(client):
    assert client.post("/api/monitoring/start", json={"mode": "bogus"}).status_code == 422
    assert client.post("/api/monitoring/start", json={"nope": 1}).status_code == 422
    assert client.post("/api/monitoring/start", json={"opening_occupancy": 1001}).status_code == 422


def test_start_plays_all_eight_and_is_idempotent(client):
    r = client.post("/api/monitoring/start")
    assert r.status_code == 200 and r.json()["already_running"] is False
    assert wait_for(lambda: all(c["frames_played"] > 3 for c in statuses(client)))
    assert len(cam_threads()) == 16  # 8 readers + 8 processors
    for _ in range(3):
        again = client.post("/api/monitoring/start").json()
        assert again["already_running"] is True
    assert len(cam_threads()) == 16  # repeated Start never adds workers
    cams = statuses(client)
    assert all(c["label"] == "DEMO" and c["recorded"] and c["playback_state"] == "playing" for c in cams)
    # OLD assertion expected "running" everywhere, which hid the fact that no model was loaded.
    # Now: roles without analytics are "not_configured"; roles that need the (absent) model report the error.
    assert wait_for(lambda: {c["id"]: c["processing_state"] for c in statuses(client)} ==
                    {1: "not_configured", 2: "inference_error", 3: "not_configured", 4: "not_configured",
                     5: "not_configured", 6: "inference_error", 7: "inference_error", 8: "not_configured"})
    assert all(c["frames_processed"] == 0 for c in statuses(client))   # nothing was actually inferred
    client.post("/api/monitoring/stop")


def test_missing_and_invalid_video_do_not_stop_others(make_settings, tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"this is not a video")
    s = make_settings(demo_camera_03_file=str(tmp_path / "nope.mp4"), demo_camera_04_file=str(bad))
    with TestClient(create_app(s)) as c:
        idle = {x["id"]: x for x in c.get("/api/monitoring/status").json()["cameras"]}
        assert "not found" in idle[3]["error"]  # warned even before Start
        c.post("/api/monitoring/start")
        assert wait_for(lambda: all(x["frames_played"] > 2 for x in statuses(c) if x["id"] not in (3, 4)))
        st = {x["id"]: x for x in statuses(c)}
        assert st[3]["playback_state"] == "error" and "not found" in st[3]["error"]
        assert st[4]["playback_state"] == "error" and "invalid" in st[4]["error"].lower()
        assert [i for i, x in st.items() if x["playback_state"] == "playing"] == [1, 2, 5, 6, 7, 8]
        c.post("/api/monitoring/stop")


def test_stop_releases_threads_and_persists_session(client, monkeypatch):
    import app.workers as workers
    monkeypatch.setattr(workers, "yolo_detect", lambda *a, **k: ([], None))   # a working (empty) detector
    client.post("/api/monitoring/start")
    assert wait_for(lambda: all(c["frames_played"] > 3 for c in statuses(client)))
    t0 = __import__("time").monotonic()
    r = client.post("/api/monitoring/stop").json()
    assert __import__("time").monotonic() - t0 < 5
    assert r["running"] is False and cam_threads() == []
    assert all(c["playback_state"] == "idle" for c in r["cameras"])
    s = client.get("/api/analytics/summary").json()
    assert s["sessions_total"] == 1 and s["frames_processed_lifetime"] > 0
    assert client.post("/api/monitoring/stop").status_code == 200  # idempotent


def test_restart_creates_new_session_without_duplicates(client):
    sid = client.post("/api/monitoring/start").json()["session_id"]
    assert wait_for(lambda: all(c["frames_played"] > 1 for c in statuses(client)))
    r = client.post("/api/monitoring/restart").json()
    assert r["running"] and r["session_id"] != sid and len(cam_threads()) == 16
    client.post("/api/monitoring/stop")


def test_mode_switch_rules(client):
    client.post("/api/monitoring/start")
    assert client.post("/api/monitoring/mode", json={"mode": "live"}).status_code == 409
    assert client.post("/api/monitoring/start", json={"mode": "live"}).status_code == 409
    client.post("/api/monitoring/stop")
    assert client.post("/api/monitoring/mode", json={"mode": "live"}).json()["mode"] == "live"


def test_live_mode_unconfigured_is_offline_with_no_demo_fallback(client):
    client.post("/api/monitoring/mode", json={"mode": "live"})
    client.post("/api/monitoring/start")
    assert wait_for(lambda: all(c["playback_state"] == "offline" for c in statuses(client)))
    for c in statuses(client):
        assert c["label"] == "OFFLINE" and c["recorded"] is False
        assert c["frames_played"] == 0 and c["source_path"] is None
        assert f"CAMERA_{c['id']:02d}_RTSP" in c["error"]
    client.post("/api/monitoring/stop")


def test_live_unreachable_stream_goes_offline_hides_credentials_and_stops_fast(make_settings):
    s = make_settings(default_mode="live", camera_01_rtsp="rtsp://admin:topsecret@127.0.0.1:1/stream")
    with TestClient(create_app(s)) as c:
        c.post("/api/monitoring/start")
        assert wait_for(lambda: statuses(c)[0]["playback_state"] == "offline", timeout=15)
        raw = json.dumps(c.get("/api/monitoring/status").json()) + c.get("/api/analytics/summary").text
        assert "topsecret" not in raw and "admin" not in raw
        assert statuses(c)[0]["label"] == "OFFLINE" and statuses(c)[1]["error"].startswith("RTSP URL not configured")
        t0 = __import__("time").monotonic()
        c.post("/api/monitoring/stop")
        assert __import__("time").monotonic() - t0 < 6 and cam_threads() == []


def test_analytics_summary_invents_nothing(client):
    d = client.get("/api/analytics/summary").json()["detection"]
    assert d["available"] is False
    assert all(v is None for k, v in d.items() if k not in ("available", "note"))


def test_mjpeg_generator_emits_jpeg(client):
    client.post("/api/monitoring/start")
    worker = client.app.state.manager.workers[1]
    assert wait_for(lambda: worker.buffer.latest()[1] is not None)

    async def first():
        gen = mjpeg_frames(worker)
        chunk = await gen.__anext__()
        await gen.aclose()
        return chunk

    chunk = asyncio.run(first())
    assert chunk.startswith(b"--frame") and b"\xff\xd8" in chunk
    assert client.get("/api/cameras/1/status").status_code == 200
    client.post("/api/monitoring/stop")
    assert client.get("/api/cameras/1/stream").status_code == 409  # not running -> no stream


def test_websocket_pushes_status(client):
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "update" and len(msg["status"]["cameras"]) == 8 and "summary" in msg


def test_system_resources_endpoint(client):
    r = client.get("/api/system/resources")
    assert r.status_code == 200
    body = r.json()
    assert "process_cpu_percent" in body
    assert "process_ram_bytes" in body
    assert "container_ram_used_bytes" in body


def test_table_cleaned_event_requires_configured_region(make_settings, tmp_path):
    import yaml
    settings = make_settings()
    cfg = Path(settings.cameras_config)
    raw = yaml.safe_load(cfg.read_text())
    next(c for c in raw["cameras"] if c["id"] == 3)["table_regions"] = [
        {"id": "table_1", "name": "Table 1", "x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.4}
    ]
    cfg.write_text(yaml.safe_dump(raw))
    with TestClient(create_app(settings)) as c:
        r = c.post("/api/tables/table_1/cleaned")
        assert r.status_code == 200 and r.json()["status"] == "staff_confirmed_clean"
        events = c.get("/api/events?limit=10").json()
        assert any(e["event_type"] == "table_cleaned" and e["payload"]["table_id"] == "table_1" for e in events)


def test_evidence_endpoint_serves_last_successful_inference_frame(client, monkeypatch):
    import app.workers as workers
    person = {"class_id": 0, "class_name": "person", "confidence": 0.9, "xyxy": [2, 2, 15, 20]}
    monkeypatch.setattr(workers, "yolo_detect", lambda *a, **k: ([person], None))
    client.post("/api/monitoring/start", json={"cameras": [2]})
    worker = client.app.state.manager.workers[2]
    assert wait_for(lambda: worker.status()["evidence_available"], timeout=5)
    response = client.get("/api/cameras/2/evidence")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert response.content.startswith(b"\xff\xd8")
    assert response.headers.get("X-Evidence-Updated-At")
    client.post("/api/monitoring/stop")
