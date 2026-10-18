"""Diagnostics dashboard backend. Uses a FAKE detector (real YOLO is not needed to test the logic/flow)."""
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.diag.main import create_app as create_diag

from .conftest import wait_for


class FakeDetect:
    """Deterministic: bigger imgsz sees more people (like a real small model), a box walks right->left per call."""
    def __init__(self, delay=0.0):
        self.calls, self.delay, self.x = 0, delay, 60

    def __call__(self, frame, model, conf, auto, device="cpu", imgsz=640, classes=None):
        self.calls += 1
        time.sleep(self.delay)
        self.x = self.x - 3 if self.x > 4 else 60
        if classes == [0]:
            n = 1 if imgsz <= 320 else 3
            return [{"class_id": 0, "class_name": "person", "confidence": .9, "xyxy": [self.x + i * 2, 5, self.x + 8 + i * 2, 40]} for i in range(n)], None
        if classes in ([40, 41], None):
            return [{"class_id": 41, "class_name": "cup", "confidence": .8, "xyxy": [5, 5, 20, 30]}], None
        return [], None


@pytest.fixture()
def diag(make_settings, tmp_path):
    weights = tmp_path / "w.pt"; weights.write_text("x")

    def build(detect=None, **over):
        s = make_settings(yolo_model=str(weights), **over)
        return TestClient(create_diag(s, detect_fn=detect or FakeDetect(), backend_url=""))
    return build


def run_job(c, kind, params=None, timeout=30):
    r = c.post(f"/api/jobs/{kind}", json=params or {})
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    assert wait_for(lambda: c.get(f"/api/jobs/{jid}").json()["status"] != "running", timeout=timeout)
    return c.get(f"/api/jobs/{jid}").json()


def test_page_and_health(diag):
    c = diag()
    assert "Cafe Monitor" in c.get("/").text
    assert c.get("/health").json() == {"status": "ok", "model_found": True}
    assert len(c.get("/api/cameras").json()) == 8


def test_health_checks_flag_real_problems(diag, make_settings, tmp_path):
    c = diag(entrance_fps=1.0, yolo_imgsz=320)
    checks = {x["title"]: x for x in c.get("/api/checks").json()}
    assert any(t.startswith("YOLO weights available") for t in checks)
    assert any(t.startswith("ENTRANCE_FPS=1.0 is low") and x["level"] == "warn" for t, x in checks.items())
    assert any("YOLO_IMGSZ=320" in t and x["level"] == "warn" for t, x in checks.items())
    assert any("Cam 03" in t and "table regions are empty" in t for t in checks)          # role table_status, no regions
    assert any("Cam 01" in t and "analytics role not configured" in t and x["level"] == "info" for t, x in checks.items())
    # missing weights + live mode without RTSP are failures, not silent
    bad = TestClient(create_diag(make_settings(yolo_model=str(tmp_path / "nope.pt"), default_mode="live"), detect_fn=FakeDetect(), backend_url=""))
    levels = {x["title"]: x["level"] for x in bad.get("/api/checks").json()}
    assert levels["YOLO weights missing"] == "fail"
    assert any(t.startswith("DEFAULT_MODE=live") and lv == "fail" for t, lv in levels.items())


def test_camera_job_reports_crowd_and_small_size_misses(diag):
    j = run_job(diag(), "camera", {"camera_id": 7})
    assert j["status"] == "done", j.get("error")
    r = j["result"]
    assert r["sizes"]["320"]["max"] == 1 and r["sizes"]["640"]["max"] == 3
    assert r["density"].startswith("moderate")
    assert any("At imgsz=320, the model detects" in f for f in r["findings"])           # 320 misses people vs 640
    assert r["thumbnails"] and r["thumbnails"][0]["image"].startswith("data:image/jpeg;base64,")


def test_camera_job_for_cup_camera_counts_cup_boxes(diag):
    r = run_job(diag(), "camera", {"camera_id": 6})["result"]
    assert r["cup_boxes_per_frame"] and all(n == 1 for n in r["cup_boxes_per_frame"])


def test_benchmark_gives_scenarios_with_verdicts(diag):
    j = run_job(diag(), "benchmark")
    assert j["status"] == "done", j.get("error")
    r = j["result"]
    assert set(r["inference_ms_by_imgsz"]) == {320, 480, 640} or set(r["inference_ms_by_imgsz"]) == {"320", "480", "640"}
    assert len(r["scenarios"]) == 5 and all(s["verdict"] in ("OK", "TIGHT", "OVERLOAD") for s in r["scenarios"])
    assert r["decode_ms_per_frame"] > 0


def test_cups_job_on_photos(diag, tmp_path, monkeypatch):
    d = tmp_path / "cups"; d.mkdir()
    cv2.imwrite(str(d / "a.jpg"), np.full((60, 80, 3), 120, np.uint8))
    cv2.imwrite(str(d / "b.png"), np.full((60, 80, 3), 40, np.uint8))
    monkeypatch.setenv("DIAG_CUP_DIR", str(d))
    r = run_job(diag(), "cups")["result"]
    assert r["photos"] == 2 and r["photos_with_cup_like_detection"] == 2
    assert "ceramic vs glass vs takeaway" in r["note"]


def test_entrance_job_compares_lines_and_scores_against_manual_count(diag):
    j = run_job(diag(), "entrance", {"expected_in": 1, "expected_out": 0, "imgsz": 320})
    assert j["status"] == "done", j.get("error")
    r = j["result"]
    labels = {row["label"] for row in r["table"]}
    assert "CURRENT config" in labels and "horizontal" in labels and "vertical" in labels
    assert all("error" in row for row in r["table"]) and r["preview"].startswith("data:image/jpeg")


def test_only_one_heavy_job_at_a_time_and_bad_params_rejected(diag):
    c = diag(detect=FakeDetect(delay=0.15))
    first = c.post("/api/jobs/camera", json={"camera_id": 7}).json()["job_id"]
    assert c.post("/api/jobs/camera", json={"camera_id": 7}).status_code == 409          # busy
    assert wait_for(lambda: c.get(f"/api/jobs/{first}").json()["status"] == "done", timeout=60)
    assert c.post("/api/jobs/camera", json={"camera_id": 7, "samples": 999}).status_code == 422
    assert c.post("/api/jobs/camera", json={}).status_code == 422
    assert c.post("/api/jobs/nonsense", json={}).status_code == 404


def test_missing_weights_gives_clear_job_error(make_settings, tmp_path):
    c = TestClient(create_diag(make_settings(yolo_model=str(tmp_path / "missing.pt")), detect_fn=FakeDetect(), backend_url=""))
    j = run_job(c, "camera", {"camera_id": 7})
    assert j["status"] == "error" and "weights not found" in j["error"]
