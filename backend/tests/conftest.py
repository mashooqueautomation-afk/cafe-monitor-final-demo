from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def write_video(path: Path, frames=20, fps=20, w=64, h=48) -> Path:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(frames):
        vw.write(np.full((h, w, 3), (i * 10) % 255, np.uint8))
    vw.release()
    return path


@pytest.fixture()
def videos(tmp_path):
    return [write_video(tmp_path / f"v{i}.mp4") for i in range(1, 9)]


@pytest.fixture()
def make_settings(tmp_path, videos):
    def factory(loop=True, **overrides) -> Settings:
        roles = {1: "none", 2: "entry_exit", 3: "table_status", 4: "none", 5: "none", 6: "cup_detection", 7: "people_count", 8: "none"}
        cams = [{"id": i, "name": f"Cam {i}", "subtitle": "", "purpose": "t", "role": roles[i], "analytics": [],
                 "demo": {"file": str(videos[i - 1]), "loop": loop}} for i in range(1, 9)]
        cfg = tmp_path / "cameras.yaml"
        cfg.write_text(yaml.safe_dump({"cameras": cams}))
        overrides.setdefault("default_mode", "demo")
        return Settings(_env_file=None, cameras_config=str(cfg), database_url=f"sqlite:///{tmp_path / 't.db'}",
                        live_reconnect_min_seconds=0.2, live_reconnect_max_seconds=0.5, **overrides)
    return factory


@pytest.fixture()
def client(make_settings):
    with TestClient(create_app(make_settings())) as c:
        yield c


def cam_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("cam0") and t.is_alive()]


def wait_for(pred, timeout=8.0, step=0.1):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(step)
    return False
