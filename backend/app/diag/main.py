"""Diagnostics dashboard (separate localhost port). Read-only helper: never touches the monitoring session or weights."""
from __future__ import annotations

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from ..config import Settings, load_cameras
from .engine import Engine, JobBusy


def create_app(settings: Settings | None = None, detect_fn=None, backend_url: str | None = None) -> FastAPI:
    settings = settings or Settings()
    cameras = load_cameras(settings)
    engine = Engine(settings, cameras, detect_fn=detect_fn, backend_url=backend_url)
    app = FastAPI(title="Cafe Monitor Diagnostics", version="1.0")
    app.state.engine = engine
    page = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return page

    @app.get("/health")
    def health():
        return {"status": "ok", "model_found": engine.model_path() is not None}

    @app.get("/api/cameras")
    def cams():
        return [{"id": c.id, "name": c.name, "role": c.role, "video_found": c.demo_file.is_file()} for c in cameras]

    @app.get("/api/checks")
    def checks():
        return engine.health_checks()

    @app.post("/api/jobs/{kind}")
    def start(kind: str, params: dict = Body(default_factory=dict)):
        try:
            return {"job_id": engine.start_job(kind, _clean(kind, params))}
        except KeyError:
            raise HTTPException(404, f"Unknown job '{kind}'")
        except JobBusy as e:
            raise HTTPException(409, str(e))

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        j = engine.jobs.get(job_id)
        if j is None:
            raise HTTPException(404, "Unknown job")
        return {k: v for k, v in j.items() if k != "trace" or j["status"] == "error"}

    return app


def _clean(kind: str, p: dict) -> dict:
    """Whitelist + coerce job parameters (the page is the only caller, but never trust input)."""
    def num(key, cast, lo, hi, default=None):
        if key not in p or p[key] in (None, ""):
            return default
        v = cast(p[key])
        if not lo <= v <= hi:
            raise HTTPException(422, f"{key} must be between {lo} and {hi}")
        return v
    if kind == "camera":
        out = {"camera_id": num("camera_id", int, 1, 99), "samples": num("samples", int, 2, 16, 8)}
        if out["camera_id"] is None:
            raise HTTPException(422, "camera_id required")
        return out
    if kind == "entrance":
        return {k: v for k, v in {"imgsz": num("imgsz", int, 160, 1280, 480), "sim_fps": num("sim_fps", float, 1, 8, 5.0),
                                  "expected_in": num("expected_in", int, 0, 500), "expected_out": num("expected_out", int, 0, 500)}.items()
                if v is not None}
    if kind == "cups":
        return {"imgsz": num("imgsz", int, 160, 1280, 640)}
    return {}

