from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .config import Settings, load_cameras
from .db import AnalyticsEvent, make_engine, make_session_factory
from .schemas import ModeRequest, MonitoringStatus, OccupancyRequest, StartRequest
from .workers import CameraWorker, ModeChangeError, MonitoringManager
from .system_metrics import get_system_resources

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
BOUNDARY = "frame"


async def mjpeg_frames(worker: CameraWorker):
    """Yield multipart MJPEG chunks from the worker's buffer until it stops."""
    last = 0
    while worker.is_running():
        fid, jpeg = worker.buffer.latest()
        if jpeg is not None and fid != last:
            last = fid
            yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n").encode() + jpeg + b"\r\n"
        else:
            await asyncio.sleep(0.01)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    cameras = load_cameras(settings)
    engine = make_engine(settings.database_url)
    manager = MonitoringManager(settings, cameras, make_session_factory(engine))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await asyncio.to_thread(manager.shutdown)  # release every capture on exit
        engine.dispose()

    app = FastAPI(title="AUTONIQ Café Monitor", version="0.1.0", lifespan=lifespan)
    app.state.manager, app.state.settings = manager, settings
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                       allow_methods=["*"], allow_headers=["*"])

    def status_model(already: bool = False) -> MonitoringStatus:
        return MonitoringStatus(**manager.snapshot(), already_running=already)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": app.version, "mode": manager.mode, "monitoring_running": manager.running}

    @app.get("/api/cameras")
    def list_cameras():
        out = []
        for c in cameras:
            demo_exists = c.demo_file.is_file() if str(c.demo_file) not in ("", ".") else False
            out.append({"id": c.id, "key": c.key, "name": c.name, "subtitle": c.subtitle, "purpose": c.purpose, "role": c.role,
                        "analytics": c.analytics,
                        "table_regions": [{"id": str(r.get("id", r.get("name", f"table_{i+1}"))), "name": r.get("name", f"Table {i+1}")}
                                          for i, r in enumerate(c.table_regions)] if c.role == "table_status" else [],
                        "demo": {"file": str(c.demo_file) if str(c.demo_file) != "." else None, "exists": demo_exists, "loop": c.demo_loop},
                        "live": {"configured": bool(settings.rtsp_url(c.id))}})  # never the URL
        return out

    @app.get("/api/cameras/{camera_id}/status")
    def camera_status(camera_id: int):
        st = manager.camera_status(camera_id)
        if st is None:
            raise HTTPException(404, f"Camera {camera_id} does not exist (valid ids: {min(c.id for c in cameras)}-{max(c.id for c in cameras)})")
        return st

    @app.get("/api/cameras/{camera_id}/stream")
    def camera_stream(camera_id: int):
        if camera_id not in {c.id for c in cameras}:
            raise HTTPException(404, f"Camera {camera_id} does not exist")
        worker = manager.workers.get(camera_id)
        if worker is None or not worker.is_running():
            raise HTTPException(409, "Monitoring is not running for this camera")
        return StreamingResponse(mjpeg_frames(worker), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/{camera_id}/snapshot")
    def camera_snapshot(camera_id: int):
        """Latest JPEG as a short-lived request.

        Dashboard tiles poll this endpoint instead of holding eight endless MJPEG streams open.
        This avoids exhausting per-host HTTP/1.1 connections and leaves /stream for a single
        expanded/fullscreen viewer.
        """
        if camera_id not in {c.id for c in cameras}:
            raise HTTPException(404, f"Camera {camera_id} does not exist")
        worker = manager.workers.get(camera_id)
        if worker is None or not worker.is_running():
            raise HTTPException(409, "Monitoring is not running for this camera")
        _, jpeg = worker.buffer.latest()
        if jpeg is None:
            raise HTTPException(503, "No frame available yet")
        return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/{camera_id}/evidence")
    def camera_evidence(camera_id: int):
        """Annotated snapshot from the exact frame used by the most recent successful inference."""
        if camera_id not in {c.id for c in cameras}:
            raise HTTPException(404, f"Camera {camera_id} does not exist")
        worker = manager.workers.get(camera_id)
        if worker is None or not worker.is_running():
            raise HTTPException(409, "Monitoring is not running for this camera")
        jpeg, updated_at, position_sec = worker.evidence_snapshot()
        if jpeg is None:
            raise HTTPException(503, "No successful AI evidence frame is available yet")
        headers = {"Cache-Control": "no-store"}
        if updated_at:
            headers["X-Evidence-Updated-At"] = updated_at
        if position_sec is not None:
            headers["X-Evidence-Position-Sec"] = str(position_sec)
        return Response(content=jpeg, media_type="image/jpeg", headers=headers)

    @app.get("/api/monitoring/status", response_model=MonitoringStatus)
    def monitoring_status():
        return status_model()

    @app.post("/api/monitoring/start", response_model=MonitoringStatus)
    def start(body: StartRequest | None = Body(default=None)):
        try:
            started = manager.start(body.mode if body else None, body.cameras if body else None,
                                    body.opening_occupancy if body else None)
        except ModeChangeError as e:
            raise HTTPException(409, str(e))
        return status_model(already=not started)

    @app.post("/api/monitoring/stop", response_model=MonitoringStatus)
    def stop():
        manager.stop()
        return status_model()

    @app.post("/api/monitoring/restart", response_model=MonitoringStatus)
    def restart():
        manager.restart()
        return status_model()

    @app.post("/api/monitoring/mode", response_model=MonitoringStatus)
    def set_mode(body: ModeRequest):
        try:
            manager.set_mode(body.mode)
        except ModeChangeError as e:
            raise HTTPException(409, str(e))
        return status_model()

    @app.get("/api/system/resources")
    def system_resources():
        return get_system_resources()

    @app.get("/api/analytics/summary")
    def analytics_summary():
        return manager.summary()


    @app.get("/api/events")
    def list_events(limit: int = Query(default=100, ge=1, le=500)):
        from sqlalchemy import select
        with manager._sf() as db:
            rows = db.scalars(select(AnalyticsEvent).order_by(AnalyticsEvent.created_at.desc()).limit(limit)).all()
            return [{"id": row.id, "camera_id": row.camera_id, "event_type": row.event_type,
                     "created_at": row.created_at.isoformat(), "payload": row.payload} for row in rows]

    @app.post("/api/tables/{table_id}/cleaned")
    def mark_table_cleaned(table_id: str):
        """Record a staff-confirmed cleaning event; CV does not auto-assert cleanliness."""
        camera = next((c for c in cameras if c.role == "table_status"), None)
        if camera is None:
            raise HTTPException(404, "Seating-area camera is not configured")
        valid = {str(r.get("id", r.get("name", f"table_{i+1}"))) for i, r in enumerate(camera.table_regions)}
        valid.update(str(r.get("name", f"Table {i+1}")) for i, r in enumerate(camera.table_regions))
        if table_id not in valid:
            raise HTTPException(404, "Table is not defined. Configure table_regions in config/cameras.yaml first.")
        from .workers import utcnow
        manager.confirm_table_cleaned(table_id)   # tell the live table monitor (if running)
        with manager._sf() as db:
            event = AnalyticsEvent(camera_id=camera.id, event_type="table_cleaned", created_at=utcnow(),
                                   payload={"table_id": table_id, "confirmed_by": "operator"})
            db.add(event)
            db.commit()
            db.refresh(event)
            return {"ok": True, "event_id": event.id, "table_id": table_id,
                    "status": "staff_confirmed_clean", "created_at": event.created_at.isoformat()}

    @app.post("/api/occupancy/set")
    def set_occupancy(body: OccupancyRequest):
        """Staff correction: 'there are really N people inside right now'. Fixes drift in entries - exits."""
        try:
            n = manager.set_people_inside(body.people_inside)
        except ModeChangeError as e:
            raise HTTPException(409, str(e))
        return {"ok": True, "people_inside": n}

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        try:
            while True:
                payload = await asyncio.to_thread(lambda: {"type": "update", "status": manager.snapshot(), "summary": manager.summary(), "resources": get_system_resources()})
                await socket.send_json(payload)
                await asyncio.sleep(settings.ws_interval_seconds)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


app = create_app()
