from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["demo", "live"] | None = None
    cameras: list[int] | None = None   # None = all cameras; e.g. [7] = analyze only camera 7
    opening_occupancy: int | None = Field(default=None, ge=0, le=1000)


class ModeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["demo", "live"]


class CameraStatus(BaseModel):
    id: int
    name: str
    mode: Literal["demo", "live"]
    label: Literal["DEMO", "LIVE", "OFFLINE"]
    recorded: bool
    source_kind: str
    source_configured: bool
    source_path: str | None
    playback_state: str
    processing_state: str
    error: str | None
    connected: bool
    position_sec: float | None
    duration_sec: float | None
    fps: float | None
    frames_played: int
    frames_processed: int          # successful inferences only
    frames_seen: int = 0           # frames handed to the processor (not inferences)
    inference_errors: int = 0
    last_inference_ms: float | None = None
    evidence_available: bool = False
    evidence_updated_at: str | None = None
    evidence_position_sec: float | None = None
    result_frame_age_ms: float | None = None
    loops_completed: int = 0       # demo mode only: how many times the clip has replayed
    reconnect_attempts: int
    sync_lag_frames: int
    last_frame_at: str | None
    selected: bool = True          # False = camera not chosen for this run (no decoding, no AI)
    analytics: dict = Field(default_factory=dict)


class MonitoringStatus(BaseModel):
    mode: Literal["demo", "live"]
    running: bool
    session_id: int | None
    started_at: str | None
    cameras: list[CameraStatus]
    selected_camera_ids: list[int] = Field(default_factory=list)
    focus_mode: bool = False       # few cameras selected -> higher AI rate
    already_running: bool = False


class OccupancyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    people_inside: int = Field(ge=0, le=1000)
