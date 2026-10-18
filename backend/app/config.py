"""Settings + camera configuration. Credentials only ever come from env/.env."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
Mode = Literal["demo", "live"]
NUM_CAMERAS = 8


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")

    default_mode: Mode = "live"
    database_url: str = f"sqlite:///{ROOT / 'cafe_monitor.db'}"
    cameras_config: str = str(ROOT / "config" / "cameras.yaml")
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    ws_interval_seconds: float = 1.0
    jpeg_quality: int = 60
    max_stream_width: int = 640
    demo_playback_fps: float = 4.0      # cached demo proxies are encoded at this low frame rate
    yolo_model: str = "yolov8n.pt"
    yolo_backend: Literal["auto", "openvino", "pytorch"] = "auto"
    yolo_confidence: float = 0.20
    yolo_auto_download: bool = False
    person_detection_confidence: float = 0.12
    person_detection_imgsz: int = 640
    person_tiled_inference: bool = True
    person_track_max_age: float = 6.0
    max_result_frame_age_ms: int = 5000
    demo_use_proxies: bool = True
    demo_proxy_fps: float = 4.0
    demo_proxy_width: int = 640
    demo_proxy_dir: str = str(ROOT / "runtime" / "demo_proxy")
    analytics_fps: float = 0.2         # low-cost default when all eight feeds are selected on a dual-core CPU
    entrance_fps: float = 0.5           # raise only after measured CPU headroom and manual count comparison
    # Focus mode improves the selected feeds without forcing an unrealistic inference rate on this CPU.
    focus_max_cameras: int = 1          # conserve CPU by prioritizing one AI camera at a time
    focus_entrance_fps: float = 2.0
    focus_analytics_fps: float = 0.2
    display_fps: float = 1.5            # low-spec dashboard preview refresh rate
    yolo_device: str = "cpu"            # "cpu", "0" / "cuda:0" (NVIDIA GPU), "mps" (Apple)
    yolo_imgsz: int = 640
    entry_line_x_ratio: float = 0.5
    entry_direction: str = "left_to_right"
    line_band_ratio: float = 0.04       # dead zone on each side of the door line (share of frame size)
    counter_min_hits: int = 2           # a person must be tracked this many times before counting
    counter_cooldown_seconds: float = 1.0
    table_occupy_seconds: float = 3.0   # person must stay this long before a table is "occupied"
    table_leave_seconds: float = 20.0   # person must be gone this long before a table is free
    table_clutter_seconds: float = 10.0  # cups/plates must stay this long before "needs check"
    live_reconnect_min_seconds: float = 2.0
    live_reconnect_max_seconds: float = 30.0

    # Live Mode sources (secret). Empty = not configured.
    camera_01_rtsp: str = ""
    camera_02_rtsp: str = ""
    camera_03_rtsp: str = ""
    camera_04_rtsp: str = ""
    camera_05_rtsp: str = ""
    camera_06_rtsp: str = ""
    camera_07_rtsp: str = ""
    camera_08_rtsp: str = ""

    # Demo Mode per-camera file overrides (optional; win over cameras.yaml).
    demo_camera_01_file: str = ""
    demo_camera_02_file: str = ""
    demo_camera_03_file: str = ""
    demo_camera_04_file: str = ""
    demo_camera_05_file: str = ""
    demo_camera_06_file: str = ""
    demo_camera_07_file: str = ""
    demo_camera_08_file: str = ""

    def rtsp_url(self, cam_id: int) -> str:
        return getattr(self, f"camera_{cam_id:02d}_rtsp")

    def demo_override(self, cam_id: int) -> str:
        return getattr(self, f"demo_camera_{cam_id:02d}_file")


@dataclass
class CameraSpec:
    id: int
    key: str
    name: str
    subtitle: str
    purpose: str
    role: str = "none"
    analytics: list[str] = field(default_factory=list)
    demo_file: Path = Path()
    demo_loop: bool = True
    table_regions: list[dict] = field(default_factory=list)
    line_x_ratio: float = 0.5           # position of the door line (0..1). Used for vertical AND horizontal lines.
    entry_direction: str = "left_to_right"
    line_orientation: str = "vertical"  # "vertical" (people walk left/right) or "horizontal" (people walk up/down)


def _resolve(p: str) -> Path:
    path = Path(p).expanduser()
    path = path if path.is_absolute() else ROOT / path
    return _match_case_insensitive(path)


def _match_case_insensitive(path: Path) -> Path:
    """Linux is case-sensitive, Windows is not: 'sample.mp4' vs 'Sample.mp4' must not break Demo Mode.

    If the exact path exists it is returned unchanged. Otherwise we look for ONE file in the same
    folder whose name differs only by capitalisation. Anything ambiguous or missing is left alone so
    resolve_source() still reports a clear 'file not found' error (never a silent substitution).
    """
    try:
        if path.exists() or not path.parent.is_dir():
            return path
        matches = [c for c in path.parent.iterdir() if c.name.lower() == path.name.lower()]
        return matches[0] if len(matches) == 1 else path
    except OSError:
        return path


def load_cameras(settings: Settings) -> list[CameraSpec]:
    config_path = Path(settings.cameras_config).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    raw = yaml.safe_load(config_path.read_text()) or {}
    default_loop = bool(raw.get("demo_loop_default", True))
    specs: list[CameraSpec] = []
    for item in raw.get("cameras", []):
        cid = int(item["id"])
        demo = item.get("demo", {}) or {}
        file_ = settings.demo_override(cid) or demo.get("file", "")
        specs.append(
            CameraSpec(
                id=cid,
                key=item.get("key", f"camera_{cid:02d}"),
                name=item["name"],
                subtitle=item.get("subtitle", ""),
                purpose=item.get("purpose", ""),
                role=str(item.get("role", "none")),
                analytics=list(item.get("analytics", [])),
                demo_file=_resolve(file_) if file_ else Path(),
                demo_loop=bool(demo.get("loop", default_loop)),
                table_regions=list(item.get("table_regions", [])),
                line_x_ratio=float(item.get("line_ratio", item.get("line_x_ratio", settings.entry_line_x_ratio))),
                entry_direction=str(item.get("entry_direction", settings.entry_direction)),
                line_orientation=str(item.get("line_orientation", "vertical")),
            )
        )
        if str(item.get("role", "none")) == "entry_exit":  # validate the entrance line configuration
            from .tracking import LineCounter
            LineCounter(specs[-1].line_orientation, specs[-1].entry_direction, specs[-1].line_x_ratio)
    if sorted(s.id for s in specs) != list(range(1, NUM_CAMERAS + 1)):
        raise ValueError(f"cameras config must define camera ids 1..{NUM_CAMERAS}")
    return sorted(specs, key=lambda s: s.id)
