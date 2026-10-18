export type Mode = "demo" | "live";

export interface CameraInfo {
  id: number;
  key: string;
  name: string;
  subtitle: string;
  purpose: string;
  role: string;
  analytics: string[];
  demo: { file: string | null; exists: boolean; loop: boolean };
  live: { configured: boolean };
  table_regions?: { id: string; name: string }[];
}

export interface CameraStatus {
  id: number;
  name: string;
  mode: Mode;
  label: "DEMO" | "LIVE" | "OFFLINE";
  recorded: boolean;
  source_kind: string;
  source_configured: boolean;
  source_path: string | null;
  playback_state: string;
  processing_state: string;
  error: string | null;
  connected: boolean;
  position_sec: number | null;
  duration_sec: number | null;
  fps: number | null;
  frames_played: number;
  selected?: boolean; // false = not chosen for this run
  frames_processed: number; // successful inferences only
  frames_seen?: number;
  inference_errors?: number;
  last_inference_ms?: number | null;
  evidence_available?: boolean;
  evidence_updated_at?: string | null;
  evidence_position_sec?: number | null;
  result_frame_age_ms?: number | null;
  loops_completed?: number;
  reconnect_attempts: number;
  sync_lag_frames: number;
  last_frame_at: string | null;
  analytics: { available: boolean; status?: "idle" | "waiting" | "ok" | "error" | "stale" | "not_configured" | "not_selected"; note?: string; people_count: number | null; entries: number | null; exits: number | null; people_inside?: number | null; occupied_tables?: number | null; empty_tables?: number | null; messy_tables: number | null; clean_tables: number | null; cup_count: number | null; detection_count?: number | null; zone_dwell_avg_seconds?: number | null; zone_dwell_max_seconds?: number | null; tracked_people?: number | null; updated_at?: string | null };
}

export interface MonitoringStatus {
  mode: Mode;
  running: boolean;
  session_id: number | null;
  started_at: string | null;
  cameras: CameraStatus[];
  selected_camera_ids?: number[];
  focus_mode?: boolean;
  already_running?: boolean;
}

export interface Summary {
  mode: Mode;
  monitoring_running: boolean;
  session_id: number | null;
  cameras_total: number;
  cameras_playing: number;
  cameras_with_problems: number;
  frames_played_session: number;
  frames_processed_session: number;
  sessions_total: number;
  frames_processed_lifetime: number;
  detection: {
    available: boolean;
    note: string;
    people_count: number | null;
    entries: number | null;
    exits: number | null;
    people_inside?: number | null;
    occupied_tables?: number | null;
    empty_tables?: number | null;
    messy_tables: number | null;
    clean_tables: number | null;
    cup_count: number | null;
    zone_dwell_avg_seconds?: number | null;
    zone_dwell_max_seconds?: number | null;
    tracked_people?: number | null;
  };
  integrations?: { cashier_pos: { available: boolean; note: string } };
}

export interface SystemResources {
  timestamp: string;
  process_cpu_percent: number;
  process_ram_bytes: number;
  host_cpu_percent: number;
  host_ram_used_bytes: number;
  host_ram_total_bytes: number;
  container_ram_used_bytes: number;
  container_ram_limit_bytes: number | null;
  cpu_count: number;
  note: string;
  error?: string;
}
