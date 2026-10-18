import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api } from "../api";
import type { CameraInfo, CameraStatus } from "../types";

interface Props {
  info: CameraInfo;
  status: CameraStatus;
  running: boolean;
  sessionId: number | null;
}

const clock = (sec: number | null) => {
  if (sec === null) return "--:--:--";
  const s = Math.floor(sec);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(Math.floor(s / 3600))}:${p(Math.floor((s % 3600) / 60))}:${p(s % 60)}`;
};

const stamp = (iso: string | null) => {
  if (!iso) return "no frame yet";
  const d = new Date(iso);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
};

function overlayMessage(s: CameraStatus, running: boolean): { text: string; tone: "error" | "info" } | null {
  if (s.playback_state === "error") return { text: s.error ?? "Video error", tone: "error" };
  if (!running) {
    return s.error
      ? { text: s.error, tone: "error" }
      : { text: "Press Start monitoring to begin", tone: "info" };
  }
  if (s.playback_state === "offline") return { text: s.error ?? "Camera offline", tone: "error" };
  if (["starting", "connecting", "reconnecting"].includes(s.playback_state) && s.frames_played === 0)
    return { text: s.playback_state === "starting" ? "Opening source…" : "Connecting to camera…", tone: "info" };
  return null;
}

function metricFor(label: string, a: CameraStatus["analytics"]): string {
  // A stale number is evidence from an older source frame, never a current measurement.
  if (a.status === "stale") {
    const l = label.toLowerCase();
    if (l.includes("people") && a.people_count != null) return `${a.people_count} · STALE`;
    return "STALE";
  }
  // Distinct states: never show a missing value as 0, and never show "ok" just because video plays.
  if (a.status === "not_selected") return "—";
  if (a.status === "not_configured") return "not set up";
  if (a.status === "error") return "error";
  if (!a.available) return "waiting";
  const l = label.toLowerCase();
  let value: number | null | undefined;
  if (l.includes("people")) value = a.people_count;
  else if (l.includes("inside")) value = a.people_inside;
  else if (l === "in" || l.includes("entries")) value = a.entries;
  else if (l === "out" || l.includes("exits")) value = a.exits;
  else if (l.includes("occupied")) value = a.occupied_tables;
  else if (l.includes("empty")) value = a.empty_tables;
  else if (l.includes("messy")) value = a.messy_tables;
  else if (l.includes("clean")) value = a.clean_tables;
  else if (l.includes("cup")) value = a.cup_count;
  else value = null;
  return value == null ? "—" : String(value);
}

/**
 * Dashboard tile video. Polls /snapshot (one short request at a time) instead of holding an endless
 * MJPEG connection open: polling short-lived snapshots prevents eight permanent video requests
 * from exhausting the browser's per-host HTTP/1.1 connection pool. Expanded view still uses /stream.
 */
function SnapshotImg({ camId, sessionId, alt, intervalMs = 333 }: { camId: number; sessionId: number | null; alt: string; intervalMs?: number }) {
  const [src, setSrc] = useState<string>("");
  const [failing, setFailing] = useState(false);
  const alive = useRef(true);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => {
    alive.current = true;
    setFailing(false);
    setSrc(api.snapshotUrl(camId, sessionId));
    return () => {
      alive.current = false;
      window.clearTimeout(timer.current);
    };
  }, [camId, sessionId]);

  // Use the displayed img itself as the poller. The old implementation loaded every
  // snapshot once in a probe Image and a second time in React's img, doubling requests.
  const onLoad = () => {
    if (!alive.current) return;
    setFailing(false);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      if (alive.current) setSrc(api.snapshotUrl(camId, sessionId));
    }, intervalMs);
  };
  const onError = () => {
    if (!alive.current) return;
    setFailing(true);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      if (alive.current) setSrc(api.snapshotUrl(camId, sessionId));
    }, 1000);
  };
  return (
    <>
      {src && <img src={src} alt={alt} onLoad={onLoad} onError={onError} />}
      {failing && <div className="placeholder error"><strong>No frames</strong><span>Waiting for the backend to deliver a frame…</span></div>}
    </>
  );
}

export function CameraPanel({ info, status, running, sessionId }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [showEvidence, setShowEvidence] = useState(false);
  const [cleaningMessage, setCleaningMessage] = useState<string>("");
  useEffect(() => setShowEvidence(false), [sessionId, info.id]);
  useEffect(() => {
    if (!status.evidence_available) setShowEvidence(false);
  }, [status.evidence_available]);
  const [cleaningBusy, setCleaningBusy] = useState<string | null>(null);
  const markCleaned = async (table: { id: string; name: string }) => {
    setCleaningBusy(table.id);
    setCleaningMessage("");
    try {
      await api.markTableCleaned(table.id);
      setCleaningMessage(`${table.name}: cleaning confirmed`);
    } catch (e) {
      setCleaningMessage(e instanceof Error ? e.message : "Could not record cleaning event");
    } finally {
      setCleaningBusy(null);
    }
  };
  const [insideInput, setInsideInput] = useState<string>("");
  const [insideMessage, setInsideMessage] = useState<string>("");
  const saveInside = async () => {
    const value = Number(insideInput);
    if (!Number.isInteger(value) || value < 0) {
      setInsideMessage("Enter a whole number, 0 or more");
      return;
    }
    try {
      await api.setPeopleInside(value);
      setInsideMessage(`Inside-now set to ${value}`);
    } catch (e) {
      setInsideMessage(e instanceof Error ? e.message : "Could not save");
    }
  };
  useEffect(() => {
    if (!expanded) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (document.fullscreenElement) void document.exitFullscreen().catch(() => undefined);
        setExpanded(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  const enterFullscreen = () => {
    const target = document.getElementById(`camera-viewer-${info.id}`);
    if (target?.requestFullscreen) void target.requestFullscreen().catch(() => undefined);
  };

  const notSelected = running && status.selected === false;
  const evidenceAgeSeconds = status.evidence_updated_at
    ? Math.max(0, (Date.now() - Date.parse(status.evidence_updated_at)) / 1000)
    : null;
  const sourceAgeSeconds = status.result_frame_age_ms != null ? status.result_frame_age_ms / 1000 : null;
  const resultStale = status.analytics.status === "stale" ||
    (status.evidence_available === true && ((evidenceAgeSeconds !== null && evidenceAgeSeconds > 10) ||
      (sourceAgeSeconds !== null && sourceAgeSeconds > 5)));
  const staleSeconds = sourceAgeSeconds !== null && sourceAgeSeconds > 5 ? sourceAgeSeconds : evidenceAgeSeconds;
  const msg = overlayMessage(status, running);
  const showVideo = running && !msg && status.frames_played > 0;
  const live = status.mode === "live";
  const badge = live ? status.label : "DEMO";

  return (
    <section className={`panel ${msg?.tone === "error" ? "has-error" : ""}`} aria-label={`Camera ${info.id} ${info.name}`}>
      <div className="panel-head">
        <h2>
          CAM {String(info.id).padStart(2, "0")}: {info.name.toUpperCase()}
          {info.subtitle && <span className="sub"> ({info.subtitle})</span>}
        </h2>
        <div className="head-right">
          <span className={`badge ${badge.toLowerCase()}`}>
            {badge === "LIVE" && <i className="dot" />}
            {badge === "DEMO" ? "DEMO · RECORDED" : badge}
          </span>
          {!live && <span className="timecode">{clock(status.position_sec)}</span>}
        </div>
      </div>

      <div className="video">
        {showVideo ? (
          showEvidence && status.evidence_available ? (
            <img
              className="evidence-image"
              src={api.evidenceUrl(info.id, sessionId, status.evidence_updated_at ?? null)}
              alt={`${info.name} last AI inference evidence frame; not live video`}
            />
          ) : (
            <SnapshotImg
              key={`${sessionId}-${info.id}`}
              camId={info.id}
              sessionId={sessionId}
              alt={`${info.name} ${live ? "live" : "recorded demo"} feed`}
            />
          )
        ) : (
          <div className={`placeholder ${msg?.tone ?? "info"}`}>
            <strong>{msg?.tone === "error" ? (live ? "OFFLINE" : "Video problem") : "Standby"}</strong>
            <span>{msg?.text}</span>
          </div>
        )}
        {notSelected && showVideo && <span className="analysis-paused-label">VIDEO ONLY · AI PAUSED</span>}
        {showVideo && !notSelected && resultStale && (
          <span className="ai-stale-label">AI RESULT STALE · {Math.round(staleSeconds ?? 0)}s</span>
        )}
        {showVideo && (
          <button
            type="button"
            className="evidence-toggle"
            disabled={!status.evidence_available}
            onClick={() => setShowEvidence((v) => !v)}
            title={status.evidence_available ? "Show the exact frame used for the last successful AI inference" : "Waiting for the first successful AI inference"}
          >{showEvidence ? "LIVE VIDEO" : status.evidence_available ? "LAST AI EVIDENCE" : "AI EVIDENCE PENDING"}</button>
        )}
        {showVideo && showEvidence && status.evidence_available && (
          <span className="evidence-caption">
            Exact inference frame · {status.evidence_updated_at ? new Date(status.evidence_updated_at).toLocaleTimeString() : "timestamp unavailable"}
            {status.evidence_position_sec != null ? ` · video ${clock(status.evidence_position_sec)}` : ""}
            {status.result_frame_age_ms != null ? ` · result age ${Math.round(status.result_frame_age_ms)} ms` : ""}
          </span>
        )}

        <button
          className="expand-video"
          type="button"
          disabled={!showVideo}
          onClick={() => setExpanded(true)}
          aria-label={`Expand Camera ${info.id} video`}
          title="Expand video"
        >⛶</button>

        <div className="chips" title={info.purpose}>
          {info.analytics.length === 0 ? (
            <span className="chip muted">Analytics not configured</span>
          ) : (
            info.analytics.map((a) => (
              <span className="chip" key={a} title={status.analytics.note ?? "Analytics status unavailable"}>
                {a.toUpperCase()}: <b>{metricFor(a, status.analytics)}</b>
              </span>
            ))
          )}
        </div>
        <span className="stamp">{stamp(status.last_frame_at)}</span>
      </div>

      {(info.role === "people_count" || status.analytics.status === "stale") && (
        <div className={`ai-quality-note ${status.analytics.status === "stale" ? "stale" : "current"}`}>
          {status.analytics.status === "stale" ? (
            <>
              <strong>HISTORICAL EVIDENCE ONLY</strong>
              <span>{status.analytics.people_count == null ? "No people count available" : `${status.analytics.people_count} people detected in the saved source frame`}</span>
              <span>{status.result_frame_age_ms != null ? `Source-frame age at inference completion: ${(status.result_frame_age_ms / 1000).toFixed(1)} s` : "Source-frame age unavailable"}. Not a current live count; delayed results do not update tracking or entrance counters.</span>
            </>
          ) : (
            <>
              <strong>MAIN HALL COUNT</strong>
              <span>Current sampled frame: {status.analytics.people_count ?? "waiting"} people · Active short-term tracks: {status.analytics.tracked_people ?? "—"}</span>
              <span>{status.analytics.note ?? "Compare the Last AI Evidence image with a manual count."}</span>
            </>
          )}
        </div>
      )}
      <dl className="panel-foot">
        <div>
          <dt>Playback</dt>
          <dd className={`st-${status.playback_state}`}>
            {status.playback_state}
            {status.fps ? ` · ${status.fps} fps` : ""}
          </dd>
        </div>
        <div>
          <dt>Processing</dt>
          <dd className={`st-${status.processing_state}`}>
            {status.processing_state.replace("_", " ")} · {status.frames_processed} inferences
            {status.last_inference_ms != null ? ` · ${Math.round(status.last_inference_ms)} ms` : ""}
            {status.result_frame_age_ms != null ? ` · source age ${Math.round(status.result_frame_age_ms)} ms` : ""}
          </dd>
        </div>
        <div>
          <dt>Last update</dt>
          <dd>{status.last_frame_at ? new Date(status.last_frame_at).toLocaleTimeString() : "—"}</dd>
        </div>
      </dl>
      {info.role === "entry_exit" && running && (
        <div className="table-actions">
          <span className="table-action-title">Set verified "inside now" count (opening/current occupancy)</span>
          <input className="count-input" inputMode="numeric" value={insideInput} placeholder="e.g. 16"
            onChange={(e) => setInsideInput(e.target.value)} aria-label="Real number of people inside" />
          <button className="btn table-action" onClick={() => void saveInside()}>Save</button>
          <span className="table-action-help">Use a staff-verified baseline. The inside total is baseline + entries − exits; do not add Main Hall detections separately.</span>
          {insideMessage && <span className="table-action-message">{insideMessage}</span>}
        </div>
      )}
      {info.role === "table_status" && (info.table_regions?.length ?? 0) > 0 && (
        <div className="table-actions">
          <span className="table-action-title">Staff cleaning confirmation</span>
          {info.table_regions!.map((table) => (
            <button className="btn table-action" key={table.id} disabled={cleaningBusy !== null}
              onClick={() => void markCleaned(table)} title="Records a staff-confirmed cleaning event">
              {cleaningBusy === table.id ? "Saving…" : `Mark ${table.name} cleaned`}
            </button>
          ))}
          {cleaningMessage && <span className="table-action-message">{cleaningMessage}</span>}
        </div>
      )}
      {expanded && createPortal(
        <div className="video-viewer-backdrop" role="dialog" aria-modal="true" aria-label={`Expanded camera ${info.id} view`} onMouseDown={() => setExpanded(false)}>
          <section className="video-viewer" id={`camera-viewer-${info.id}`} onMouseDown={(event) => event.stopPropagation()}>
            <header className="video-viewer-head">
              <div>
                <strong>CAM {String(info.id).padStart(2, "0")}: {info.name}</strong>
                <span className={`badge ${badge.toLowerCase()}`}>{badge === "DEMO" ? "DEMO · RECORDED" : badge}</span>
              </div>
              <div className="video-viewer-actions">
                <button type="button" className="btn" onClick={enterFullscreen} title="Browser full screen">⛶ Full screen</button>
                <button type="button" className="btn danger" onClick={() => setExpanded(false)} aria-label="Close expanded video">✕ Close</button>
              </div>
            </header>
            <div className="video-viewer-stage">
              {showVideo ? (
                <img src={api.streamUrl(info.id, sessionId)} alt={`Expanded ${info.name} ${live ? "live" : "recorded demo"} feed`} />
              ) : (
                <div className="viewer-placeholder"><strong>{live ? "Live feed unavailable" : "Demo feed unavailable"}</strong><span>{msg?.text ?? "Start monitoring to view this camera."}</span></div>
              )}
            </div>
            <footer className="video-viewer-foot">{live ? "Live source configured by authorized IT" : "Recorded client footage · Demo mode"} · Press Esc to close</footer>
          </section>
        </div>,
        document.body
      )}
    </section>
  );
}
