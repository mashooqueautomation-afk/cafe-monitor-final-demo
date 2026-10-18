import type { MonitoringStatus, Summary } from "../types";

const n = (v: number | null | undefined) => (v === null || v === undefined ? "—" : v.toLocaleString());

export function Overview({ summary, status }: { summary: Summary | null; status: MonitoringStatus | null }) {
  const d = summary?.detection;
  return (
    <section className="overview" aria-label="Analytics overview">
      <h2>Overview</h2>
      <div className="cards">
        <div className="card">
          <h3>Monitoring</h3>
          <p className="big">{summary?.monitoring_running ? "Running" : "Stopped"}</p>
          <p className="muted">{(status?.mode ?? "live") === "demo" ? "Demo mode · recorded footage" : "Live mode · RTSP cameras"}</p>
        </div>
        <div className="card">
          <h3>Cameras</h3>
          <p className="big">
            {n(summary?.cameras_playing)} <small>/ {n(summary?.cameras_total)} playing</small>
          </p>
          <p className="muted">{n(summary?.cameras_with_problems)} with errors or offline</p>
        </div>
        <div className="card">
          <h3>Frames this session</h3>
          <p className="big">{n(summary?.frames_processed_session)}</p>
          <p className="muted">{n(summary?.frames_played_session)} played · {n(summary?.frames_processed_lifetime)} processed all-time</p>
        </div>
        <div className="card">
          <h3>Stored sessions</h3>
          <p className="big">{n(summary?.sessions_total)}</p>
          <p className="muted">saved in the database</p>
        </div>
        <div className="card">
          <h3>Cashier / POS integration</h3>
          <p className="big">{summary && summary?.integrations?.cashier_pos?.available ? "Connected" : "Not configured"}</p>
          <p className="muted">Requires a dedicated cashier feed and/or authorized POS data. No cash values are fabricated.</p>
        </div>
        <div className="card wide">
          <h3>Detection analytics</h3>
          <p className="muted">{status?.running && d?.available ? "Active" : d?.available ? "Inference worked earlier in this mode today" : d?.note ?? "Waiting for backend"}</p>
          <ul className="metrics">
            <li>People <b>{n(d?.people_count)}</b></li>
            <li>In <b>{n(d?.entries)}</b></li>
            <li>Out <b>{n(d?.exits)}</b></li>
            <li>Inside now <b>{n(d?.people_inside)}</b></li>
            <li>Occupied tables <b>{n(d?.occupied_tables)}</b></li>
            <li>Empty tables <b>{n(d?.empty_tables)}</b></li>
            <li>Tables to check <b>{n(d?.messy_tables)}</b></li>
            <li>Cups <b>{n(d?.cup_count)}</b></li>
            <li>Hall-zone avg. dwell <b>{d?.zone_dwell_avg_seconds == null ? "—" : `${d.zone_dwell_avg_seconds.toFixed(1)}s`}</b></li>
            <li>Tracked in hall <b>{n(d?.tracked_people)}</b></li>
          </ul>
        </div>
      </div>
    </section>
  );
}
