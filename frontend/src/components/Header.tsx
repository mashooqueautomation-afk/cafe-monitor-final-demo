import type { Mode, MonitoringStatus } from "../types";

interface Props {
  status: MonitoringStatus | null;
  busy: boolean;
  wsUp: boolean;
  backendDown: boolean;
  onStart: () => void;
  onStop: () => void;
  onRestart: () => void;
  onMode: (m: Mode) => void;
}

export function Header({ status, busy, wsUp, backendDown, onStart, onStop, onRestart, onMode }: Props) {
  const running = status?.running ?? false;
  const mode: Mode = status?.mode ?? "live";
  const link = backendDown ? "Backend offline" : wsUp ? "Real-time link up" : "Polling (socket down)";

  return (
    <header className="topbar">
      <div className="brand">
        <h1>AUTONIQ Café Monitor</h1>
        <span className={`session ${running ? "on" : ""}`}>
          {running ? `Session #${status?.session_id} running` : "Monitoring stopped"}
        </span>
      </div>

      <div className="modes" role="group" aria-label="Operating mode">
        {(["demo", "live"] as Mode[]).map((m) => (
          <button
            key={m}
            className={`mode ${mode === m ? "active" : ""} ${m}`}
            disabled={busy || (running && mode !== m)}
            title={running && mode !== m ? "Stop monitoring before switching mode" : undefined}
            aria-pressed={mode === m}
            onClick={() => mode !== m && onMode(m)}
          >
            {m === "demo" ? "Demo mode" : "Live mode"}
          </button>
        ))}
      </div>

      <div className="controls">
        <button className="btn primary" disabled={busy || running} onClick={onStart}>Start monitoring</button>
        <button className="btn" disabled={busy || !running} onClick={onRestart}>Restart</button>
        <button className="btn danger" disabled={busy || !running} onClick={onStop}>Stop</button>
      </div>

      <span className={`link ${backendDown ? "down" : wsUp ? "up" : "poll"}`}>{link}</span>
    </header>
  );
}
