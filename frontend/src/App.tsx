import { CameraPanel } from "./components/CameraPanel";
import { CameraSelector } from "./components/CameraSelector";
import { Header } from "./components/Header";
import { Overview } from "./components/Overview";
import { SystemResources } from "./components/SystemResources";
import { useMonitoring } from "./useMonitoring";

export default function App() {
  const m = useMonitoring();
  const running = m.status?.running ?? false;

  return (
    <div className="app">
      <Header
        status={m.status}
        busy={m.busy}
        wsUp={m.wsUp}
        backendDown={m.backendDown}
        onStart={m.start}
        onStop={m.stop}
        onRestart={m.restart}
        onMode={m.setMode}
      />
      {m.error && (
        <div className="banner error" role="alert">
          <span>{m.error}</span>
          <button onClick={m.dismissError}>Dismiss</button>
        </div>
      )}
      {m.backendDown && !m.status && (
        <div className="banner error" role="alert">
          Cannot reach the backend. Start it with <code>uvicorn app.main:app --port 8000</code> from the backend folder.
        </div>
      )}
      <CameraSelector
        cameras={m.cameras}
        selected={m.selected}
        onChange={m.setSelected}
        running={running}
        busy={m.busy}
        runningIds={m.status?.selected_camera_ids ?? []}
        focus={m.status?.focus_mode ?? false}
        openingOccupancy={m.openingOccupancy}
        onOpeningOccupancyChange={m.setOpeningOccupancy}
      />
      <SystemResources metrics={m.resources} />
      <main className="grid" aria-label="Camera feeds">
        {m.cameras.map((info) => {
          const st = m.status?.cameras.find((c) => c.id === info.id);
          return st ? (
            <CameraPanel key={info.id} info={info} status={st} running={running} sessionId={m.status?.session_id ?? null} />
          ) : null;
        })}
      </main>
      <Overview summary={m.summary} status={m.status} />
    </div>
  );
}
