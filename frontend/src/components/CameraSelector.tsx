import type { CameraInfo } from "../types";

interface Props {
  cameras: CameraInfo[];
  selected: number[]; // remembered choice for the NEXT start ([] = all)
  onChange: (ids: number[]) => void;
  running: boolean;
  busy: boolean;
  runningIds: number[]; // what the current run is actually analyzing
  focus: boolean;
  openingOccupancy: string;
  onOpeningOccupancyChange: (value: string) => void;
}

/** Choose which cameras run AI analytics. All configured video tiles continue to play; fewer selected cameras get a higher AI rate. */
export function CameraSelector({ cameras, selected, onChange, running, busy, runningIds, focus, openingOccupancy, onOpeningOccupancyChange }: Props) {
  if (cameras.length === 0) return null;
  const all = cameras.map((c) => c.id);
  const effective = selected.length === 0 ? all : selected.filter((id) => all.includes(id));
  const locked = running || busy;
  const toggle = (id: number) => {
    const next = effective.includes(id) ? effective.filter((x) => x !== id) : [...effective, id];
    onChange(next.length === 0 ? [id] : next.sort((a, b) => a - b)); // never allow an empty selection
  };
  const active = running ? runningIds : effective;
  return (
    <section className="cam-select" aria-label="Cameras to analyze">
      <span className="cam-select-title">Analyze cameras</span>
      <div className="cam-chips" role="group">
        {cameras.map((c) => {
          const on = active.includes(c.id);
          return (
            <button
              key={c.id}
              className={`chip ${on ? "on" : ""}`}
              disabled={locked}
              aria-pressed={on}
              title={`${c.name}${locked ? " (stop monitoring to change)" : ""}`}
              onClick={() => toggle(c.id)}
            >
              {String(c.id).padStart(2, "0")}
            </button>
          );
        })}
      </div>
      <button className="btn small" disabled={locked} onClick={() => onChange(all)}>All</button>
      <div className="cam-presets" aria-label="Recommended analysis profiles">
        <button className="btn small" disabled={locked} onClick={() => onChange([3])}>Main Hall count · 03</button>
        <button className="btn small" disabled={locked} onClick={() => onChange([7])}>Entrance occupancy · 07</button>
      </div>
      {effective.includes(7) && !running && (
        <div className="occupancy-baseline-control">
          <label htmlFor="opening-occupancy">Verified opening occupancy (Camera 07)</label>
          <input id="opening-occupancy" className="count-input" type="number" min="0" max="1000" step="1"
            value={openingOccupancy} placeholder="Enter staff count, e.g. 16"
            onChange={(e) => onOpeningOccupancyChange(e.target.value)} disabled={busy} />
          <small>Set this before Start so crossings in the first frames are included. Leave blank if unknown.</small>
        </div>
      )}
      <span className="cam-select-hint">
        {running
          ? `Running: ${runningIds.length === all.length ? "all cameras" : "camera " + runningIds.join(", ")}${focus ? " · focus mode (higher AI rate)" : ""}`
          : effective.length === all.length
            ? `All ${all.length} cameras together can overload a small CPU. Pick one (or two) to analyze at a time.`
            : `Next AI analysis: camera ${effective.join(", ")}${effective.length <= 2 ? " · focus mode (other feeds stay video-only)" : ""}. Camera 03 = visible people check; Camera 07 = entrance/inside counter.`}
      </span>
    </section>
  );
}
