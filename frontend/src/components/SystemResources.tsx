import type { SystemResources as Metrics } from "../types";

const mb = (bytes: number | null | undefined) =>
  bytes == null ? "—" : `${(bytes / (1024 * 1024)).toFixed(bytes >= 1024 * 1024 * 1024 ? 1 : 0)} MB`;

function Meter({ label, value, detail, tone = "cyan" }: { label: string; value: number | null; detail: string; tone?: string }) {
  const width = Math.max(0, Math.min(100, value ?? 0));
  return (
    <div className="resource-meter">
      <div className="resource-label"><span>{label}</span><b>{value == null ? "—" : `${value.toFixed(1)}%`}</b></div>
      <div className="resource-track"><span className={`resource-fill ${tone}`} style={{ width: `${width}%` }} /></div>
      <div className="resource-detail">{detail}</div>
    </div>
  );
}

export function SystemResources({ metrics }: { metrics: Metrics | null }) {
  const containerPercent = metrics?.container_ram_limit_bytes
    ? (metrics.container_ram_used_bytes / metrics.container_ram_limit_bytes) * 100 : null;
  return (
    <section className="system-resources" aria-label="Backend system resources">
      <div className="resource-title"><span className="resource-pulse" /> SYSTEM RESOURCES <small>BACKEND / CONTAINER</small></div>
      {!metrics || metrics.error ? (
        <div className="resource-detail">Resource metrics unavailable</div>
      ) : (
        <div className="resource-grid">
          <Meter label="Process CPU" value={metrics.process_cpu_percent} detail="Python backend process" />
          <Meter label="Process RAM" value={(metrics.container_ram_limit_bytes ?? metrics.host_ram_total_bytes) ? metrics.process_ram_bytes / (metrics.container_ram_limit_bytes ?? metrics.host_ram_total_bytes) * 100 : null}
            detail={`${mb(metrics.process_ram_bytes)} resident memory`} tone="blue" />
          <Meter label="Container RAM" value={containerPercent}
            detail={`${mb(metrics.container_ram_used_bytes)}${metrics.container_ram_limit_bytes ? ` / ${mb(metrics.container_ram_limit_bytes)}` : " used · limit unavailable"}`}
            tone={containerPercent != null && containerPercent > 85 ? "red" : "green"} />
        </div>
      )}
      <div className="resource-foot">{metrics?.note ?? "Live process metrics load from the backend"}</div>
    </section>
  );
}
