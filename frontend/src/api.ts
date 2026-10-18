import type { CameraInfo, Mode, MonitoringStatus, Summary, SystemResources } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

const post = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  cameras: () => request<CameraInfo[]>("/api/cameras"),
  status: () => request<MonitoringStatus>("/api/monitoring/status"),
  summary: () => request<Summary>("/api/analytics/summary"),
  resources: () => request<SystemResources>("/api/system/resources"),
  markTableCleaned: (id: string) => post<{ ok: boolean; status: string; table_id: string }>(`/api/tables/${encodeURIComponent(id)}/cleaned`),
  setPeopleInside: (people_inside: number) => post<{ ok: boolean; people_inside: number }>("/api/occupancy/set", { people_inside }),
  start: (cameras?: number[], openingOccupancy?: number) => {
    const body: { cameras?: number[]; opening_occupancy?: number } = {};
    if (cameras) body.cameras = cameras;
    if (openingOccupancy !== undefined) body.opening_occupancy = openingOccupancy;
    return post<MonitoringStatus>("/api/monitoring/start", Object.keys(body).length ? body : undefined);
  },
  stop: () => post<MonitoringStatus>("/api/monitoring/stop"),
  restart: () => post<MonitoringStatus>("/api/monitoring/restart"),
  setMode: (mode: Mode) => post<MonitoringStatus>("/api/monitoring/mode", { mode }),
  streamUrl: (id: number, session: number | null) => `/api/cameras/${id}/stream?s=${session ?? 0}`,
  snapshotUrl: (id: number, session: number | null) => `/api/cameras/${id}/snapshot?s=${session ?? 0}&t=${Date.now()}`,
  evidenceUrl: (id: number, session: number | null, evidenceKey: string | null) => `/api/cameras/${id}/evidence?s=${session ?? 0}&e=${encodeURIComponent(evidenceKey ?? "last")}`,
};
