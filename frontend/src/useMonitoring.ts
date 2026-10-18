import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { CameraInfo, Mode, MonitoringStatus, Summary, SystemResources } from "./types";

export function useMonitoring() {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [status, setStatus] = useState<MonitoringStatus | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [resources, setResources] = useState<SystemResources | null>(null);
  const [wsUp, setWsUp] = useState(false);
  const [busy, setBusy] = useState(false);
  const [openingOccupancy, setOpeningOccupancy] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  const closed = useRef(false);
  // Start with Main Hall (camera 03) so the default demo visibly tests seated-person detection.
  // Only one camera is analyzed at a time on the client's dual-core CPU; all other tiles remain video-only.
  const [selected, setSelectedState] = useState<number[]>(() => {
    try {
      const raw = localStorage.getItem("cafe.selectedCameras");
      const arr = raw ? (JSON.parse(raw) as number[]) : [];
      const valid = Array.isArray(arr) ? arr.filter((n) => Number.isInteger(n) && n >= 1 && n <= 8) : [];
      // Migrate empty/all-camera and known previous app defaults to Main Hall-only.
      // Camera 07 can be selected explicitly for entrance / occupancy tracking.
      const oldDefault = (valid.length === 1 && valid[0] === 7) ||
        (valid.length === 2 && valid[0] === 2 && valid[1] === 7);
      return valid.length === 0 || valid.length === 8 || oldDefault ? [3] : valid;
    } catch {
      return [3];
    }
  });
  const setSelected = useCallback((ids: number[]) => {
    setSelectedState(ids);
    try {
      localStorage.setItem("cafe.selectedCameras", JSON.stringify(ids));
    } catch {
      /* storage unavailable: selection just isn't remembered */
    }
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [s, sum, sys] = await Promise.all([api.status(), api.summary(), api.resources()]);
      setStatus(s);
      setSummary(sum);
      setResources(sys);
      setBackendDown(false);
    } catch {
      setBackendDown(true);
    }
  }, []);

  useEffect(() => {
    api.cameras().then(setCameras).catch(() => setBackendDown(true));
    void refresh();
  }, [refresh]);

  // WebSocket for real-time updates, auto-reconnect.
  useEffect(() => {
    closed.current = false;
    let ws: WebSocket | null = null;
    let timer: number | undefined;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onopen = () => setWsUp(true);
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "update") {
          setStatus(msg.status);
          setSummary(msg.summary);
          if (msg.resources) setResources(msg.resources);
          setBackendDown(false);
        }
      };
      ws.onclose = () => {
        setWsUp(false);
        if (!closed.current) timer = window.setTimeout(connect, 2000);
      };
    };
    connect();
    return () => {
      closed.current = true;
      window.clearTimeout(timer);
      ws?.close();
    };
  }, []);

  // Polling fallback only while the socket is down.
  useEffect(() => {
    if (wsUp) return;
    const t = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(t);
  }, [wsUp, refresh]);

  const act = useCallback(async (fn: () => Promise<MonitoringStatus>) => {
    setBusy(true);
    setError(null);
    try {
      setStatus(await fn());
      setSummary(await api.summary());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  return {
    cameras, status, summary, resources, wsUp, busy, error, backendDown,
    selected, setSelected, openingOccupancy, setOpeningOccupancy,
    start: () => {
      const all = cameras.map((c) => c.id);
      const chosen = selected.filter((id) => all.includes(id));
      let baseline: number | undefined;
      if (chosen.includes(7) && openingOccupancy.trim() !== "") {
        baseline = Number(openingOccupancy);
        if (!Number.isInteger(baseline) || baseline < 0 || baseline > 1000) {
          setError("Opening occupancy must be a whole number from 0 to 1000.");
          return Promise.resolve();
        }
      }
      // All cameras are analyzed only when the user explicitly chooses All in the selector.
      return act(() => api.start(chosen.length === 0 || chosen.length === all.length ? undefined : chosen, baseline));
    },
    stop: () => act(api.stop),
    restart: () => act(api.restart),
    setMode: (m: Mode) => act(() => api.setMode(m)),
    dismissError: () => setError(null),
  };
}
