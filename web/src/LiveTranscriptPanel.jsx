import React, { useCallback, useEffect, useRef, useState } from "react";

import { api } from "./api.js";

// The backend sends `time` ("HH:MM:SS") for every line, including lines read
// back from live-transcript.txt, whose started_at is 0; this is the fallback.
export function formatLiveTime(startedAt) {
  if (!startedAt) return "";
  return new Date(startedAt * 1000).toLocaleTimeString([], {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

// Same shape the backend appends to live-transcript.txt: "[HH:MM:SS] source: text".
export function formatLiveLine(line) {
  const time = line?.time || formatLiveTime(line?.started_at);
  const body = `${line?.source ?? ""}: ${line?.text ?? ""}`;
  return time ? `[${time}] ${body}` : body;
}

function statusLabel(status) {
  if (!status) return "…";
  const parts = [
    status.status === "loading" ? "loading model…" : status.status,
    status.model_size,
    `${status.pending ?? 0} pending`,
    `${status.clips_transcribed ?? 0} transcribed`,
    `${Number(status.average_latency || 0).toFixed(1)}s avg`
  ];
  return parts.filter(Boolean).join(" · ");
}

// `live` is only the seed taken from the meeting payload; once mounted the
// panel polls the live endpoint itself and that answer is the truth.
export default function LiveTranscriptPanel({ project, meeting, live, onChanged }) {
  const [status, setStatus] = useState(live || null);
  const [lines, setLines] = useState([]);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [model, setModel] = useState(live?.model_size || live?.default_model || "");
  const after = useRef(0);
  const logRef = useRef(null);
  const followTail = useRef(true);
  const wasActive = useRef(false);

  const poll = useCallback(async () => {
    try {
      const data = await api.live(project, meeting, after.current);
      setStatus(data);
      setError(null);
      // A fresh session numbers its lines from zero again, so a total below
      // our cursor means the stream restarted: rewind and read it from 0.
      if (typeof data.lines_total === "number" && data.lines_total < after.current) {
        after.current = 0;
        setLines([]);
        return data;
      }
      if (data.lines?.length) {
        after.current = data.lines[data.lines.length - 1].index + 1;
        setLines((current) => [...current, ...data.lines]);
      }
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }, [project, meeting]);

  useEffect(() => {
    after.current = 0;
    setLines([]);
    poll();
  }, [poll]);

  const active = Boolean(status?.active);

  useEffect(() => {
    if (!active) return undefined;
    const timer = setInterval(() => poll(), 1200);
    return () => clearInterval(timer);
  }, [active, poll]);

  // The last clip finishes after the session reports stopped, so read once more.
  useEffect(() => {
    if (wasActive.current && !active) poll();
    wasActive.current = active;
  }, [active, poll]);

  useEffect(() => {
    if (!status) return;
    setModel((current) => current || status.model_size || status.default_model || "");
  }, [status]);

  // Stay pinned to the newest line unless the reader has scrolled back up.
  useEffect(() => {
    const node = logRef.current;
    if (node && followTail.current) node.scrollTop = node.scrollHeight;
  }, [lines]);

  const onScroll = () => {
    const node = logRef.current;
    if (!node) return;
    followTail.current = node.scrollHeight - node.scrollTop - node.clientHeight < 24;
  };

  const runAction = async (action) => {
    setError(null);
    setBusy(true);
    try {
      setStatus(await action());
      await poll();
      await onChanged?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const start = (mode) => {
    after.current = 0;
    setLines([]);
    followTail.current = true;
    return runAction(() => api.liveStart(project, meeting, { model_size: model, mode }));
  };

  const stop = () => runAction(() => api.liveStop(project, meeting));

  const options = status?.model_options?.length ? status.model_options : [model].filter(Boolean);
  const resumable = Boolean(status?.watermark);

  return (
    <>
      <p className="hint" style={{ marginTop: 0 }}>
        Transcribes new clips as they are recorded, from the moment you start. Independent of recording and of the offline
        transcript; nothing here changes the normal pipeline.
      </p>

      {error ? <div className="error" style={{ marginTop: 10 }}>{error}</div> : null}

      <div className="row" style={{ marginTop: 10 }}>
        <label className="hint">
          Model{" "}
          <select value={model} disabled={active || busy} onChange={(event) => setModel(event.target.value)}>
            {options.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>

        {active ? (
          <button className="danger" disabled={busy} onClick={stop}>
            Stop
          </button>
        ) : resumable ? (
          <>
            <button className="primary" disabled={busy} onClick={() => start("now")}>
              Start from now
            </button>
            <button disabled={busy} onClick={() => start("catch_up")}>
              Resume (catch up)
            </button>
          </>
        ) : (
          <button className="primary" disabled={busy} onClick={() => start("now")}>
            Start live
          </button>
        )}

        <div className="spacer" />
        <span
          className={`pill${status?.status === "running" ? " live" : status?.status === "loading" ? " warn" : ""}`}
        >
          {statusLabel(status)}
        </span>
      </div>

      {status?.status === "error" && status.error ? (
        <div className="error" style={{ marginTop: 10 }}>
          {status.error}
        </div>
      ) : null}

      <div className="log" style={{ maxHeight: "40vh" }} ref={logRef} onScroll={onScroll}>
        {lines.length
          ? lines.map((line) => <div key={line.index}>{formatLiveLine(line)}</div>)
          : active
          ? "Listening for new clips…"
          : "Nothing transcribed live yet."}
      </div>
    </>
  );
}
