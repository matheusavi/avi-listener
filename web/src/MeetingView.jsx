import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { prepareChromeTabCapture } from "./chromeTabCapture.js";
import LiveTranscriptPanel from "./LiveTranscriptPanel.jsx";
import Speakers from "./Speakers.jsx";
import { api } from "./api.js";

const SOURCES = [
  { id: "mic", label: "My microphone", note: "your voice" },
  { id: "system", label: "PC audio", note: "everyone else, via speaker loopback" },
  { id: "chrome", label: "Chrome tab", note: "audio from one tab only" },
  // Discord names the speaker itself, so its files are already per-person.
  { id: "discord", label: "Discord", note: "one file per person, no splitting needed", needsToken: true }
];

function Step({ index, title, done, enabled, blockedReason, children }) {
  return (
    <div className={`card${enabled ? "" : " disabled"}`}>
      <div className="card-head">
        <span className={`step-number${done ? " done" : ""}`}>{done ? "✓" : index}</span>
        <h3>{title}</h3>
        {!enabled && blockedReason ? <span className="pill">{blockedReason}</span> : null}
      </div>
      {children}
    </div>
  );
}

const FILE_STATE_LABELS = {
  waiting: "waiting for audio",
  growing: "growing",
  stalled: "not growing",
  failed: "failed",
  complete: "complete",
  empty: "empty"
};

function formatBytes(bytes = 0) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function Meter({ name, peak, file }) {
  // 0.05 is roughly a quiet-but-usable level, so it makes a sensible full bar.
  const width = Math.min(100, (peak / 0.05) * 100);
  const silent = peak < 0.01;
  return (
    <div className="meter">
      <div className="label">
        <span>{name}</span>
        <span>{silent ? "silent" : peak.toFixed(3)}</span>
      </div>
      <div className="bar">
        <div className={`fill${silent ? " silent" : ""}`} style={{ width: `${width}%` }} />
      </div>
      {file ? (
        <div className={`file-growth ${file.state}`} data-state={file.state} title={file.filename || undefined}>
          {formatBytes(file.bytes)} · {FILE_STATE_LABELS[file.state] || file.state}
        </div>
      ) : null}
    </div>
  );
}

export default function MeetingView({ projectSlug, meetingSlug, onChanged }) {
  const [meeting, setMeeting] = useState(null);
  const [error, setError] = useState(null);
  const [sources, setSources] = useState(["mic", "system"]);
  const [numSpeakers, setNumSpeakers] = useState("");
  const [me, setMe] = useState("host");
  const [tab, setTab] = useState("merged");
  const [transcript, setTranscript] = useState(null);
  const [logs, setLogs] = useState(null);
  const [liveText, setLiveText] = useState("");
  const [channels, setChannels] = useState(null);
  const [channelId, setChannelId] = useState("");
  const [busy, setBusy] = useState(false);
  const chromeCapture = useRef(null);

  const load = useCallback(async () => {
    try {
      const data = await api.meeting(projectSlug, meetingSlug);
      setMeeting(data);
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }, [projectSlug, meetingSlug]);

  useEffect(() => {
    setMeeting(null);
    setTranscript(null);
    setError(null);
    load().then((data) => {
      if (!data) return;
      setSources(data.config.sources?.length ? data.config.sources : ["mic", "system"]);
      setMe(data.config.me || "host");
      setNumSpeakers(data.config.num_speakers ? String(data.config.num_speakers) : "");
    });
  }, [load]);

  const recording = meeting?.recording;
  const job = meeting?.active_job;
  // Discord alone is a recording too: the receiver runs as its own process
  // and there is no local audio session to report on, so ask it directly.
  const discordRunning = Boolean(meeting?.discord?.running);
  const isRecording = Boolean(recording?.running) || discordRunning;
  const elapsed = recording?.elapsed ?? meeting?.discord?.elapsed ?? 0;
  const jobRunning = job?.status === "running";
  const liveActive = Boolean(meeting?.live?.active);

  // Only poll while something is actually happening, so an idle dashboard is
  // not hammering the API. A live session counts: it keeps writing lines, so
  // the meeting payload (and its `live` summary) keeps changing.
  useEffect(() => {
    if (!isRecording && !jobRunning && !liveActive) return undefined;
    const timer = setInterval(() => {
      load().then((data) => {
        if (data && !data.recording?.running && !data.discord?.running && !data.active_job && !data.live?.active) {
          onChanged?.();
        }
      });
    }, 1200);
    return () => clearInterval(timer);
  }, [isRecording, jobRunning, liveActive, load, onChanged]);

  useEffect(() => {
    if (!meeting?.discord_available || !sources.includes("discord") || channels !== null) return;
    api
      .discordChannels(projectSlug)
      .then((data) => {
        setChannels(data);
        setChannelId((current) => current || meeting.config.discord_channel_id || "");
      })
      .catch((err) => setError(err.message));
  }, [meeting?.discord_available, sources, channels, projectSlug]);

  // One place decides what a tab loads. Logs are not a transcript kind, and
  // routing this in more than one caller is what asked the transcript endpoint
  // for "logs" and surfaced its error to the user.
  const showTab = useCallback(
    async (which) => {
      try {
        if (which === "logs") {
          setLogs(await api.logs(projectSlug, meetingSlug));
        } else if (which === "live") {
          setLiveText(await api.liveTranscript(projectSlug, meetingSlug));
        } else {
          setTranscript(await api.transcript(projectSlug, meetingSlug, which));
        }
      } catch (err) {
        setError(err.message);
      }
    },
    [projectSlug, meetingSlug]
  );

  useEffect(() => {
    if (!meeting) return;
    showTab(tab);
  }, [tab, showTab, meeting?.artifacts.has_merged, meeting?.artifacts.has_diarized, meeting?.artifacts.has_transcripts,
      meeting?.jobs?.[0]?.finished_at, meeting?.live?.lines_total]);

  const act = async (fn) => {
    setError(null);
    setBusy(true);
    try {
      await fn();
      await load();
      onChanged?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const stopSelectedRecording = async () => {
    const capture = chromeCapture.current;
    chromeCapture.current = null;
    let captureError = null;
    if (capture) {
      try {
        await capture.stop();
      } catch (err) {
        captureError = err;
      }
    }
    await api.stopRecording(projectSlug, meetingSlug);
    if (captureError) throw captureError;
  };

  const startSelectedRecording = async () => {
    let capture = null;
    let backendStarted = false;
    try {
      if (sources.includes("chrome")) {
        capture = await prepareChromeTabCapture({
          onChunk: (audio, sampleRate) => api.pushChromeAudio(projectSlug, meetingSlug, sampleRate, audio),
          onEnded: () => {
            if (chromeCapture.current === capture) act(stopSelectedRecording);
          }
        });
      }
      if (sources.includes("discord") && channelId) {
        await api.meetingConfig(projectSlug, meetingSlug, { discord_channel_id: channelId });
      }
      await api.startRecording(projectSlug, meetingSlug, sources);
      backendStarted = true;
      if (capture) {
        chromeCapture.current = capture;
        await capture.start();
      }
    } catch (err) {
      if (chromeCapture.current === capture) chromeCapture.current = null;
      if (capture) await capture.stop().catch(() => {});
      if (backendStarted) await api.stopRecording(projectSlug, meetingSlug).catch(() => {});
      throw err;
    }
  };

  // A tab capture cannot outlive the meeting view that owns it. Finalise the
  // local recording as well so the WAV header is never left open.
  useEffect(
    () => () => {
      const capture = chromeCapture.current;
      chromeCapture.current = null;
      if (capture) capture.stop().finally(() => api.stopRecording(projectSlug, meetingSlug)).catch(() => {});
    },
    [projectSlug, meetingSlug]
  );

  const availableTabs = useMemo(() => {
    if (!meeting) return [];
    const tabs = [];
    if (meeting.artifacts.has_merged) tabs.push({ id: "merged", label: "Everyone (merged)" });
    if (meeting.artifacts.has_diarized) tabs.push({ id: "diarized", label: "Shared audio by speaker" });
    if (meeting.artifacts.has_transcripts) tabs.push({ id: "segments", label: "By source" });
    if (meeting.live?.lines_total > 0) tabs.push({ id: "live", label: "Live" });
    tabs.push({ id: "logs", label: "Logs" });
    return tabs;
  }, [meeting]);

  useEffect(() => {
    if (availableTabs.length && !availableTabs.some((item) => item.id === tab)) {
      setTab(availableTabs[0].id);
    }
  }, [availableTabs, tab]);

  if (!meeting) {
    return <div className="main">{error ? <div className="error">{error}</div> : <div className="empty">Loading…</div>}</div>;
  }

  const { artifacts, can } = meeting;
  const toggleSource = (id) =>
    setSources((current) => {
      if (current.includes(id)) return current.filter((item) => item !== id);
      // Capturing both would record the same remote voices twice and duplicate
      // them in the merged transcript. Chrome tab is the isolated alternative
      // to whole-PC loopback, not an additional copy of it.
      const incompatible = id === "chrome" ? "system" : id === "system" ? "chrome" : null;
      return [...current.filter((item) => item !== incompatible), id];
    });

  return (
    <div className="main">
      <h2>{meeting.name}</h2>
      <div className="breadcrumb">
        {projectSlug} / {meeting.slug}
        {artifacts.loopback ? " · recorded" : ""}
      </div>

      {error ? <div className="error">{error}</div> : null}

      {artifacts.diarization_stale || artifacts.transcription_stale ? (
        <p className="hint">New recording parts detected. Transcribe and split again before merging. Existing transcripts show the previous recording.</p>
      ) : artifacts.has_merged && artifacts.merged_stale ? (
        <p className="hint">Transcription or speaker results changed. Merge again to update the complete transcript.</p>
      ) : null}

      <Step index={1} title="Record" done={artifacts.has_recording} enabled>
        <div className="row">
          {SOURCES.map((source) => {
            const blocked = source.needsToken && !meeting.discord_available;
            return (
            <label
              key={source.id}
              className={`toggle${sources.includes(source.id) ? " on" : ""}`}
              title={blocked ? "Set DISCORD_BOT_TOKEN, or paste a token in settings" : undefined}
              style={blocked ? { opacity: 0.45 } : undefined}
            >
              <input
                type="checkbox"
                checked={sources.includes(source.id) && !blocked}
                disabled={isRecording || blocked}
                onChange={() => toggleSource(source.id)}
              />
              <span>
                {source.label} <span style={{ color: "var(--muted)" }}>· {source.note}</span>
              </span>
            </label>
            );
          })}
          <div className="spacer" />
          {isRecording ? (
            <>
              <span className="pill live">● recording {Math.round(elapsed)}s</span>
              <button className="danger" disabled={busy} onClick={() => act(stopSelectedRecording)}>
                Stop
              </button>
            </>
          ) : (
            <button
              className="primary"
              disabled={busy || !sources.length || jobRunning}
              onClick={() => act(startSelectedRecording)}
            >
              Start recording
            </button>
          )}
        </div>

        {isRecording && recording ? (
          <div className="meters">
            {Object.entries(recording.peaks || {}).map(([name, peak]) => (
              <Meter key={name} name={name} peak={peak} file={recording.continuous_files?.[name]} />
            ))}
          </div>
        ) : null}

        {sources.includes("discord") && meeting.discord_available && !isRecording ? (
          <div className="row" style={{ marginTop: 10 }}>
            <label className="hint">Voice channel</label>
            <select value={channelId} onChange={(event) => setChannelId(event.target.value)} style={{ flex: 1 }}>
              <option value="">Ask in chat with !listen</option>
              {(channels?.servers || []).map((server) =>
                server.channels.map((channel) => (
                  <option key={channel.id} value={channel.id}>
                    {server.guild} · {channel.name}
                  </option>
                ))
              )}
            </select>
          </div>
        ) : null}
        {channels?.error ? <p className="hint">Discord: {channels.error}</p> : null}

        {meeting.discord?.running ? (
          <div className="log">{(meeting.discord.log || []).slice(-8).join("\n") || "Discord receiver starting…"}</div>
        ) : null}

        {Object.keys(artifacts.segments).length ? (
          <p className="hint">
            Captured:{" "}
            {Object.entries(artifacts.segments)
              .map(([name, count]) => `${name} (${count} clip${count === 1 ? "" : "s"})`)
              .join(", ")}
          </p>
        ) : (
          <p className="hint">Nothing recorded yet. Stopping writes one clip per utterance plus a full session file.</p>
        )}
      </Step>

      {/* Not a numbered step: live transcription is a side channel, so the
          offline pipeline keeps its 1-2-3-4 numbering. */}
      <Step index="•" title="Live transcript" enabled>
        <LiveTranscriptPanel
          project={projectSlug}
          meeting={meetingSlug}
          live={meeting.live}
          onChanged={load}
        />
      </Step>

      <Step
        index={2}
        title="Transcribe"
        done={artifacts.has_transcripts && !artifacts.transcription_stale}
        enabled={can.transcribe && !isRecording}
        blockedReason={isRecording ? "recording" : "record first"}
      >
        <div className="row">
          <button
            disabled={!can.transcribe || busy || isRecording || jobRunning}
            onClick={() => act(() => api.transcribe(projectSlug, meetingSlug))}
          >
            {artifacts.has_transcripts ? "Transcribe again" : "Transcribe"}
          </button>
          <span className="hint">Each source becomes its own transcript.</span>
        </div>
      </Step>

      <Step
        index={3}
        title="Split shared audio by speaker"
        done={artifacts.has_diarized && !artifacts.diarization_stale}
        enabled={can.diarize && !isRecording}
        blockedReason="needs PC or Chrome tab audio"
      >
        <div className="row">
          <label className="hint">
            Number of people{" "}
            <input
              type="number"
              min="1"
              max="10"
              style={{ width: 70 }}
              value={numSpeakers}
              placeholder="auto"
              onChange={(event) => setNumSpeakers(event.target.value)}
            />
          </label>
          <button
            disabled={!can.diarize || busy || isRecording || jobRunning}
            onClick={() =>
              act(() =>
                api.diarize(projectSlug, meetingSlug, {
                  num_speakers: numSpeakers ? Number(numSpeakers) : null,
                  max_speakers: numSpeakers ? Number(numSpeakers) : 6
                })
              )
            }
          >
            {artifacts.has_diarized ? "Split again" : "Split by speaker"}
          </button>
          {meeting?.config?.diarization_engine === "pyannote" ? (
            <span className="hint">using pyannote</span>
          ) : null}
        </div>
        <p className="hint">
          {Object.entries(artifacts.continuous_parts || {}).filter(([name]) => name !== "mic").reduce((sum, [, count]) => sum + count, 0) > 1
            ? "All recording parts will be processed together, preserving their original times and gaps. "
            : ""}
          Set the number when you know it: left on auto it under-counts on short recordings. Your microphone is already one
          person, so only shared PC or Chrome-tab audio is split. Splitting again resets speaker names; name them after the final split.
          The model can be changed in Settings.
        </p>
      </Step>

      <Step
        index={4}
        title="Merge into one transcript"
        done={artifacts.has_merged && !artifacts.merged_stale}
        enabled={can.combine && !isRecording}
        blockedReason="needs steps 2 and 3"
      >
        <div className="row">
          <label className="hint">
            Call me{" "}
            <input value={me} style={{ width: 140 }} onChange={(event) => setMe(event.target.value)} />
          </label>
          <button
            disabled={!can.combine || busy || isRecording || jobRunning}
            onClick={() => act(() => api.combine(projectSlug, meetingSlug, me))}
          >
            {artifacts.has_merged ? "Merge again" : "Merge"}
          </button>
        </div>
      </Step>

      {artifacts.has_diarized ? (
        <Speakers
          key={`${meetingSlug}-${meeting.jobs?.find((item) => item.kind === "diarize" && item.status === "done")?.finished_at || 0}`}
          projectSlug={projectSlug}
          meetingSlug={meetingSlug}
          onRenamed={() => load().then(() => showTab(tab))}
        />
      ) : null}

      {job ? (
        <div className="card">
          <div className="card-head">
            <h3>{job.kind}</h3>
            <span className={`pill${job.status === "done" ? " good" : job.status === "error" ? " live" : ""}`}>{job.status}</span>
          </div>
          {job.error ? <div className="error">{job.error}</div> : null}
          {job.log.length ? <div className="log">{job.log.join("\n")}</div> : null}
        </div>
      ) : null}

      {availableTabs.length ? (
        <div className="card">
          <div className="tabs">
            {availableTabs.map((item) => (
              <button key={item.id} className={`tab${tab === item.id ? " active" : ""}`} onClick={() => setTab(item.id)}>
                {item.label}
              </button>
            ))}
          </div>
          {tab === "live" ? (
            <div className="transcript">
              {liveText.trim() ? (
                liveText.trimEnd().split("\n").map((line, index) => {
                  // Live lines are "[HH:MM:SS] source: text", not the pipe-separated merged format.
                  const match = line.match(/^\[(.+?)\]\s*([^:]+):\s*(.*)$/);
                  if (!match) return <div key={index} className="line"><div className="what" style={{ gridColumn: "1 / -1" }}>{line}</div></div>;
                  return (
                    <div key={index} className="line">
                      <div className="time">{match[1]}</div>
                      <div className="who">{match[2]}</div>
                      <div className="what">{match[3]}</div>
                    </div>
                  );
                })
              ) : (
                <div className="empty">Nothing transcribed live yet.</div>
              )}
            </div>
          ) : tab === "logs" ? (
            <>
              {logs?.files?.length > 1 ? (
                <select
                  value={logs.name || ""}
                  onChange={(event) =>
                    api.logs(projectSlug, meetingSlug, event.target.value).then(setLogs).catch((e) => setError(e.message))
                  }
                  style={{ marginBottom: 10 }}
                >
                  {logs.files.map((file) => (
                    <option key={file} value={file}>
                      {file}
                    </option>
                  ))}
                </select>
              ) : null}
              <div className="log" style={{ maxHeight: 320 }}>
                {logs?.lines?.length ? logs.lines.join("\n") : "No logs yet."}
              </div>
            </>
          ) : (
          <div className="transcript">
            {transcript?.lines?.length ? (
              transcript.lines.map((line, index) => {
                const match = line.match(/^\[(.+?)\]\s*(\S[^|]*?)\s*\|\s*(.*)$/);
                if (!match) return <div key={index} className="line"><div className="what" style={{ gridColumn: "1 / -1" }}>{line}</div></div>;
                return (
                  <div key={index} className="line">
                    <div className="time">{match[1]}</div>
                    <div className="who">{match[2]}</div>
                    <div className="what">{match[3]}</div>
                  </div>
                );
              })
            ) : (
              <div className="empty">Nothing here yet.</div>
            )}
          </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
