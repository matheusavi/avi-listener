import React, { useCallback, useEffect, useState } from "react";

import { api } from "./api.js";

export default function Speakers({ projectSlug, meetingSlug, onRenamed }) {
  const [speakers, setSpeakers] = useState([]);
  const [names, setNames] = useState({});
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState(null);

  const load = useCallback(async () => {
    try {
      const data = await api.speakers(projectSlug, meetingSlug);
      setSpeakers(data.speakers || []);
      setNames(Object.fromEntries((data.speakers || []).map((item) => [item.label, item.name || ""])));
    } catch (err) {
      setError(err.message);
    }
  }, [projectSlug, meetingSlug]);

  useEffect(() => {
    load();
  }, [load]);

  if (!speakers.length) return null;

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const result = await api.renameSpeakers(projectSlug, meetingSlug, names);
      setStatus(result.merged_updated ? "Names saved, transcript updated." : "Names saved.");
      await load();
      await onRenamed?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="card">
      <div className="card-head">
        <h3>Who is who</h3>
        <span className="pill">{speakers.length} detected</span>
      </div>
      <p className="hint">
        Splitting by voice cannot know names. Play a few seconds of each, type who it is, and the transcript is relabelled.
      </p>

      {error ? <div className="error">{error}</div> : null}

      {speakers.map((speaker) => (
        <div key={speaker.label} className="speaker-row">
          <span className="who">{speaker.label}</span>
          <span className="hint">{speaker.seconds}s</span>
          {speaker.has_sample ? (
            <audio
              controls
              preload="none"
              src={`/api/projects/${projectSlug}/meetings/${meetingSlug}/speakers/${speaker.label}/sample.wav`}
            />
          ) : (
            <span className="hint">no sample</span>
          )}
          <input
            placeholder="name this person"
            value={names[speaker.label] ?? ""}
            onChange={(event) => setNames((current) => ({ ...current, [speaker.label]: event.target.value }))}
            onKeyDown={(event) => event.key === "Enter" && save()}
          />
        </div>
      ))}

      <div className="row" style={{ marginTop: 10 }}>
        {status ? <span className="hint">{status}</span> : null}
        <div className="spacer" />
        <button className="primary" disabled={saving} onClick={save}>
          {saving ? "Saving…" : "Save names"}
        </button>
      </div>
    </div>
  );
}
