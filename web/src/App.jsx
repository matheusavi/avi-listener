import React, { useCallback, useEffect, useState } from "react";

import MeetingView from "./MeetingView.jsx";
import Settings from "./Settings.jsx";
import { api } from "./api.js";

function NewRow({ placeholder, onCreate }) {
  const [value, setValue] = useState("");
  const submit = () => {
    const name = value.trim();
    if (!name) return;
    setValue("");
    onCreate(name);
  };
  return (
    <div className="new-row">
      <input
        value={value}
        placeholder={placeholder}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => event.key === "Enter" && submit()}
      />
      <button onClick={submit}>Add</button>
    </div>
  );
}

export default function App() {
  const [projects, setProjects] = useState([]);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);
  const [settingsFor, setSettingsFor] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const data = await api.projects();
      setProjects(data.projects);
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }, []);

  useEffect(() => {
    refresh().then((data) => {
      if (!data) return;
      // Drop straight into the newest meeting: reaching the thing you were
      // working on should not take two clicks every time.
      for (const project of data.projects) {
        if (project.meetings.length) {
          setSelected({ project: project.slug, meeting: project.meetings[0].slug });
          return;
        }
      }
    });
  }, [refresh]);

  const createProject = async (name) => {
    try {
      await api.createProject(name);
      await refresh();
    } catch (err) {
      setError(err.message);
    }
  };

  const createMeeting = async (projectSlug, name) => {
    try {
      const meeting = await api.createMeeting(projectSlug, name);
      await refresh();
      setSelected({ project: projectSlug, meeting: meeting.slug });
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>AviListener</h1>
        <div className="tagline">Record, transcribe, split by speaker.</div>

        {error ? <div className="error">{error}</div> : null}

        {projects.map((project) => (
          <div key={project.slug} className="project">
            <div className="project-name">
              <span>{project.name}</span>
              <span className="row" style={{ gap: 6 }}>
                <span className="pill">{project.meetings.length}</span>
                <button className="icon-button" title="Settings" onClick={() => setSettingsFor(project)}>
                  ⚙
                </button>
              </span>
            </div>
            {project.meetings.map((meeting) => (
              <button
                key={meeting.slug}
                className={`meeting-item${
                  selected?.project === project.slug && selected?.meeting === meeting.slug ? " active" : ""
                }`}
                onClick={() => setSelected({ project: project.slug, meeting: meeting.slug })}
              >
                {meeting.name}
                <small>
                  {meeting.artifacts.has_merged
                    ? "merged"
                    : meeting.artifacts.has_diarized
                    ? "split by speaker"
                    : meeting.artifacts.has_transcripts
                    ? "transcribed"
                    : meeting.artifacts.has_recording
                    ? "recorded"
                    : "empty"}
                </small>
              </button>
            ))}
            <NewRow placeholder="New meeting…" onCreate={(name) => createMeeting(project.slug, name)} />
          </div>
        ))}

        <div style={{ marginTop: 20, borderTop: "1px solid var(--line)", paddingTop: 14 }}>
          <NewRow placeholder="New project…" onCreate={createProject} />
        </div>
      </aside>

      {selected ? (
        <MeetingView
          key={`${selected.project}/${selected.meeting}`}
          projectSlug={selected.project}
          meetingSlug={selected.meeting}
          onChanged={refresh}
        />
      ) : (
        <div className="main">
          <div className="empty">
            {projects.length ? "Pick or create a meeting to start." : "Create a project, then a meeting inside it."}
          </div>
        </div>
      )}

      {settingsFor ? (
        <Settings
          project={projects.find((item) => item.slug === settingsFor.slug) || settingsFor}
          onSaved={refresh}
          onClose={() => setSettingsFor(null)}
        />
      ) : null}
    </div>
  );
}
