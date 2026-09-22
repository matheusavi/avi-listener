async function request(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined
  });
  if (!response.ok) {
    // FastAPI puts the reason in `detail`; surfacing it beats "request failed".
    const problem = await response.json().catch(() => ({}));
    throw new Error(problem.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function textRequest(path) {
  const response = await fetch(`/api${path}`);
  if (!response.ok) {
    const problem = await response.json().catch(() => ({}));
    throw new Error(problem.detail || `${response.status} ${response.statusText}`);
  }
  return response.text();
}

async function binaryRequest(path, body) {
  const response = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body
  });
  if (!response.ok) {
    const problem = await response.json().catch(() => ({}));
    throw new Error(problem.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

export const api = {
  projects: () => request("/projects"),
  createProject: (name) => request("/projects", { method: "POST", body: { name } }),
  projectConfig: (project, config) =>
    request(`/projects/${project}/config`, { method: "PUT", body: { config } }),

  createMeeting: (project, name) =>
    request(`/projects/${project}/meetings`, { method: "POST", body: { name } }),
  meeting: (project, meeting) => request(`/projects/${project}/meetings/${meeting}`),
  meetingConfig: (project, meeting, config) =>
    request(`/projects/${project}/meetings/${meeting}/config`, { method: "PUT", body: { config } }),

  startRecording: (project, meeting, sources) =>
    request(`/projects/${project}/meetings/${meeting}/record/start`, { method: "POST", body: { sources } }),
  pushChromeAudio: (project, meeting, sampleRate, audio) =>
    binaryRequest(
      `/projects/${project}/meetings/${meeting}/record/chrome?sample_rate=${encodeURIComponent(sampleRate)}`,
      audio.buffer.slice(audio.byteOffset, audio.byteOffset + audio.byteLength)
    ),
  stopRecording: (project, meeting) =>
    request(`/projects/${project}/meetings/${meeting}/record/stop`, { method: "POST" }),

  // Live transcription runs beside the offline pipeline: `after` is the index
  // of the first line we have not seen, so polling only carries new lines.
  live: (project, meeting, after = 0) =>
    request(`/projects/${project}/meetings/${meeting}/live?after=${after}`),
  liveStart: (project, meeting, { model_size, mode } = {}) =>
    request(`/projects/${project}/meetings/${meeting}/live/start`, {
      method: "POST",
      body: { ...(model_size ? { model_size } : {}), ...(mode ? { mode } : {}) }
    }),
  liveStop: (project, meeting) =>
    request(`/projects/${project}/meetings/${meeting}/live/stop`, { method: "POST" }),
  // Plain text, not JSON: this is the file an external agent tails.
  liveTranscript: (project, meeting) => textRequest(`/projects/${project}/meetings/${meeting}/live/transcript`),

  transcribe: (project, meeting) =>
    request(`/projects/${project}/meetings/${meeting}/transcribe`, { method: "POST" }),
  diarize: (project, meeting, body) =>
    request(`/projects/${project}/meetings/${meeting}/diarize`, { method: "POST", body }),
  combine: (project, meeting, me) =>
    request(`/projects/${project}/meetings/${meeting}/combine`, { method: "POST", body: { me } }),

  transcript: (project, meeting, kind) =>
    request(`/projects/${project}/meetings/${meeting}/transcript?kind=${kind}`),
  speakers: (project, meeting) => request(`/projects/${project}/meetings/${meeting}/speakers`),
  renameSpeakers: (project, meeting, names) =>
    request(`/projects/${project}/meetings/${meeting}/speakers`, { method: "PUT", body: { names } }),
  logs: (project, meeting, name) =>
    request(`/projects/${project}/meetings/${meeting}/logs${name ? `?name=${encodeURIComponent(name)}` : ""}`),
  defaults: () => request("/defaults"),
  devices: () => request("/devices"),
  discordChannels: (project) => request(`/projects/${project}/discord/channels`)
};
