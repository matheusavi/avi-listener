import React, { useEffect, useState } from "react";

import { api } from "./api.js";

const LANGUAGES = [
  { id: "pt", label: "Portuguese" },
  { id: "en", label: "English" },
  { id: "es", label: "Spanish" },
  { id: "", label: "Detect automatically" }
];

export default function Settings({ project, onSaved, onClose }) {
  const [devices, setDevices] = useState({ microphones: [], speakers: [] });
  const [tokenInEnv, setTokenInEnv] = useState(false);
  const [pyannote, setPyannote] = useState({ ok: false, reason: "checking…" });
  const [form, setForm] = useState(() => ({
    language: project.config.language ?? "pt",
    device: project.config.device ?? "cuda",
    model_size: project.config.model_size ?? "large-v3",
    diarization_engine: project.config.diarization_engine ?? "nemo",
    me: project.config.me ?? "host",
    mic: project.config.devices?.mic ?? "",
    system: project.config.devices?.system ?? "",
    discord_channel_id: project.config.discord_channel_id ?? "",
    discord_token: "",
    hf_token: "",
    hotwords: (project.config.transcription?.hotwords ?? []).join("\n"),
    ignored: (project.config.transcription?.ignored_phrases ?? []).join("\n")
  }));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.devices().then(setDevices).catch((err) => setError(err.message));
    api.defaults().then((data) => {
      setTokenInEnv(Boolean(data.discord_token_in_env));
      setPyannote(data.pyannote ?? { ok: false, reason: "server too old" });
    }).catch(() => {});
  }, []);

  const set = (key) => (event) => setForm((current) => ({ ...current, [key]: event.target.value }));

  // A device saved by hand, or one that has since been unplugged, would match
  // no option and silently render as "System default" while still being used.
  // Keep showing it so what is saved is always what is displayed.
  const withSaved = (options, saved) =>
    saved && !options.some((item) => item.name === saved) ? [{ name: saved, missing: true }, ...options] : options;

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const config = {
        language: form.language,
        device: form.device,
        model_size: form.model_size,
        diarization_engine: form.diarization_engine,
        me: form.me,
        devices: { mic: form.mic || null, system: form.system || null },
        discord_channel_id: form.discord_channel_id,
        // Merge into the tuned block rather than replacing it, so editing a
        // word list cannot silently drop the thresholds around it.
        transcription: {
          ...(project.config.transcription || {}),
          hotwords: form.hotwords.split("\n").map((line) => line.trim()).filter(Boolean),
          ignored_phrases: form.ignored.split("\n").map((line) => line.trim()).filter(Boolean)
        }
      };
      // Only send the token when one was typed, so saving other settings does
      // not wipe a token that is already stored.
      if (form.discord_token.trim()) config.discord_token = form.discord_token.trim();
      if (form.hf_token.trim()) config.hf_token = form.hf_token.trim();
      await api.projectConfig(project.slug, config);
      await onSaved?.();
      onClose?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(event) => event.stopPropagation()}>
        <div className="card-head">
          <h3>Settings · {project.name}</h3>
          <div className="spacer" />
          <button onClick={onClose}>Close</button>
        </div>

        {error ? <div className="error">{error}</div> : null}

        <h4>Devices</h4>
        <div className="field">
          <label>Microphone</label>
          <select value={form.mic} onChange={set("mic")}>
            <option value="">System default</option>
            {withSaved(devices.microphones, form.mic).map((item) => (
              <option key={item.name} value={item.name}>
                {item.name}
                {item.default ? " (default)" : ""}
                {item.missing ? " (not found now)" : ""}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>PC audio</label>
          <select value={form.system} onChange={set("system")}>
            <option value="">System default</option>
            {withSaved(devices.speakers, form.system).map((item) => (
              <option key={item.name} value={item.name}>
                {item.name}
                {item.default ? " (default)" : ""}
                {item.missing ? " (not found now)" : ""}
              </option>
            ))}
          </select>
        </div>
        <p className="hint">
          PC audio is captured from an output, so pick the device you actually hear the call through.
        </p>

        <h4>Transcription</h4>
        <div className="field">
          <label>Language</label>
          <select value={form.language} onChange={set("language")}>
            {LANGUAGES.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Model</label>
          <select value={form.model_size} onChange={set("model_size")}>
            {["large-v3", "medium", "small", "base"].map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Run on</label>
          <select value={form.device} onChange={set("device")}>
            <option value="cuda">GPU (cuda)</option>
            <option value="cpu">CPU</option>
          </select>
        </div>
        <div className="field">
          <label>Call me</label>
          <input value={form.me} onChange={set("me")} />
        </div>

        <h4>Split by speaker</h4>
        <div className="field">
          <label>Model</label>
          <select value={form.diarization_engine} onChange={set("diarization_engine")}>
            <option value="nemo">NeMo clustering (local)</option>
            <option value="pyannote" disabled={!pyannote.ok}>
              pyannote community-1{pyannote.ok ? "" : ` — unavailable: ${pyannote.reason}`}
            </option>
          </select>
        </div>
        <div className="field">
          <label>Hugging Face token</label>
          <input
            type="password"
            placeholder={
              project.config.hf_token_set
                ? "•••••••• saved"
                : pyannote.token_in_env
                ? "using HF_TOKEN from your environment"
                : "paste token (hf_…)"
            }
            value={form.hf_token}
            onChange={set("hf_token")}
          />
        </div>
        <p className="hint">
          pyannote separates similar voices better and understands two people talking at once. Its model is gated:
          create a token at huggingface.co/settings/tokens (read access) and accept the conditions on the
          pyannote/speaker-diarization-community-1 page. A token pasted here is stored on this machine and never sent
          back to the browser; the <code>HF_TOKEN</code> environment variable also works.
        </p>

        <div className="field field-tall">
          <label>Names &amp; jargon</label>
          <textarea
            rows={4}
            placeholder={"one per line\nAvi\nmaxtuist"}
            value={form.hotwords}
            onChange={set("hotwords")}
          />
        </div>
        <p className="hint">Words Whisper otherwise misspells: people, games, car models.</p>
        <div className="field field-tall">
          <label>Drop these lines</label>
          <textarea rows={4} placeholder="one per line" value={form.ignored} onChange={set("ignored")} />
        </div>
        <p className="hint">
          Phrases Whisper invents over silence, like subtitle credits or "se inscreva no canal". Matching lines are
          discarded.
        </p>

        <h4>Discord</h4>
        <p className="hint">
          Discord names whoever is speaking, so it produces one file per person and needs no speaker splitting.
        </p>
        <div className="field">
          <label>Bot token</label>
          <input
            type="password"
            placeholder={
              project.config.discord_token_set
                ? "•••••••• saved"
                : tokenInEnv
                ? "using DISCORD_BOT_TOKEN from your environment"
                : "paste token"
            }
            value={form.discord_token}
            onChange={set("discord_token")}
          />
        </div>
        <div className="field">
          <label>Voice channel ID</label>
          <input placeholder="optional, joins automatically" value={form.discord_channel_id} onChange={set("discord_channel_id")} />
        </div>
        <details className="hint">
          <summary style={{ cursor: "pointer" }}>How do I get a bot token?</summary>
          <ol style={{ paddingLeft: 18, marginTop: 8 }}>
            <li>
              Open the{" "}
              <a href="https://discord.com/developers/applications" target="_blank" rel="noreferrer">
                Discord Developer Portal
              </a>{" "}
              and create an application.
            </li>
            <li>Under <strong>Bot</strong>, click <strong>Reset Token</strong> and copy it. It is shown only once.</li>
            <li>Enable the <strong>Message Content Intent</strong> on the same page.</li>
            <li>
              Under <strong>OAuth2 → URL Generator</strong> tick <strong>bot</strong>, then the permissions
              <strong> View Channels</strong>, <strong>Send Messages</strong>, <strong>Connect</strong> and{" "}
              <strong>Speak</strong>. Open the generated URL to invite it to your server.
            </li>
            <li>
              For the channel ID: enable <strong>Developer Mode</strong> in Discord settings, then right-click the voice
              channel and choose <strong>Copy Channel ID</strong>. Leave it blank to type <code>!listen</code> in Discord
              instead.
            </li>
          </ol>
          <p>
            If <code>DISCORD_BOT_TOKEN</code> is set in your environment it is used automatically and there is nothing to
            paste. A token typed here takes precedence and is stored on this machine in the workspace folder; it is never
            sent back to the browser. Treat it like a password: anyone holding it can act as your bot.
          </p>
        </details>

        <div className="row" style={{ marginTop: 16 }}>
          <div className="spacer" />
          <button className="primary" disabled={saving} onClick={save}>
            {saving ? "Saving…" : "Save settings"}
          </button>
        </div>
      </div>
    </div>
  );
}
