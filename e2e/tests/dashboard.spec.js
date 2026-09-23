// What the dashboard has to get right, driven through the real interface
// against a real server.
//
// Recording, transcription and diarization are deliberately not run: they need
// a microphone, a GPU and a Whisper model, and a test that needs those is a
// test nobody runs. Everything they produce is seeded on disk instead, which
// leaves the parts that are actually this project's logic - what may be done
// next, merging two clocks into one timeline, naming speakers, and keeping
// tokens out of the browser - fully exercised.

import { copyFileSync, existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";

import { expect, test } from "@playwright/test";

import { workspaceDir } from "../environment.js";

/** Open a meeting from the sidebar by its name. */
async function openMeeting(page, name) {
  await page.goto("/");
  await page.getByRole("button", { name: new RegExp(`^${name}`) }).click();
  await expect(page.getByRole("heading", { level: 2, name })).toBeVisible();
}

/** The card for one numbered step, found by its heading. */
function stepCard(page, title) {
  return page.locator(".card").filter({ has: page.getByRole("heading", { name: title }) });
}

/** The card holding the transcript/log tabs, at the bottom of a meeting. */
function tabCard(page) {
  return page.locator(".card").filter({ has: page.locator(".tabs") });
}

/** A 16 kHz mono PCM16 WAV holding a short tone - the shape both recorders write. */
function wavClip(seconds = 0.5, frequency = 220) {
  const rate = 16000;
  const frames = Math.round(seconds * rate);
  const audio = Buffer.alloc(frames * 2);
  for (let index = 0; index < frames; index += 1) {
    audio.writeInt16LE(Math.round(6000 * Math.sin((2 * Math.PI * frequency * index) / rate)), index * 2);
  }
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + audio.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20); // PCM
  header.writeUInt16LE(1, 22); // mono
  header.writeUInt32LE(rate, 24);
  header.writeUInt32LE(rate * 2, 28); // bytes per second
  header.writeUInt16LE(2, 32); // block align
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(audio.length, 40);
  return Buffer.concat([header, audio]);
}

/**
 * Write one clip the way a recorder would: `<utc stamp>Z-<source>-<digits>.wav`
 * (see `segment_filename` in avilistener/recorder.py). The stamp is taken now,
 * which is what puts the clip after the running session's watermark; the
 * trailing digits are the source's id and any digits will do.
 */
function writeClip(meetingDir, source) {
  const stamp = new Date().toISOString().replace(/:/g, "-").replace(".", "-");
  const name = `${stamp}-${source}-1000000001.wav`;
  const directory = path.join(meetingDir, "recordings");
  mkdirSync(directory, { recursive: true });
  writeFileSync(path.join(directory, name), wavClip());
  return name;
}

/** The "Call me" box in a meeting's merge step, where it labels the input. */
function callMeInMeeting(scope) {
  return scope.locator("label", { hasText: "Call me" }).locator("input");
}

/** The "Call me" box in Settings, where the label is a sibling of the input. */
function callMeInSettings(modal) {
  return modal.locator(".field", { hasText: "Call me" }).locator("input");
}

/** Open one project's settings by name, so tests cannot edit each other's. */
async function openSettings(page, projectName) {
  await page.goto("/");
  const project = page.locator(".project").filter({ hasText: projectName });
  await expect(project).toBeVisible();
  await project.getByTitle("Settings").click();
  const modal = page.locator(".modal");
  await expect(modal).toContainText(`Settings · ${projectName}`);
  return modal;
}

test.describe("projects and meetings", () => {
  test("the sidebar lists the seeded project and its meetings", async ({ page }) => {
    await page.goto("/");

    await expect(page.getByText("E2E Project")).toBeVisible();
    for (const name of ["Empty Meeting", "Recorded Meeting", "Merge Meeting", "Speakers Meeting"]) {
      await expect(page.getByRole("button", { name: new RegExp(`^${name}`) })).toBeVisible();
    }
  });

  test("each meeting is labelled with how far it has got", async ({ page }) => {
    await page.goto("/");

    await expect(page.getByRole("button", { name: /^Empty Meeting/ })).toContainText("empty");
    await expect(page.getByRole("button", { name: /^Recorded Meeting/ })).toContainText("recorded");
    await expect(page.getByRole("button", { name: /^Speakers Meeting/ })).toContainText("split by speaker");
  });

  test("a project and a meeting can be created", async ({ page }) => {
    await page.goto("/");

    await page.getByPlaceholder("New project…").fill("Sales Calls");
    await page.getByPlaceholder("New project…").press("Enter");
    await expect(page.getByText("Sales Calls")).toBeVisible();

    const project = page.locator(".project").filter({ hasText: "Sales Calls" });
    await project.getByPlaceholder("New meeting…").fill("Kickoff");
    await project.getByPlaceholder("New meeting…").press("Enter");

    // Creating a meeting opens it, which is the whole point of creating one.
    await expect(page.getByRole("heading", { level: 2, name: "Kickoff" })).toBeVisible();
    await expect(page.getByText("sales-calls / kickoff")).toBeVisible();
  });

  test("an empty name creates nothing", async ({ page }) => {
    await page.goto("/");
    // Count only once the list has rendered: counting an empty page proves
    // nothing about what a blank name does.
    await expect(page.locator(".project").filter({ hasText: "E2E Project" })).toBeVisible();
    const before = await page.locator(".project").count();

    await page.getByPlaceholder("New project…").fill("   ");
    await page.getByPlaceholder("New project…").press("Enter");

    await expect(page.locator(".project")).toHaveCount(before);
  });
});

test.describe("what may be done next", () => {
  test("a meeting with nothing recorded offers nothing but recording", async ({ page }) => {
    await openMeeting(page, "Empty Meeting");

    await expect(page.getByRole("button", { name: "Start recording" })).toBeEnabled();
    await expect(stepCard(page, "Transcribe").getByRole("button", { name: "Transcribe" })).toBeDisabled();
    await expect(
      stepCard(page, "Split shared audio by speaker").getByRole("button", { name: "Split by speaker" })
    ).toBeDisabled();
    await expect(stepCard(page, "Merge into one transcript").getByRole("button", { name: "Merge" })).toBeDisabled();
    await expect(stepCard(page, "Transcribe")).toContainText("record first");
  });

  test("a recorded meeting can transcribe and split, but not merge", async ({ page }) => {
    await openMeeting(page, "Recorded Meeting");

    await expect(stepCard(page, "Transcribe").getByRole("button", { name: "Transcribe" })).toBeEnabled();
    await expect(
      stepCard(page, "Split shared audio by speaker").getByRole("button", { name: "Split by speaker" })
    ).toBeEnabled();
    await expect(stepCard(page, "Merge into one transcript").getByRole("button", { name: "Merge" })).toBeDisabled();
    await expect(stepCard(page, "Merge into one transcript")).toContainText("needs steps 2 and 3");
    await expect(page.getByText(/mic \(2 clips\)/)).toBeVisible();
  });

  test("a transcribed and split meeting can merge", async ({ page }) => {
    await openMeeting(page, "Merge Meeting");

    await expect(stepCard(page, "Merge into one transcript").getByRole("button", { name: "Merge" })).toBeEnabled();
    await expect(page.getByText("· recorded")).toBeVisible();
  });
});

test.describe("merging into one timeline", () => {
  test("a resumed meeting merges both parts with the real gap and playable samples", async ({ page, request }) => {
    await openMeeting(page, "Resumed Meeting");
    await expect(page.getByText(/All recording parts will be processed together/)).toBeVisible();
    await stepCard(page, "Merge into one transcript").getByRole("button", { name: "Merge", exact: true }).click();
    await expect(page.getByRole("button", { name: "Everyone (merged)" })).toBeVisible();
    await page.getByRole("button", { name: "Everyone (merged)" }).click();
    const transcript = page.locator(".transcript");
    await expect(transcript).toContainText("Before interruption");
    await expect(transcript).toContainText("After resuming");
    await expect(transcript).toContainText("Microphone 0");
    await expect(transcript).toContainText("Microphone 1000");
    await expect(transcript).not.toContainText("duplicate clip");
    await expect(transcript).toContainText("12:48:49");
    await expect(transcript).toContainText("13:05:29");
    const response = await request.get("/api/projects/e2e-project/meetings/resumed-meeting/speakers/speaker_0/sample.wav");
    expect(response.status()).toBe(200);
    expect((await response.body()).length).toBeGreaterThan(1000);
    const merge = stepCard(page, "Merge into one transcript");
    await callMeInMeeting(merge).fill("Resumed Host");
    await merge.getByRole("button", { name: "Merge again" }).click();
    // The existing tab must refresh after a repeated job, without navigating away.
    await expect(transcript).toContainText("Resumed Host");
    const recordings = path.join(workspaceDir, "e2e-project", "resumed-meeting", "recordings", "continuous");
    copyFileSync(path.join(recordings, "2026-08-24T15-48-48-732Z-chrome-continuous.wav"),
      path.join(recordings, "2026-08-24T16-30-00-000Z-chrome-continuous.wav"));
    await page.reload();
    await page.getByRole("button", { name: /^Resumed Meeting/ }).click();
    await expect(page.getByText(/New recording parts detected/)).toBeVisible();
    await expect(stepCard(page, "Merge into one transcript").getByRole("button", { name: "Merge again" })).toBeDisabled();
    const stale = await request.post("/api/projects/e2e-project/meetings/resumed-meeting/combine", { data: { me: "host" } });
    expect(stale.status()).toBe(409);
  });

  test("merging interleaves the microphone with the split speakers", async ({ page }) => {
    await openMeeting(page, "Merge Meeting");

    const merge = stepCard(page, "Merge into one transcript");
    await callMeInMeeting(merge).fill("Alice");
    await merge.getByRole("button", { name: "Merge" }).click();

    await expect(page.getByRole("button", { name: "Everyone (merged)" })).toBeVisible({ timeout: 20_000 });
    await page.getByRole("button", { name: "Everyone (merged)" }).click();

    const transcript = page.locator(".transcript");
    // Both sides of the call, on one clock: the diarized loopback speakers and
    // the microphone, which is now called Alice rather than "mic".
    await expect(transcript).toContainText("speaker_0");
    await expect(transcript).toContainText("speaker_1");
    await expect(transcript).toContainText("Alice");
    await expect(transcript).toContainText("Tudo certo por aqui tambem.");

    // The loopback's own segments must not appear twice: they are already
    // covered by the diarized lines.
    await expect(transcript).not.toContainText("system");

    const order = await page.locator(".transcript .who").allTextContents();
    expect(order.map((item) => item.trim())).toEqual([
      "speaker_0",
      "speaker_1",
      "Alice",
      "speaker_0",
      "Alice",
    ]);
  });
});

test.describe("naming the speakers", () => {
  test("a named speaker is relabelled everywhere", async ({ page }) => {
    await openMeeting(page, "Speakers Meeting");

    // Merge first, so there is a transcript for the rename to update.
    await stepCard(page, "Merge into one transcript").getByRole("button", { name: "Merge" }).click();
    await expect(page.getByRole("button", { name: "Everyone (merged)" })).toBeVisible({ timeout: 20_000 });

    const panel = page.locator(".card").filter({ has: page.getByRole("heading", { name: "Who is who" }) });
    await expect(panel).toContainText("2 detected");

    // Ordered by how much each spoke, so speaker_0 leads.
    const rows = panel.locator(".speaker-row");
    await expect(rows.first()).toContainText("speaker_0");
    await expect(rows.first().locator("audio")).toHaveCount(1);

    await rows.first().getByPlaceholder("name this person").fill("Bruno");
    await panel.getByRole("button", { name: "Save names" }).click();
    await expect(panel).toContainText("Names saved, transcript updated.");

    await page.getByRole("button", { name: "Everyone (merged)" }).click();
    const transcript = page.locator(".transcript");
    await expect(transcript).toContainText("Bruno");
    await expect(transcript).not.toContainText("speaker_0");
    // The speaker who was not named keeps their label.
    await expect(transcript).toContainText("speaker_1");
  });

  test("a speaker sample is playable audio", async ({ page, request }) => {
    await openMeeting(page, "Speakers Meeting");
    await expect(page.locator(".speaker-row audio").first()).toBeVisible();

    const response = await request.get(
      "/api/projects/e2e-project/meetings/speakers-meeting/speakers/speaker_1/sample.wav"
    );
    expect(response.status()).toBe(200);
    expect(response.headers()["content-type"]).toBe("audio/wav");
    expect((await response.body()).length).toBeGreaterThan(1000);
  });
});

test.describe("reading the results", () => {
  test("the transcript tabs show each view of the meeting", async ({ page }) => {
    await openMeeting(page, "Merge Meeting");

    await page.getByRole("button", { name: "Shared audio by speaker" }).click();
    await expect(page.locator(".transcript")).toContainText("Consigo ouvir sem problemas.");

    await page.getByRole("button", { name: "By source" }).click();
    await expect(page.locator(".transcript")).toContainText("Combinado, obrigado pessoal.");
  });

  test("the logs tab shows what the recorder wrote", async ({ page }) => {
    await openMeeting(page, "Recorded Meeting");

    // Scoped to the tab card: the live panel renders a `.log` of its own.
    await page.locator(".tabs").getByRole("button", { name: "Logs" }).click();
    await expect(tabCard(page).locator(".log")).toContainText("recording started: mic, system");
  });

  test("a meeting with no logs says so instead of failing", async ({ page }) => {
    await openMeeting(page, "Empty Meeting");

    await page.locator(".tabs").getByRole("button", { name: "Logs" }).click();
    await expect(tabCard(page).locator(".log")).toContainText("No logs yet.");
  });

  test("a second log can be picked from the list", async ({ page }) => {
    await openMeeting(page, "Logs Meeting");
    await page.locator(".tabs").getByRole("button", { name: "Logs" }).click();

    // Newest first, so the transcribe log is what opens.
    await expect(tabCard(page).locator(".log")).toContainText("Transcribed 2 file(s)");

    const picker = tabCard(page).locator("select");
    await expect(picker).toBeVisible();
    await picker.selectOption("2026-08-24T15-48-48-recording.log");
    await expect(tabCard(page).locator(".log")).toContainText("recording started: mic, system");
  });
});

test.describe("recording", () => {
  test("nothing is recording when the dashboard is idle", async ({ request }) => {
    const body = await (await request.get("/api/recording")).json();
    expect(body.recording).toBeNull();
  });

  test("stopping when nothing is recording is harmless", async ({ request }) => {
    // The button is hidden while idle, but a stale tab can still send this and
    // it must not become a 500.
    const response = await request.post("/api/projects/e2e-project/meetings/empty-meeting/record/stop");
    expect(response.status()).toBe(200);
    expect((await response.json()).recording).toBeNull();
  });

  test("Discord is offered but unusable without a token", async ({ page }) => {
    await openMeeting(page, "Empty Meeting");

    const discord = page.locator("label.toggle", { hasText: "Discord" });
    await expect(discord).toBeVisible();
    await expect(discord.locator("input")).toBeDisabled();
    await expect(discord).toHaveAttribute("title", /DISCORD_BOT_TOKEN/);
  });
});

test.describe("meetings override their project", () => {
  test("a meeting keeps its own settings without touching its siblings", async ({ request }) => {
    const update = await request.put("/api/projects/e2e-project/meetings/empty-meeting/config", {
      data: { config: { language: "en", num_speakers: 3 } },
    });
    expect(update.status()).toBe(200);

    const changed = await (await request.get("/api/projects/e2e-project/meetings/empty-meeting")).json();
    expect(changed.config.language).toBe("en");
    expect(changed.config.num_speakers).toBe(3);

    // The project, and every other meeting in it, keep the inherited value.
    const sibling = await (await request.get("/api/projects/e2e-project/meetings/recorded-meeting")).json();
    expect(sibling.config.language).toBe("pt");
  });
});

test.describe("jobs", () => {
  test("a finished job can be read back by id", async ({ page, request }) => {
    await openMeeting(page, "Merge Meeting");
    await stepCard(page, "Merge into one transcript").getByRole("button", { name: "Merge" }).click();
    await expect(page.getByRole("button", { name: "Everyone (merged)" })).toBeVisible({ timeout: 20_000 });

    const meeting = await (await request.get("/api/projects/e2e-project/meetings/merge-meeting")).json();
    const [job] = meeting.jobs;
    expect(job.kind).toBe("combine");

    // The merged transcript appears as soon as the files are written, which is
    // a moment before the job marks itself finished.
    await expect
      .poll(async () => (await (await request.get(`/api/jobs/${job.id}`)).json()).status)
      .toBe("done");

    const byId = await (await request.get(`/api/jobs/${job.id}`)).json();
    expect(byId.error).toBeNull();
    expect(byId.log.join("\n")).toContain("speaker(s)");
  });

  test("an unknown job id is a 404", async ({ request }) => {
    const response = await request.get("/api/jobs/does-not-exist");
    expect(response.status()).toBe(404);
  });
});

test.describe("test isolation", () => {
  test("the server writes to the throwaway workspace, not the checkout", async ({ request }) => {
    // The guard for the whole suite. If AVILISTENER_WORKSPACE were dropped,
    // every test above would still pass or fail for its own reasons while
    // quietly editing the workspace holding real meetings.
    const created = await request.post("/api/projects", { data: { name: "Isolation Check" } });
    expect(created.status()).toBe(200);

    expect(existsSync(path.join(workspaceDir, "isolation-check", "project.json"))).toBe(true);
  });
});

test.describe("settings", () => {
  test("settings are saved and read back", async ({ page }) => {
    const modal = await openSettings(page, "Settings Project");

    await modal.locator("select").filter({ hasText: "Portuguese" }).selectOption("en");
    await callMeInSettings(modal).fill("Carla");
    await modal.getByPlaceholder(/one per line\s*Avi/).fill("AviListener\npyannote");
    await modal.getByRole("button", { name: "Save settings" }).click();
    await expect(page.locator(".modal")).toHaveCount(0);

    const reopened = await openSettings(page, "Settings Project");
    await expect(reopened.locator("select").filter({ hasText: "English" })).toHaveValue("en");
    await expect(callMeInSettings(reopened)).toHaveValue("Carla");
    await expect(reopened.getByPlaceholder(/one per line\s*Avi/)).toHaveValue("AviListener\npyannote");
  });

  test("editing a word list leaves the tuned thresholds alone", async ({ page, request }) => {
    const modal = await openSettings(page, "Settings Project");
    await modal.getByPlaceholder(/one per line\s*Avi/).fill("Whisper");
    await modal.getByRole("button", { name: "Save settings" }).click();
    await expect(page.locator(".modal")).toHaveCount(0);

    const body = await (await request.get("/api/projects")).json();
    const project = body.projects.find((item) => item.slug === "settings-project");
    expect(project.config.transcription.hotwords).toEqual(["Whisper"]);
    // Saving a word list must not drop the thresholds it sits beside.
    expect(project.config.transcription.beam_size).toBe(8);
    expect(project.config.transcription.vad_filter).toBe(true);
  });

  test("the device pickers offer what the machine has", async ({ page }) => {
    const modal = await openSettings(page, "Settings Project");

    // Every machine has at least the "System default" entry; asserting on real
    // device names would tie the suite to the machine running it.
    await expect(modal.getByText("Microphone", { exact: true })).toBeVisible();
    await expect(modal.getByText("PC audio", { exact: true })).toBeVisible();
    await expect(modal.locator("select").first().locator("option")).not.toHaveCount(0);
  });

  test("a saved token never comes back to the browser", async ({ page, request }) => {
    const modal = await openSettings(page, "Token Project");

    await modal.getByPlaceholder("paste token (hf_…)").fill("hf_pretend_token_value");
    await modal.getByRole("button", { name: "Save settings" }).click();
    await expect(page.locator(".modal")).toHaveCount(0);

    // The API must report that a token exists without ever sending it.
    const body = await (await request.get("/api/projects")).json();
    const project = body.projects.find((item) => item.slug === "token-project");
    expect(project.config.hf_token_set).toBe(true);
    expect(JSON.stringify(body)).not.toContain("hf_pretend_token_value");

    // And the interface shows it as saved rather than blank.
    const reopened = await openSettings(page, "Token Project");
    await expect(reopened.getByPlaceholder("•••••••• saved")).toHaveCount(1);
  });
});

test.describe("the API refuses work it cannot do", () => {
  test("transcribing with nothing recorded is refused with a reason", async ({ request }) => {
    const response = await request.post("/api/projects/e2e-project/meetings/empty-meeting/transcribe");
    expect(response.status()).toBe(409);
    expect((await response.json()).detail).toBe("Nothing recorded yet");
  });

  test("splitting with no loopback recording is refused", async ({ request }) => {
    const response = await request.post("/api/projects/e2e-project/meetings/empty-meeting/diarize", {
      data: { max_speakers: 6 },
    });
    expect(response.status()).toBe(409);
    expect((await response.json()).detail).toBe("No loopback recording to diarize");
  });

  test("merging without both halves is refused", async ({ request }) => {
    const response = await request.post("/api/projects/e2e-project/meetings/recorded-meeting/combine", {
      data: { me: "host" },
    });
    expect(response.status()).toBe(409);
    expect((await response.json()).detail).toContain("Need both");
  });

  test("pyannote without a token explains what is missing", async ({ request }) => {
    const response = await request.post("/api/projects/e2e-project/meetings/merge-meeting/diarize", {
      data: { max_speakers: 6, engine: "pyannote" },
    });
    expect(response.status()).toBe(409);
    expect((await response.json()).detail).toContain("Hugging Face token");
  });

  test("a meeting that does not exist is a 404, not a crash", async ({ request }) => {
    const response = await request.get("/api/projects/e2e-project/meetings/no-such-meeting");
    expect(response.status()).toBe(404);
  });

  test("a speaker label cannot escape the meeting folder", async ({ request }) => {
    const response = await request.get(
      "/api/projects/e2e-project/meetings/speakers-meeting/speakers/..%2F..%2Fproject/sample.wav"
    );
    expect(response.status()).toBe(404);
  });
});

test.describe("Chrome tab audio", () => {
  test("records one selected tab as its own source", async ({ page }) => {
    await page.addInitScript(() => {
      navigator.mediaDevices.getDisplayMedia = async () => {
        const context = new AudioContext({ sampleRate: 16000 });
        await context.resume();
        const oscillator = context.createOscillator();
        const gain = context.createGain();
        const destination = context.createMediaStreamDestination();
        gain.gain.value = 0.2;
        oscillator.connect(gain);
        gain.connect(destination);
        oscillator.start();
        window.__e2eTabAudio = { context, oscillator };
        return destination.stream;
      };
    });
    await openMeeting(page, "Empty Meeting");

    const record = stepCard(page, "Record");
    await record.getByLabel(/^My microphone/).uncheck();
    await record.getByLabel(/^Chrome tab/).check();
    await expect(record.getByLabel(/^PC audio/)).not.toBeChecked();
    await record.getByRole("button", { name: "Start recording" }).click();
    await expect(record).toContainText("recording");
    await expect(record).toContainText(/\d+(\.\d+)? KB · growing/);

    await page.waitForTimeout(1200);
    await record.getByRole("button", { name: "Stop" }).click();

    await expect(record).toContainText(/chrome \(1 clip/);
    await expect(stepCard(page, "Split shared audio by speaker").getByRole("button", { name: "Split by speaker" })).toBeEnabled();
  });
});

test.describe("live transcript", () => {
  // The one test that watches files on disk turn into text on screen. Clips
  // are written by the test rather than recorded, exactly as the Discord
  // receiver writes them from another process - which is the case the scan
  // loop exists for. The transcriber is the fake one selected by
  // AVILISTENER_LIVE_FAKE_TRANSCRIBER in the config, so no model is loaded.
  const meetingDir = path.join(workspaceDir, "e2e-project", "live-meeting");
  const liveLine = (source) =>
    new RegExp(`^\\[\\d{2}:\\d{2}:\\d{2}\\] ${source}: simulated transcript of ${source} clip$`);

  test("clips dropped into recordings become live lines, and stop resumes where it left off", async ({
    page,
    request
  }) => {
    // Longer than the suite default: each clip has to age a second before it
    // is eligible, and this walks through start, stop and resume.
    test.setTimeout(90_000);

    await openMeeting(page, "Live Meeting");
    const card = stepCard(page, "Live transcript");
    const log = card.locator(".log");
    const pill = card.locator(".pill");

    await card.locator("select").selectOption("tiny");
    await card.getByRole("button", { name: "Start live" }).click();
    await expect(pill).toContainText("running");
    await expect(pill).toContainText("tiny");

    // Only now, so the filenames are stamped after the session's watermark.
    const first = writeClip(meetingDir, "mic");
    const second = writeClip(meetingDir, "system");

    // A second of file age plus a second of scan, then the panel's own poll.
    await expect(log.locator("div")).toHaveCount(2, { timeout: 15_000 });
    const shown = await log.locator("div").allTextContents();
    expect(shown[0]).toMatch(liveLine("mic"));
    expect(shown[1]).toMatch(liveLine("system"));

    const transcript = await (
      await request.get("/api/projects/e2e-project/meetings/live-meeting/live/transcript")
    ).text();
    const written = transcript.trimEnd().split("\n");
    expect(written).toHaveLength(2);
    expect(written[0]).toMatch(liveLine("mic"));
    expect(written[1]).toMatch(liveLine("system"));

    // What was fed to the model is kept beside the transcript...
    expect(readdirSync(path.join(meetingDir, "live", "clips")).sort()).toEqual([first, second].sort());
    // ...and `recordings/` is left exactly as the recorder left it. Live never
    // moves, renames or deletes there: the offline pipeline decides staleness
    // by what that folder holds.
    expect(readdirSync(path.join(meetingDir, "recordings")).sort()).toEqual([first, second].sort());

    await card.getByRole("button", { name: "Stop" }).click();
    await expect(card.getByRole("button", { name: "Start from now" })).toBeEnabled();
    await expect(card.getByRole("button", { name: "Resume (catch up)" })).toBeEnabled();
    await expect(pill).toContainText("stopped");

    // Missed while stopped: catching up is what makes a model change, or a
    // restart, cost nothing but the pause itself.
    const third = writeClip(meetingDir, "mic");
    await card.getByRole("button", { name: "Resume (catch up)" }).click();
    await expect(log.locator("div")).toHaveCount(3, { timeout: 15_000 });
    // The two earlier lines are re-read from the file, which carries no clock,
    // so only the new one is stamped.
    expect((await log.locator("div").allTextContents())[2]).toMatch(liveLine("mic"));
    expect(readdirSync(path.join(meetingDir, "live", "clips"))).toContain(third);

    await card.getByRole("button", { name: "Stop" }).click();
    await expect(card.getByRole("button", { name: "Resume (catch up)" })).toBeEnabled();

    // The same lines, read back from live-transcript.txt by the tab strip.
    await page.locator(".tabs").getByRole("button", { name: "Live" }).click();
    const panel = page.locator(".transcript");
    await expect(panel.locator(".line")).toHaveCount(3);
    expect((await panel.locator(".who").allTextContents()).map((item) => item.trim())).toEqual([
      "mic",
      "system",
      "mic"
    ]);
    await expect(panel).toContainText("simulated transcript of system clip");
  });
});
