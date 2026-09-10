// Drives the demo dashboard through two flows and captures frames for the
// README GIFs plus a few still screenshots. Nothing here touches a real
// workspace: it talks to demo_server.py, whose data is invented.
//
//   node e2e/demo/record.mjs <out-dir>
//
// Frames land in <out-dir>/frames/<flow>/NNNN.png with a <flow>.json listing
// how long each should be shown; make_gif.py turns them into a GIF.

import fs from "node:fs";
import path from "node:path";

import { chromium } from "@playwright/test";

const base = process.env.DEMO_URL || "http://127.0.0.1:8124";
const outDir = path.resolve(process.argv[2] || "demo-out");
const shotsDir = path.join(outDir, "shots");
fs.mkdirSync(shotsDir, { recursive: true });

// A visible pointer, since screenshots do not include the OS cursor.
const CURSOR = `(() => {
  const style = document.createElement('style');
  style.textContent = '#demo-cursor{position:fixed;left:0;top:0;pointer-events:none;z-index:2147483647;transform:translate(-3px,-2px)}' +
    '#demo-cursor svg{display:block;filter:drop-shadow(0 1px 2px rgba(0,0,0,.5))}' +
    '#demo-ring{position:fixed;width:36px;height:36px;border:2px solid #5b8cff;border-radius:50%;pointer-events:none;z-index:2147483646;' +
    'transform:translate(-50%,-50%) scale(.3);opacity:0;transition:transform .4s ease-out,opacity .4s ease-out}' +
    '#demo-ring.on{transform:translate(-50%,-50%) scale(1);opacity:1;transition:none}';
  const cursor = document.createElement('div');
  cursor.id = 'demo-cursor';
  cursor.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24"><path d="M5 3l14 8-6.2 1.6L9.6 19.5z" fill="#fff" stroke="#111" stroke-width="1.6" stroke-linejoin="round"/></svg>';
  const ring = document.createElement('div');
  ring.id = 'demo-ring';
  const attach = () => { document.head.appendChild(style); document.body.appendChild(cursor); document.body.appendChild(ring); };
  if (document.body) attach(); else document.addEventListener('DOMContentLoaded', attach);
  const move = (e) => { for (const el of [cursor, ring]) { el.style.left = e.clientX + 'px'; el.style.top = e.clientY + 'px'; } };
  window.addEventListener('mousemove', move, true);
  window.addEventListener('mousedown', (e) => { move(e); ring.classList.add('on'); requestAnimationFrame(() => requestAnimationFrame(() => ring.classList.remove('on'))); }, true);
})();`;

class Flow {
  constructor(page, name) {
    this.page = page;
    this.name = name;
    this.frames = [];
    this.dir = path.join(outDir, "frames", name);
    fs.rmSync(this.dir, { recursive: true, force: true });
    fs.mkdirSync(this.dir, { recursive: true });
  }

  async frame(holdMs = 600) {
    const file = `${String(this.frames.length).padStart(4, "0")}.png`;
    await this.page.screenshot({ path: path.join(this.dir, file) });
    this.frames.push({ file, duration: holdMs });
  }

  hold(ms) {
    if (this.frames.length) this.frames[this.frames.length - 1].duration += ms;
  }

  async click(locator, holdMs = 700) {
    await locator.hover();
    await this.frame(400);
    await locator.click();
    await this.page.waitForTimeout(200);
    await this.frame(holdMs);
  }

  async type(locator, text, holdMs = 500) {
    await locator.click();
    // Replace, never append: a field that already holds a default would
    // otherwise end up as "hostCarol".
    await locator.fill("");
    await this.frame(250);
    for (let i = 0; i < text.length; i += 3) {
      await locator.pressSequentially(text.slice(i, i + 3), { delay: 15 });
      await this.frame(110);
    }
    this.hold(holdMs);
  }

  async select(locator, value) {
    await locator.hover();
    await this.frame(300);
    await locator.selectOption(value);
    await this.frame(600);
  }

  // Keep taking frames at real-time pace until `done` says so.
  async watch(done, { every = 700, timeout = 40000 } = {}) {
    const started = Date.now();
    while (Date.now() - started < timeout) {
      await this.frame(every);
      if (await done()) return;
      await this.page.waitForTimeout(every);
    }
    throw new Error(`${this.name}: timed out waiting`);
  }

  save() {
    fs.writeFileSync(path.join(outDir, "frames", `${this.name}.json`), JSON.stringify(this.frames));
    console.log(`${this.name}: ${this.frames.length} frames`);
  }
}

// A step is ticked as soon as its first output file exists, which is before
// the job finishes; wait for the job card to go away as well.
const stepsDone = (page, n) => async () =>
  (await page.locator(".step-number.done").count()) >= n &&
  (await page.locator(".pill", { hasText: "running" }).count()) === 0;

async function shot(page, name, height) {
  const before = page.viewportSize();
  if (height) await page.setViewportSize({ width: before.width, height });
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(shotsDir, `${name}.png`) });
  if (height) await page.setViewportSize(before);
}

async function meetFlow(page) {
  const flow = new Flow(page, "meet");
  await page.goto(base);
  await page.waitForSelector(".sidebar h1");
  await page.mouse.move(640, 400);
  await flow.frame(1300);

  // Project.
  await flow.type(page.getByPlaceholder("New project…"), "Product team");
  await flow.click(page.locator(".new-row", { hasText: "Add" }).last().getByRole("button"), 900);

  // Settings: devices, language, engine, name.
  await flow.click(page.locator(".icon-button[title=Settings]"), 900);
  const modal = page.locator(".modal");
  const field = (label) => modal.locator(".field", { has: page.locator(`label:text-is("${label}")`) });
  await flow.select(field("Microphone").locator("select"), "Headset Microphone (USB Audio Device)");
  await flow.select(field("PC audio").locator("select"), "Headset Earphone (USB Audio Device)");
  await flow.select(field("Language").locator("select"), "en");
  await flow.type(field("Call me").locator("input"), "Carol", 300);
  await flow.select(field("Model").nth(1).locator("select"), "pyannote");
  await flow.type(modal.locator("textarea").first(), "AviListener\nWhisper", 300);
  await shot(page, "settings", 1280);
  await flow.click(modal.getByRole("button", { name: "Save settings" }), 900);

  // Meeting.
  await flow.type(page.getByPlaceholder("New meeting…"), "Weekly sync");
  await flow.click(page.locator(".project .new-row").getByRole("button"), 1200);

  // Record: microphone and PC audio are on by default.
  await page.locator(".toggle", { hasText: "PC audio" }).hover();
  await flow.frame(700);
  await flow.click(page.getByRole("button", { name: "Start recording" }), 500);
  await page.waitForTimeout(4000);
  await shot(page, "recording");
  await flow.watch(async () => /recording (?:[6-9]|\d\d)s/.test(await page.locator(".pill.live").innerText()), { every: 800 });
  await flow.click(page.getByRole("button", { name: "Stop" }), 1400);

  // Transcribe.
  await flow.click(page.getByRole("button", { name: "Transcribe", exact: true }), 400);
  await flow.watch(stepsDone(page, 2));
  flow.hold(800);

  // Split by speaker.
  await flow.click(page.getByRole("button", { name: "Split by speaker" }), 400);
  await flow.watch(stepsDone(page, 3));
  flow.hold(800);

  // Name the speakers.
  const rows = page.locator(".speaker-row");
  await rows.first().scrollIntoViewIfNeeded();
  await flow.frame(900);
  await flow.type(rows.nth(0).locator("input"), "Alice", 200);
  await flow.type(rows.nth(1).locator("input"), "Bob", 200);
  await flow.click(page.getByRole("button", { name: "Save names" }), 900);

  // Merge.
  await flow.click(page.getByRole("button", { name: "Merge", exact: true }), 400);
  await flow.watch(stepsDone(page, 4));
  // The view stays on whichever tab was open (Logs, before there was a
  // transcript), so switch to the merged one explicitly.
  await flow.click(page.getByRole("button", { name: "Everyone (merged)" }), 600);
  await page.locator(".transcript").scrollIntoViewIfNeeded();
  await page.mouse.move(900, 700);
  await flow.frame(3500);
  flow.save();
  await shot(page, "merged", 1500);
}

async function discordFlow(page) {
  const flow = new Flow(page, "discord");
  await page.mouse.move(640, 400);
  await flow.type(page.getByPlaceholder("New project…"), "Tuesday Raiders");
  await flow.click(page.locator(".new-row", { hasText: "Add" }).last().getByRole("button"), 900);
  const project = page.locator(".project", { hasText: "Tuesday Raiders" });
  await flow.type(project.getByPlaceholder("New meeting…"), "Raid night");
  await flow.click(project.locator(".new-row").getByRole("button"), 1200);

  // Sources: Discord alone. The bot hears everyone in the channel, the host
  // included, and names each of them, so no local source is needed.
  await flow.click(page.locator(".toggle", { hasText: "My microphone" }), 400);
  await flow.click(page.locator(".toggle", { hasText: "PC audio" }), 500);
  await flow.click(page.locator(".toggle", { hasText: "Discord" }), 900);
  await flow.select(page.locator("select").filter({ hasText: "Ask in chat" }), "200000000000000001");
  await flow.click(page.getByRole("button", { name: "Start recording" }), 500);
  await flow.watch(async () => (await page.locator(".log").first().innerText()).includes("erin is speaking"), { every: 900 });
  await page.waitForTimeout(500);
  await shot(page, "discord-recording");
  await flow.click(page.getByRole("button", { name: "Stop" }), 1400);

  await flow.click(page.getByRole("button", { name: "Transcribe", exact: true }), 400);
  await flow.watch(stepsDone(page, 2), { every: 700 });
  await page.getByRole("button", { name: "By source" }).click();
  await page.locator(".transcript").scrollIntoViewIfNeeded();
  await page.mouse.move(900, 700);
  await flow.frame(3500);
  flow.save();
  await shot(page, "discord", 1300);
}

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 });
await context.addInitScript(CURSOR);
const page = await context.newPage();
try {
  await meetFlow(page);
  await discordFlow(page);
} finally {
  await browser.close();
}
