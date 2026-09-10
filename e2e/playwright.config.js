import { defineConfig } from "@playwright/test";

import { baseURL, port, projectRoot, pythonExecutable, workspaceDir } from "./environment.js";

export default defineConfig({
  testDir: "./tests",
  // One worker, in order: the dashboard has process-wide state (one recording,
  // one job per meeting), so tests racing each other would test the harness
  // rather than the product.
  fullyParallel: false,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  // No retries, on purpose. The workspace is seeded once per run and several
  // tests change it, so a retry replays against state the first attempt
  // already altered - a renamed speaker stays renamed, and the assertion
  // passes for a reason that has nothing to do with the code.
  retries: 0,
  reporter: process.env.CI ? [["github"], ["list"]] : [["list"]],
  timeout: 30_000,
  expect: { timeout: 10_000 },
  globalSetup: "./global-setup.js",
  use: {
    baseURL,
    trace: "retain-on-failure",
  },
  webServer: {
    command: `"${pythonExecutable()}" -m uvicorn avilistener.server.app:app --host 127.0.0.1 --port ${port} --log-level warning`,
    cwd: projectRoot,
    url: `${baseURL}/api/health`,
    reuseExistingServer: false,
    stdout: "pipe",
    stderr: "pipe",
    timeout: 60_000,
    env: {
      AVILISTENER_WORKSPACE: workspaceDir,
      // Blanked rather than inherited: whether the machine running the tests
      // happens to have a Discord or Hugging Face token is not something the
      // assertions should depend on.
      DISCORD_BOT_TOKEN: "",
      HF_TOKEN: "",
      HUGGING_FACE_HUB_TOKEN: "",
    },
  },
});
