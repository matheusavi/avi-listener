// Where the test dashboard lives, resolved once so the global setup and the
// server it launches cannot disagree about it.
//
// The workspace is a throwaway directory in the system temp folder, never the
// checkout's own `workspace/`: these tests create and rename things, and doing
// that to the folder holding real meetings would be unforgivable.
//
// Both the directory and the port are fixed rather than per-run, so only one
// suite can run on a machine at a time. Two concurrent runs would fight over
// the port regardless, and a fixed directory is one that cleans itself up on
// the next run instead of accumulating.

import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

export const projectRoot = path.resolve(here, "..");
export const workspaceDir = path.join(tmpdir(), "avilistener-e2e-workspace");
// Not 8000: a run must not collide with, or quietly talk to, a dashboard the
// developer already has open on the default port.
export const port = 8123;
export const baseURL = `http://127.0.0.1:${port}`;

/** The interpreter that has the dashboard installed. */
export function pythonExecutable() {
  if (process.env.AVILISTENER_PYTHON) return process.env.AVILISTENER_PYTHON;
  const venv = path.join(projectRoot, ".venv", "Scripts", "python.exe");
  return existsSync(venv) ? venv : "python";
}
