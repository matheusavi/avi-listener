import { execFileSync } from "node:child_process";
import path from "node:path";

import { projectRoot, pythonExecutable, workspaceDir } from "./environment.js";

export default function globalSetup() {
  const seeder = path.join(projectRoot, "e2e", "seed_workspace.py");
  execFileSync(pythonExecutable(), [seeder, workspaceDir], { stdio: "inherit" });
}
