<#
.SYNOPSIS
    One-file setup for AviListener on a fresh Windows machine.

.DESCRIPTION
    Creates the virtual environments, installs everything, and builds the web
    dashboard. Run it from the repository root in PowerShell:

        .\setup.ps1              # GPU setup (NVIDIA + CUDA), recommended
        .\setup.ps1 -Cpu         # no NVIDIA GPU
        .\setup.ps1 -WithNemo    # also install the optional NeMo engine

    What it installs:
      .venv           main environment: transcription, dashboard, CLI
      .venv-pyannote  the default speaker-splitting engine (pyannote)
      .venv-nemo      optional alternative engine (-WithNemo; large download)
      web/dist        the built dashboard interface

    After it finishes you still need, depending on what you use:
      - a Hugging Face token for speaker splitting (free; the dashboard's
        Settings explains where to get it and stores it for you)
      - a Discord bot token if you record Discord calls (same: see Settings)
#>
param(
    [switch]$Cpu,
    [switch]$WithNemo
)

$ErrorActionPreference = "Stop"

function Step($message) { Write-Host "`n==> $message" -ForegroundColor Cyan }

if (-not (Test-Path "pyproject.toml")) {
    throw "Run this from the avilistener repository root."
}

# ---- prerequisites ---------------------------------------------------------
Step "Checking prerequisites"
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) { throw "Python launcher 'py' not found. Install Python 3.11 from python.org (tick 'py launcher')." }
try { py -3.11 -c "pass" } catch { throw "Python 3.11 not installed. Install it from python.org; other versions are untested." }
$node = Get-Command npm -ErrorAction SilentlyContinue
if (-not $node) { Write-Warning "npm not found: the web dashboard cannot be built. Install Node.js 22+ and re-run, or use the CLI only." }

# ---- main environment ------------------------------------------------------
Step "Main environment (.venv): transcription, dashboard, CLI"
if (-not (Test-Path ".venv")) { py -3.11 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
.\.venv\Scripts\python.exe -m pip install -e ".[web,test]" --quiet
Write-Host "    ok"

# ---- pyannote environment (default speaker-splitting engine) ---------------
# Separate venv on purpose: torch-heavy dependencies conflict with the rest.
Step "Speaker splitting (.venv-pyannote): pyannote community-1, the default engine"
if (-not (Test-Path ".venv-pyannote")) { py -3.11 -m venv .venv-pyannote }
.\.venv-pyannote\Scripts\python.exe -m pip install --upgrade pip --quiet
if ($Cpu) {
    .\.venv-pyannote\Scripts\python.exe -m pip install torch torchaudio --quiet
} else {
    # cu128 covers RTX 30/40/50 series; older cards may need a different index.
    .\.venv-pyannote\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet
}
.\.venv-pyannote\Scripts\python.exe -m pip install pyannote.audio --quiet
Write-Host "    ok"

# ---- optional NeMo environment ---------------------------------------------
if ($WithNemo) {
    Step "Alternative engine (.venv-nemo): NVIDIA NeMo (this one is big)"
    if (-not (Test-Path ".venv-nemo")) { py -3.11 -m venv .venv-nemo }
    .\.venv-nemo\Scripts\python.exe -m pip install --upgrade pip --quiet
    .\.venv-nemo\Scripts\python.exe -m pip install "nemo_toolkit[asr]" --quiet
    Write-Host "    ok"
}

# ---- web dashboard ---------------------------------------------------------
if ($node) {
    Step "Building the web dashboard"
    Push-Location web
    npm install --silent
    npm run build --silent
    Pop-Location
    Write-Host "    ok"
}

# ---- local config from the example -----------------------------------------
Step "Local configuration file"
if ((Test-Path "config.example.yaml") -and -not (Test-Path "config.yaml")) {
    Copy-Item "config.example.yaml" "config.yaml"
    Write-Host "    created config.yaml from config.example.yaml"
}

# ---- done ------------------------------------------------------------------
Step "Done. Next steps"
Write-Host @"
    1. Start the dashboard:      .\.venv\Scripts\avilistener.exe dashboard
       then open                 http://127.0.0.1:8000

    2. In the dashboard's Settings (gear icon), fine-tune for THIS machine —
       the defaults are a working starting point, not universal truth:
         - pick your real microphone and the output you hear calls through
         - set the transcription language
         - paste a Hugging Face token for speaker splitting (the panel
           explains where to get one; the model download is free)
         - add the names and jargon Whisper misspells in your world
         - no NVIDIA GPU? switch 'Run on' to CPU

    3. Check your microphone level before the first real recording:
         .\.venv\Scripts\avilistener.exe levels --config config.yaml

    See README.md for the full walkthrough.
"@
