# Start a detached dashboard. Closing the terminal or Codex will not stop it.
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    Write-Output "Port 8000 is already in use (PID $($listener[0].OwningProcess)). No second server started."
    exit 0
}
$executable = Join-Path $projectRoot '.venv\Scripts\avilistener.exe'
if (-not (Test-Path -LiteralPath $executable)) { throw 'Run setup.ps1 first: .venv is missing.' }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdout = Join-Path $projectRoot "avilistener-background-$stamp.out.log"
$stderr = Join-Path $projectRoot "avilistener-background-$stamp.err.log"
$server = Start-Process -FilePath $executable -ArgumentList 'dashboard' -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    Start-Sleep -Milliseconds 500
    $server.Refresh()
    if ($server.HasExited) { throw "Dashboard exited. See $stderr" }
    try {
        $health = Invoke-RestMethod 'http://127.0.0.1:8000/api/health' -TimeoutSec 2
        if ($health.ok) {
            Write-Output "Dashboard ready: http://127.0.0.1:8000 (launcher PID $($server.Id))"
            Write-Output "Logs: $stdout and $stderr"
            exit 0
        }
    } catch { }
}
throw "Dashboard not ready yet; inspect $stderr (launcher PID $($server.Id))."
