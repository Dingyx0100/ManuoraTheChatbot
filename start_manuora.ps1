# Run from any directory. Use the page controls to load or unload the model.
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python environment not found: $pythonPath"
}
$qdrantRunning = & docker container inspect --format '{{.State.Running}}' manuora-qdrant
if ($LASTEXITCODE -ne 0) { throw 'Qdrant container not found. Start Docker Desktop and create it first.' }
if ($qdrantRunning -ne 'true') {
    & docker start manuora-qdrant
    if ($LASTEXITCODE -ne 0) { throw 'Could not start Qdrant.' }
}

Push-Location -LiteralPath $projectRoot
try {
    Write-Host 'Manuora: http://127.0.0.1:8000 (press Ctrl+C to stop the web server)'
    & $pythonPath -m uvicorn manuora.api.app:app --app-dir backend/src --host 127.0.0.1 --port 8000
}
finally {
    Pop-Location
}
