# Run HeatWatch AI Platform
$root = $PSScriptRoot
Set-Location $root
$python = if (Test-Path "$root\.venv\Scripts\python.exe") { "$root\.venv\Scripts\python.exe" } else { "python" }

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "Starting HeatWatch AI Platform" -ForegroundColor Green
Write-Host "Opening http://127.0.0.1:8000 ..." -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

Start-Process "http://127.0.0.1:8000"
& $python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
