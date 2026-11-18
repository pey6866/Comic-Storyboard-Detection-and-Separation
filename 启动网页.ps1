$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
Write-Host 'Open http://127.0.0.1:8877 in your browser. Press Ctrl+C to stop.'
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\web_server.py"
