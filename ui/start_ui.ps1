# Starts the local Flask UI on http://127.0.0.1:7860.
# Requires the SSH tunnel to the Grafilabs GPU to already be open
# (see remote_inference\start_tunnel.ps1).

param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 7860
)

if (-not $env:BADMINTON_INFER_URL) {
    $env:BADMINTON_INFER_URL = "http://127.0.0.1:8000"
}

$env:UI_HOST = $BindHost
$env:UI_PORT = $Port

Write-Host "Inference target: $env:BADMINTON_INFER_URL"
Write-Host "Open http://${BindHost}:${Port} in your browser."

python "$PSScriptRoot\app.py"
