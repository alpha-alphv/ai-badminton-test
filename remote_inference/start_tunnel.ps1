# Opens an SSH local-port-forward to the GraphLabs/Grafilabs GPU host so that
# http://127.0.0.1:8000 on this machine reaches the inference server running
# inside the GPU instance.
#
# Usage:
#   $env:GRAFI_HOST = "user@gpu.grafilabs.example"
#   $env:GRAFI_PORT = "22"            # optional, defaults to 22
#   $env:GRAFI_KEY  = "$HOME\.ssh\grafilabs_id_ed25519"   # optional
#   .\start_tunnel.ps1

param(
    [int]$LocalPort = 8000,
    [int]$RemotePort = 8000
)

$Remote = $env:GRAFI_HOST
if (-not $Remote) {
    Write-Error "Set `$env:GRAFI_HOST to user@host of your Grafilabs GPU instance."
    exit 1
}

$SshPort = if ($env:GRAFI_PORT) { $env:GRAFI_PORT } else { "22" }

$Args = @(
    "-N",
    "-L", "$($LocalPort):127.0.0.1:$RemotePort",
    "-p", $SshPort
)

if ($env:GRAFI_KEY) {
    $Args += @("-i", $env:GRAFI_KEY)
}

$Args += @(
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-o", "ExitOnForwardFailure=yes",
    $Remote
)

Write-Host "Opening SSH tunnel localhost:$LocalPort -> ${Remote}:$RemotePort"
& ssh @Args
