# Opens an SSH local-port-forward to the GraphLabs/Grafilabs GPU host so that
# http://localhost:8000 on this machine reaches the inference server running
# inside the GPU instance.
#
# Usage:
#   $env:GRAFI_HOST = "user@gpu.grafilabs.example"
#   $env:GRAFI_PORT = "22"            # optional, defaults to 22
#   $env:GRAFI_KEY  = "$HOME\.ssh\grafilabs_id_ed25519"   # optional
#   .\start_tunnel.ps1
#
# Docker case: the UI container reaches the host as host.docker.internal,
# which lands on a non-loopback interface, so a tunnel bound to 127.0.0.1
# is invisible to it. Pass -Bind '*' (or set $env:GRAFI_TUNNEL_BIND='*') to
# bind all interfaces — only safe on a trusted network because anyone who
# can reach this machine on :8000 can then POST to your remote GPU.

param(
    [int]$LocalPort = 8000,
    [int]$RemotePort = 8000,
    [string]$Bind = ""
)

$Remote = $env:GRAFI_HOST
if (-not $Remote) {
    Write-Error "Set `$env:GRAFI_HOST to user@host of your Grafilabs GPU instance."
    exit 1
}

$SshPort = if ($env:GRAFI_PORT) { $env:GRAFI_PORT } else { "22" }

if (-not $Bind) {
    $Bind = if ($env:GRAFI_TUNNEL_BIND) { $env:GRAFI_TUNNEL_BIND } else { "127.0.0.1" }
}

# -L accepts an optional bind_address prefix. "127.0.0.1" is implicit when
# omitted; anything else requires GatewayPorts on the client side.
$ForwardSpec = "$($Bind):$($LocalPort):127.0.0.1:$RemotePort"

$Args = @(
    "-N",
    "-L", $ForwardSpec,
    "-p", $SshPort
)

if ($Bind -ne "127.0.0.1" -and $Bind -ne "localhost") {
    $Args += @("-o", "GatewayPorts=yes")
}

if ($env:GRAFI_KEY) {
    $Args += @("-i", $env:GRAFI_KEY)
}

$Args += @(
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-o", "ExitOnForwardFailure=yes",
    $Remote
)

Write-Host "Opening SSH tunnel ${Bind}:$LocalPort -> ${Remote}:$RemotePort"
& ssh @Args
