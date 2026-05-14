#!/usr/bin/env bash
# Open an SSH local-port-forward to the GraphLabs / Grafilabs GPU host so that
# http://127.0.0.1:8000 on this machine reaches the inference server inside
# the GPU instance.
#
# Usage:
#   export GRAFI_HOST=user@gpu.grafilabs.example
#   export GRAFI_PORT=22                      # optional
#   export GRAFI_KEY=~/.ssh/grafilabs_id_ed25519  # optional
#   ./start_tunnel.sh

set -euo pipefail

LOCAL_PORT=${1:-8000}
REMOTE_PORT=${2:-8000}

: "${GRAFI_HOST:?Set GRAFI_HOST=user@host of your Grafilabs GPU instance}"
GRAFI_PORT=${GRAFI_PORT:-22}

args=(-N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" -p "${GRAFI_PORT}")
if [[ -n "${GRAFI_KEY:-}" ]]; then
  args+=(-i "${GRAFI_KEY}")
fi
args+=(-o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes "${GRAFI_HOST}")

echo "Opening SSH tunnel localhost:${LOCAL_PORT} -> ${GRAFI_HOST}:${REMOTE_PORT}"
exec ssh "${args[@]}"
