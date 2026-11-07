#!/bin/bash
# Start an experimental llama-server container bound to 127.0.0.1 only.
# usage: scripts/run_server.sh NAME [--image IMG] [--port P] [--docker "extra docker args"] -- <llama-server args>
# The model directory is mounted read-only; the GGUF is never modified.
set -euo pipefail
LAB="$(cd "$(dirname "$0")/.." && pwd)"
DK="$LAB/bin/dk"
NAME="$1"; shift
# pinned: b11977 / commit 7ac5727d6 (the tag server-cuda moves upstream)
IMAGE="ghcr.io/ggml-org/llama.cpp@sha256:0283744a35a6e7698b4f7a2d63360233dc769ece51e84bdba7aa187a33978276"
PORT=8081
EXTRA=""
while [[ $# -gt 0 && "$1" != "--" ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --docker) EXTRA="$2"; shift 2;;
    *) echo "unknown option $1"; exit 1;;
  esac
done
[[ "${1:-}" == "--" ]] && shift

"$DK" rm -f "$NAME" >/dev/null 2>&1 || true
# shellcheck disable=SC2086
"$DK" run -d --name "$NAME" --gpus all $EXTRA \
  -v "$HOME/AI/llama-models:/models:ro" \
  -p "127.0.0.1:${PORT}:8080" \
  "$IMAGE" \
  -m /models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --host 0.0.0.0 --port 8080 "$@"

# wait for health
for i in $(seq 1 600); do
  if curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "READY after ${i}s"; exit 0
  fi
  if [[ "$("$DK" inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]]; then
    echo "CONTAINER EXITED"; "$DK" logs --tail 40 "$NAME" 2>&1; exit 1
  fi
  sleep 1
done
echo "TIMEOUT"; exit 1
