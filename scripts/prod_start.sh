#!/bin/bash
# Start the optimized production server on 127.0.0.1:8080.
# Configuration validated in Fase 8 (F1) and refined after the BIOS ReBAR test (F1r):
# see docs/RESULTS.md / docs/OPTIMIZATION.md.
set -euo pipefail
LAB="$(cd "$(dirname "$0")/.." && pwd)"
DK="$LAB/bin/dk"
NAME="qwen36-35b-opt"
# b11977, pinned by digest (the :server-cuda tag moves upstream)
IMAGE="ghcr.io/ggml-org/llama.cpp@sha256:0283744a35a6e7698b4f7a2d63360233dc769ece51e84bdba7aa187a33978276"
PORT=8080

"$DK" rm -f "$NAME" >/dev/null 2>&1 || true
"$DK" run -d --name "$NAME" --gpus all \
  -v "$HOME/AI/llama-models:/models:ro" \
  -p "127.0.0.1:${PORT}:8080" \
  --restart no \
  "$IMAGE" \
  -m /models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf --host 0.0.0.0 --port 8080 \
  --ctx-size 200000 --n-predict -1 --n-gpu-layers auto \
  --n-cpu-moe 37 --moe-cache-mib 0 --flash-attn on --parallel 1 --jinja \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-cpu-moe \
  --spec-draft-n-ubatch 512 \
  -ctk q8_0 -ctv q8_0 -b 3072 -ub 3072 --load-mode none

for i in $(seq 1 600); do
  if curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "qwen36-35b-opt READY after ${i}s on 127.0.0.1:${PORT}"; exit 0
  fi
  if [[ "$("$DK" inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]]; then
    echo "CONTAINER EXITED"; "$DK" logs --tail 40 "$NAME" 2>&1; exit 1
  fi
  sleep 1
done
echo "TIMEOUT"; exit 1
