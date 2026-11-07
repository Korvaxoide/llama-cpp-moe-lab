#!/bin/bash
# One controlled experiment: start server -> bench -> graceful stop -> save logs.
# usage: scripts/exp.sh TAG DEPTHS REPS [LONG_REPS] [--docker "..."] [--image IMG] -- <llama-server args>
set -uo pipefail
LAB="$(cd "$(dirname "$0")/.." && pwd)"
TAG="$1"; DEPTHS="$2"; REPS="$3"; shift 3
LONG_REPS=1
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then LONG_REPS="$1"; shift; fi
RUNOPTS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do RUNOPTS+=("$1"); shift; done
shift
NAME="llopt-exp"
mkdir -p "$LAB/logs/exp" "$LAB/results"
echo "== $TAG :: $*" | tee -a "$LAB/logs/exp/index.txt"
T0=$(date +%s)
if ! "$LAB/scripts/run_server.sh" "$NAME" "${RUNOPTS[@]}" -- "$@" -lv 4; then
  "$LAB/bin/dk" logs "$NAME" > "$LAB/logs/exp/$TAG.log" 2>&1
  echo "   FAILED TO START" | tee -a "$LAB/logs/exp/index.txt"
  exit 1
fi
echo "   load_s=$(( $(date +%s) - T0 ))" | tee -a "$LAB/logs/exp/index.txt"
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$LAB/logs/exp/$TAG.vram_idle.txt"
free -m > "$LAB/logs/exp/$TAG.mem_idle.txt"
python3 "$LAB/bench/bench.py" --tag "$TAG" --depths "$DEPTHS" --reps "$REPS" --long-reps "$LONG_REPS" --gen "${GEN:-256}"
RC=$?
"$LAB/bin/dk" stop -t 60 "$NAME" >/dev/null
"$LAB/bin/dk" logs "$NAME" > "$LAB/logs/exp/$TAG.log" 2>&1
grep -E "llama_moe_cache:|MoE cache size|KV buffer size|compute buffer size|model buffer size|RS buffer" "$LAB/logs/exp/$TAG.log" | grep -v CPU_Mapped | sed 's/^/   /' | tee -a "$LAB/logs/exp/index.txt"
"$LAB/bin/dk" rm "$NAME" >/dev/null
exit $RC
