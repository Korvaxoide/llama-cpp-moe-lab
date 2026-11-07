#!/bin/bash
# Run a list of experiment commands sequentially, one server at a time.
# usage: scripts/queue.sh <file-with-one-bash-command-per-line>
# Empty lines and lines starting with # are ignored. Runs from $LAB.
set -uo pipefail
LAB="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LAB" || exit 1
QUEUE="$1"
while IFS= read -r line; do
  [[ -z "${line// }" || "$line" == \#* ]] && continue
  echo "### QUEUE START $(date +%H:%M:%S) :: $line"
  bash -c "$line"
  rc=$?
  echo "### QUEUE END rc=$rc $(date +%H:%M:%S)"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/### gpu_now /'
  bin/dk ps --format '{{.Names}} {{.Status}}' | sed 's/^/### ps /'
  sleep 5
done < "$QUEUE"
echo "### QUEUE DONE $(date +%H:%M:%S)"
