#!/bin/bash
# Stop the optimized production server (container kept for a quick restart).
set -euo pipefail
LAB="$(cd "$(dirname "$0")/.." && pwd)"
"$LAB/bin/dk" stop -t 60 qwen36-35b-opt
echo "qwen36-35b-opt stopped (container preserved)."
