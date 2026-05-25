#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-multi_news}"
SPLIT="${SPLIT:-test}"
RESUME="${RESUME:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
LOG_DIR="${LOG_DIR:-logs/main_results}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${DATASET}_${SPLIT}_$(date +%Y%m%d_%H%M%S).log}"

args=(uv run python -m src.run --dataset "$DATASET" --split "$SPLIT")
if [[ -n "${NUM_SAMPLES:-}" ]]; then
  args+=(--num-samples "$NUM_SAMPLES")
fi
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  args+=(--output-root "$OUTPUT_ROOT")
fi
if [[ "$RESUME" == "1" ]]; then
  args+=(--resume)
else
  args+=(--no-resume)
fi
if [[ "$SKIP_EXISTING" == "1" ]]; then
  args+=(--skip-existing)
else
  args+=(--no-skip-existing)
fi
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  args+=(--continue-on-error)
else
  args+=(--no-continue-on-error)
fi

echo "log_file=$LOG_FILE"
"${args[@]}" 2>&1 | tee "$LOG_FILE"
