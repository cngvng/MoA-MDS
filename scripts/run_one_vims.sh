#!/usr/bin/env bash
set -euo pipefail

NUM_SAMPLES=1 DATASET=vims "$(dirname "$0")/run_vims.sh"
