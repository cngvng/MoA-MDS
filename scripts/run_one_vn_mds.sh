#!/usr/bin/env bash
set -euo pipefail

NUM_SAMPLES=1 DATASET=vn_mds "$(dirname "$0")/run_vn_mds.sh"
