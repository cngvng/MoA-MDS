#!/usr/bin/env bash
set -euo pipefail

NUM_SAMPLES=1 DATASET=multi_xscience "$(dirname "$0")/run_multi_xscience.sh"
