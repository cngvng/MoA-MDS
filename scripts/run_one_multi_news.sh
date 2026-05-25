#!/usr/bin/env bash
set -euo pipefail

NUM_SAMPLES=1 DATASET=multi_news "$(dirname "$0")/run_multi_news.sh"
