#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/mds}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/data/hf_cache}"
DATASETS="${DATASETS:-all}"

echo "============================================================"
echo "[DOWNLOAD RAW DATASETS]"
echo "ROOT_DIR=$ROOT_DIR"
echo "HF_CACHE_DIR=$HF_CACHE_DIR"
echo "DATASETS=$DATASETS"
echo "============================================================"

mkdir -p "$ROOT_DIR"
mkdir -p "$HF_CACHE_DIR"

uv run python -m src.utils.download_raw_datasets \
  --root "$ROOT_DIR" \
  --hf-cache-dir "$HF_CACHE_DIR" \
  --datasets $DATASETS

echo "============================================================"
echo "[DONE] Raw dataset download script finished."
echo "============================================================"

find "$ROOT_DIR" -maxdepth 3 -type d | sort