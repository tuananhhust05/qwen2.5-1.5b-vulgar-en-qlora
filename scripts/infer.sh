#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -m vulgar_en.infer --config configs/default.yaml --text "${1:-What the fuck is this?}"
