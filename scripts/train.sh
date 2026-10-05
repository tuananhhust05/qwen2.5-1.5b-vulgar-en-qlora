#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
python -m vulgar_en.env_check
python -m vulgar_en.download_data
python -m vulgar_en.prepare_data --config configs/default.yaml
python -m vulgar_en.train --config configs/default.yaml
