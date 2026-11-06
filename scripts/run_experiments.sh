#!/usr/bin/env bash
# Reproduces every number in README.md:
#   download -> leakage-safe split -> non-LLM baselines -> zero-shot base model
#   -> QLoRA x N seeds (round-robin over $GPUS) -> full test-set evaluation -> report
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/runpod.yaml}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
SEEDS=${SEEDS:-"42 43 44"}
GPUS=(${GPUS:-0 1})
mkdir -p results logs outputs

[ -f data/raw/labeled.jsonl ] || python -m vulgar_en.download_data | tee results/download_counts.json
python -m vulgar_en.prepare_data --config "$CONFIG" > logs/prepare.log
cp data/processed/summary.json results/data_summary.json
python -m vulgar_en.baselines --config "$CONFIG" --out results/baselines.json

CUDA_VISIBLE_DEVICES=${GPUS[0]} python -m vulgar_en.evaluate --config "$CONFIG" --model "$MODEL" \
  --no-adapter --out results/base_zero_shot.json > logs/eval_base.log 2>&1

run_seed() {
  local seed=$1 gpu=$2 out=outputs/vulgar-en-qlora-seed$1
  CUDA_VISIBLE_DEVICES=$gpu python -m vulgar_en.train --config "$CONFIG" --model "$MODEL" \
    --seed "$seed" --output-dir "$out" > "logs/train_seed$seed.log" 2>&1
  CUDA_VISIBLE_DEVICES=$gpu python -m vulgar_en.evaluate --config "$CONFIG" --model "$MODEL" \
    --adapter "$out" --out "results/finetuned_seed$seed.json" > "logs/eval_seed$seed.log" 2>&1
  python - "$out/train_meta.json" "results/train_meta_seed$seed.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
meta.pop("log_history", None)
json.dump(meta, open(sys.argv[2], "w"), indent=2)
PY
}
index=0
for seed in $SEEDS; do
  run_seed "$seed" "${GPUS[$((index % ${#GPUS[@]}))]}" &
  index=$((index + 1))
  if (( index % ${#GPUS[@]} == 0 )); then wait; fi
done
wait

python -m vulgar_en.report --results results
