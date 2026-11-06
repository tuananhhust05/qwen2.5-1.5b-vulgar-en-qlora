#!/usr/bin/env bash
# Inference-cost benchmark on the test split (run on an otherwise idle GPU, after run_experiments.sh):
#   1. base model and adapter on the 4-bit base (the evaluation setting), HF generate
#   2. adapter merged into bf16 weights, HF generate
#   3. merged bf16 weights served with vLLM
# Writes results/bench_*.json and results/vllm_seed*.json.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/runpod.yaml}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
SEEDS=${SEEDS:-"42 43"}
mkdir -p results logs

python -m vulgar_en.evaluate --config "$CONFIG" --model "$MODEL" --no-adapter \
  --out results/bench_base_nf4.json > logs/bench_base_nf4.log 2>&1
first=${SEEDS%% *}
python -m vulgar_en.evaluate --config "$CONFIG" --model "$MODEL" --adapter "outputs/vulgar-en-qlora-seed$first" \
  --out "results/bench_seed${first}_nf4.json" > "logs/bench_seed${first}_nf4.log" 2>&1
for seed in $SEEDS; do
  python -m vulgar_en.evaluate --config "$CONFIG" --model "$MODEL" --adapter "outputs/vulgar-en-qlora-seed$seed" \
    --merge --batch-size 64 --out "results/bench_seed${seed}_bf16_merged.json" > "logs/bench_seed${seed}_bf16_merged.log" 2>&1
  python -m vulgar_en.merge --config "$CONFIG" --model "$MODEL" --adapter "outputs/vulgar-en-qlora-seed$seed" \
    --out "outputs/vulgar-en-merged-seed$seed" > "logs/merge_seed$seed.log" 2>&1
  python -m vulgar_en.vllm_eval --model "outputs/vulgar-en-merged-seed$seed" \
    --out "results/vllm_seed$seed.json" > "logs/vllm_seed$seed.log" 2>&1
done
echo "benchmark done"
