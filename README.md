# English Vulgar-Language Detector: QLoRA Fine-Tuning of Qwen2.5-1.5B

A yes/no toxicity score does not tell a moderator how serious a message is or which words caused the flag. This project studies whether a small open LLM can provide all three in one structured answer: whether an English text is vulgar, how severe it is, and which terms are responsible. It fine-tunes a 1.5B-parameter model with QLoRA on a single consumer GPU, using public social-media datasets, and compares the result with zero-shot prompting and classical baselines. The work covers the full path from data to serving: correcting label errors in the source data, building a leakage-safe split, evaluating over multiple seeds, and benchmarking high-throughput inference with vLLM.

QLoRA fine-tuning of [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) that turns an English text into a structured JSON verdict:

```json
{"is_vulgar": true, "severity": "explicit", "vulgar_terms": ["fuck"]}
```

`severity` is one of `none` | `mild` | `strong` | `explicit`.

## Results

All numbers are on the full held-out test split (**3,743 examples**, never seen in training or model selection), evaluated with greedy decoding. Fine-tuned rows are mean ± std over 2 training seeds.

| Model (test, n = 3,743) | is_vulgar acc. | is_vulgar F1 | Severity macro-F1 | Vulgar-terms F1 | Valid JSON |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lexicon (2,630 train terms) | 46.4% | 0.629 | 0.148 | 0.248 | n/a |
| TF-IDF + logistic regression | 76.5% | 0.749 | 0.544 | 0.342 | n/a |
| Qwen2.5-1.5B-Instruct, zero-shot | 57.6% | 0.670 | 0.218 | 0.247 | 99.25% |
| **Qwen2.5-1.5B + QLoRA (2 seeds)** | **80.0 ± 0.2%** | **0.777 ± 0.014** | **0.575 ± 0.013** | **0.499 ± 0.014** | **99.97%** |

Compared with the same model zero-shot, fine-tuning gives:

* **+22.4 pp** accuracy and **+10.6 pp** F1 on `is_vulgar`;
* **2.6×** severity macro-F1;
* **2.0×** vulgar-term F1;
* invalid JSON down from 28 outputs to 1.

The TF-IDF classifier is a strong baseline on the binary decision. Fine-tuning beats it by +2.8 pp F1 on `is_vulgar`, which is close to the bootstrap uncertainty: TF-IDF 95% CI [0.733, 0.764], seed 42 [0.750, 0.782], seed 43 [0.770, 0.801].

The clear gains are in span extraction (+15.7 pp term F1) and in returning everything in one well-formed JSON object.

Raw outputs, including every prediction, are stored under [`results/`](results/):

* aggregate table: [`results/summary.json`](results/summary.json);
* fine-tuned runs: `finetuned_seed*.json`;
* zero-shot run: `base_zero_shot.json`;
* non-LLM baselines: `baselines.json`.

Training and evaluation logs are in [`logs/`](logs/).

### Per seed

| Seed | is_vulgar P / R / F1 | 95% CI (F1) | Severity macro-F1 | Severity acc. | Terms P / R / F1 | Best epoch (eval loss) |
| --- | --- | --- | ---: | ---: | --- | --- |
| 42 | 0.808 / 0.730 / 0.767 | [0.750, 0.782] | 0.566 | 74.9% | 0.648 / 0.393 / 0.489 | 2 of 3 (0.0567) |
| 43 | 0.769 / 0.804 / 0.786 | [0.770, 0.801] | 0.583 | 74.8% | 0.607 / 0.437 / 0.508 | 2 of 3 (0.0565) |

### By source (seed 42 → seed 43; zero-shot in brackets)

| Source (test rows) | is_vulgar F1 | Accuracy | Terms F1 |
| --- | --- | --- | --- |
| HateXplain (1,869) | 0.818 / 0.832 (0.774) | 79.1% / 79.8% (65.4%) | 0.477 / 0.496 (0.306) |
| Toxic Spans (1,038) | 0.665 / 0.703 (0.685) | 64.9% / 64.7% (54.9%) | 0.583 / 0.602 (0.179) |
| TweetEval, non-offensive only (836) | n/a (all clean) | **100% / 100%** (43.5%) | n/a |

The zero-shot model flags 56.5% of clean tweets as vulgar. The fine-tuned model flags none of the 836.

### Serving: from 4 to 294 examples per second

Measured with [`scripts/benchmark.sh`](scripts/benchmark.sh) on one otherwise idle RTX 4090 over the full test split. Latency is one request at a time, p50/p95 over 50 requests after warm-up.

| Serving setup (seed 42) | Throughput | Latency p50 / p95 | is_vulgar F1 | Severity macro-F1 | Terms F1 | Valid JSON |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NF4 base + LoRA, HF `generate`, batch 32 (evaluation setting) | 4.4 ex/s | 2,183 / 3,424 ms | 0.767 | 0.566 | 0.489 | 99.97% |
| LoRA merged into bf16, HF `generate`, batch 64 | 20.2 ex/s | 813 / 1,581 ms | 0.770 | 0.539 | 0.513 | 99.97% |
| **LoRA merged into bf16, vLLM 0.23** | **294 ex/s** (8.2k gen. tok/s) | **111 / 343 ms** | 0.767 | 0.537 | 0.530 | 99.89% |

Seed 43 behaves the same way: 283 ex/s, p50 114 ms with vLLM; F1 0.794, severity 0.572, terms 0.555. vLLM loads the merged model in 3.0 GB of GPU memory.

* The bf16 merged model is a slightly different function from the NF4 + LoRA model it was trained as. Quality moves by up to about 0.05 per metric: terms F1 goes up by 0.04–0.05 and severity F1 goes down by up to 0.03.
* The headline table above keeps the NF4 evaluation setting, which was fixed before any results were seen.
* Re-running the NF4 evaluation (base and seed 42) on an idle GPU reproduced all 3,743 raw outputs byte for byte. The evaluation is deterministic, and the original runs were only slower, not different.

Raw files: `results/bench_*.json`, `results/vllm_seed*.json`.

---

## What changed relative to the first version

| Issue in v1 | Fix |
| --- | --- |
| **Label bug.** `heegyu/toxic-spans` stores its span/score dict, `position` and `type` columns as Python-literal *strings*. v1 read the dict and type columns as empty, so severity fell through to the default `mild`. It iterated the position string character by character, so its digits were taken as character offsets and the extracted terms were fragments. 4,415 training rows were labelled `mild`; the correct count is 401. | `ast.literal_eval` of these columns (`download_data._literal`) plus unit tests on the real row format (`tests/test_download_data.py`). Results on the buggy labels are kept in [`results/legacy_buggy_labels/`](results/legacy_buggy_labels/). |
| Results reported on a 400-example slice, single run | Full 3,743-example test split, 2 seeds, bootstrap 95% CIs, per-source breakdown |
| No baselines other than the base model | Lexicon and TF-IDF + LR baselines fitted on the train split only (`vulgar_en/baselines.py`) |
| Loss on the whole chat (system prompt + text + answer) | `completion_only_loss`: only the assistant JSON is supervised. A start-up check decodes the supervised tokens of a real batch and aborts if any prompt text is included (see `loss_mask_check` in `results/train_meta_seed*.json`) |
| Exact-text de-duplication only, conflicting labels silently overwritten (last one wins), and hand-written seed-bank examples could land in the test split | De-duplication on normalised text before the split, label-conflicting duplicates dropped (13), split stratified by source × label. The seed bank goes to train only, so the test split contains public data only |
| Last checkpoint used, no model selection | Per-epoch eval; best checkpoint by eval loss |

The v1 README reported is_vulgar F1 0.844 and severity macro-F1 0.741 on a 400-example slice of the buggy labels. Those numbers are not comparable with the table above. Most v1 severity labels for Toxic Spans were the constant `mild`, which made severity much easier to predict.

---

## Method

| | |
| --- | --- |
| Base model | Qwen2.5-1.5B-Instruct (1.54B params) |
| Quantization | NF4, double quantization, bf16 compute |
| LoRA | r = 16, α = 32, dropout 0.05 on `q/k/v/o/gate/up/down_proj`, **18.5M trainable params (1.18%)** |
| Objective | SFT (TRL `SFTTrainer`, prompt–completion format), loss on the JSON answer only |
| Optimisation | paged AdamW 8-bit, lr 2e-4, cosine, 5% warm-up, weight decay 0.01, grad-clip 1.0, batch 16, 3 epochs (5,706 steps), length-grouped batches, gradient checkpointing |
| Model selection | Eval loss after each epoch on the validation split. Epoch 2 was best for both seeds |
| Hardware | 1× RTX 4090 per run (power-capped at 250 W). **Peak VRAM 3.4 GB**. 57–60 min per run |
| Software | torch 2.11.0, transformers 5.12.0, trl 1.17.1, peft 0.21.2, bitsandbytes 0.49.2 (pinned in [`requirements.txt`](requirements.txt)) |

## Data

| Source | Use | Label mapping |
| --- | --- | --- |
| [heegyu/toxic-spans](https://huggingface.co/datasets/heegyu/toxic-spans) | vulgar + clean | spans with score ≥ 0.5 → terms. Severity from the `profane/obscene` / `insult` type scores |
| [HateXplain](https://github.com/punyajoy/HateXplain) | vulgar + clean | majority label: hatespeech → `explicit`, offensive → `strong`, normal → clean. Terms = rationale tokens marked by a majority of annotators |
| [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) `offensive` | clean only | `NOT` tweets as clean negatives |
| `vulgar_en/data_bank.py` | train only | 480 hand-written seed examples |

| Split | Rows | Vulgar | none / mild / strong / explicit |
| --- | ---: | ---: | --- |
| train | 30,419 | 13,951 | 16,468 / 401 / 8,427 / 5,123 |
| eval | 3,742 | 1,701 | 2,041 / 38 / 1,046 / 617 |
| test | 3,743 | 1,701 | 2,042 / 37 / 1,034 / 630 |

Counts: [`results/data_summary.json`](results/data_summary.json), [`results/download_counts.json`](results/download_counts.json).

---

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
SEEDS="42 43" bash scripts/run_experiments.sh    # download -> split -> baselines -> zero-shot -> train/eval per seed -> report
bash scripts/benchmark.sh                        # inference cost; the vLLM step needs `pip install vllm==0.23.0`
pytest -q                                        # 20 CPU unit tests
```

Single steps:

```bash
python -m vulgar_en.download_data
python -m vulgar_en.prepare_data --config configs/runpod.yaml
python -m vulgar_en.baselines --config configs/runpod.yaml --out results/baselines.json
python -m vulgar_en.train --config configs/runpod.yaml --seed 42 --output-dir outputs/vulgar-en-qlora-seed42
python -m vulgar_en.evaluate --config configs/runpod.yaml --adapter outputs/vulgar-en-qlora-seed42 --out results/finetuned_seed42.json
python -m vulgar_en.report --results results
python -m vulgar_en.infer --config configs/runpod.yaml --adapter outputs/vulgar-en-qlora-seed42 --text "What the fuck is wrong with this compiler?"
```

The runs used weights downloaded once to local disk (`--model /path/to/Qwen2.5-1.5B-Instruct`). This is why the `model` field in the result files holds a local path. Adapters (`outputs/`) are not committed.

## Layout

```text
configs/      default.yaml | runpod.yaml (setting used for all results) | full_lora.yaml
vulgar_en/
  download_data.py  source datasets -> unified labels
  prepare_data.py   de-duplication, leakage-safe stratified split
  train.py          QLoRA SFT with completion-only loss
  evaluate.py       batched greedy evaluation, throughput, latency (--merge for bf16 merged weights)
  metrics.py        binary / severity / term metrics, bootstrap CIs, per-source breakdown
  baselines.py      lexicon and TF-IDF + LR baselines
  merge.py          fold the adapter into bf16 weights
  vllm_eval.py      evaluate merged weights served by vLLM
  report.py         seed aggregation -> results/summary.json
scripts/      run_experiments.sh | benchmark.sh
tests/        CPU unit tests
results/  logs/
```

## Limitations

* **The label definition is broader than "vulgar".** HateXplain marks hate speech and offensive posts, and its rationale tokens include non-vulgar words that carry the offence. The task the model learns is therefore closer to "offensive language + offending span" than to profanity only. Term F1 is limited by these noisy rationale labels.
* `mild` is rare: 37 test examples, 2–3 of them recovered. Severity macro-F1 is dominated by the other three classes.
* TweetEval is used only for clean negatives, so performance on its offensive tweets is not measured.
* 2 seeds only, because of limited GPU time. The seed-to-seed spread (0.014 F1) is of the same order as the bootstrap CI half-width.
* English only, social-media text. No adversarial obfuscation set.

## License

Code: Apache-2.0. The base model and source datasets keep their own licenses.
