# English vulgar-language detector — QLoRA fine-tuning

Fine-tune [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) with QLoRA to detect English profanity, score severity, and list vulgar terms. CUDA-first stack: PyTorch, Transformers, PEFT, TRL, bitsandbytes.

## Output schema

```json
{
  "is_vulgar": true,
  "severity": "explicit",
  "vulgar_terms": ["fuck"]
}
```

`severity`: `none` | `mild` | `strong` | `explicit`

## Results (RTX 4090, 3 epochs)

On a 400-example held-out slice vs the same base model without the adapter:

| Metric | Base | Fine-tuned |
| --- | --- | --- |
| is_vulgar accuracy | 0.640 | **0.825** |
| is_vulgar F1 | 0.744 | **0.844** |
| severity macro-F1 | 0.170 | **0.741** |
| vulgar_terms F1 | 0.288 | **0.593** |

Adapter for Hugging Face upload: build under `hf_upload/vulgar-en-qlora/` after training (ignored by git).

## Setup (NVIDIA GPU)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
python -m vulgar_en.env_check
```

## Train

```bash
python -m vulgar_en.download_data
python -m vulgar_en.prepare_data --config configs/default.yaml
python -m vulgar_en.train --config configs/default.yaml
```

RTX 4090 preset: `configs/runpod.yaml` (batch 8, same QLoRA 1.5B).

```bash
bash scripts/train.sh
```

## Infer / evaluate / compare

```bash
python -m vulgar_en.infer --text "What the fuck is wrong with this compiler?"
python -m vulgar_en.evaluate --config configs/default.yaml
python -m vulgar_en.compare --config configs/default.yaml --limit 400
```

`--no-adapter` evaluates the base model only.

## Merge (optional)

```bash
python -m vulgar_en.merge --out outputs/vulgar-en-merged
```

## Docker (NGC PyTorch)

```bash
docker build -t vulgar-en-finetune .
docker run --gpus all --rm -it \
  -v "$PWD/outputs:/workspace/llm-finetune/outputs" \
  -v "$HOME/.cache/huggingface:/workspace/hf_cache" \
  vulgar-en-finetune
```

## Layout

```
configs/      default.yaml | runpod.yaml | full_lora.yaml
vulgar_en/    data, train, infer, evaluate, compare
scripts/      train / infer / evaluate helpers
tests/        unit tests (no GPU)
```

## Data

Downloaded at train time (not committed):

- [heegyu/toxic-spans](https://huggingface.co/datasets/heegyu/toxic-spans)
- [HateXplain](https://github.com/punyajoy/HateXplain)
- [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (`offensive`)
- seed bank in `vulgar_en/data_bank.py`

## Tests

```bash
pip install pytest pyyaml scikit-learn
pytest -q
```

## License

Code: Apache-2.0. Base model and source datasets follow their own licenses.
