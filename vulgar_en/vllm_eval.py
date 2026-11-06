"""Evaluate a merged checkpoint (python -m vulgar_en.merge) served with vLLM.

Same prompt, greedy decoding and metrics as vulgar_en.evaluate; adds serving throughput and
single-request latency.

    python -m vulgar_en.vllm_eval --model outputs/vulgar-en-merged-seed42 --out results/vllm_seed42.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vulgar_en.metrics import full_report
from vulgar_en.prompts import build_messages
from vulgar_en.schema import DetectionLabel, label_from_dict, parse_model_json

CLEAN = DetectionLabel(False, "none", ())


def parse_or_clean(raw: str) -> tuple[DetectionLabel, bool]:
    """Unparseable output counts as a 'clean' prediction (and as a parse failure), as in evaluate.py."""
    try:
        return parse_model_json(raw), True
    except (ValueError, json.JSONDecodeError):
        return CLEAN, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="merged model directory")
    parser.add_argument("--split", default="data/processed/test.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--latency-requests", type=int, default=50)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in Path(args.split).read_text(encoding="utf-8").splitlines() if line.strip()]
    llm = LLM(model=args.model, dtype="bfloat16", seed=0, gpu_memory_utilization=args.gpu_memory_utilization)
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(build_messages(row["input_text"]), tokenize=False, add_generation_prompt=True)
        for row in rows
    ]
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    llm.generate(prompts[:32], params, use_tqdm=False)  # warm-up
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - started
    raw_outputs = [out.outputs[0].text.strip() for out in outputs]
    generated_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)

    timings = []
    for prompt in prompts[: args.latency_requests]:
        start = time.perf_counter()
        llm.generate([prompt], params, use_tqdm=False)
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()

    gold = [label_from_dict(row["label"]) for row in rows]
    parsed = [parse_or_clean(raw) for raw in raw_outputs]
    sources = [row.get("source", "unknown") for row in rows]
    report = full_report(gold, [item[0] for item in parsed], sources)
    report["split"] = args.split
    report["model"] = args.model
    report["weights"] = "bf16, LoRA merged, vLLM"
    report["parse_failures"] = sum(1 for _, ok in parsed if not ok)
    report["json_valid_rate"] = 1 - report["parse_failures"] / max(len(rows), 1)
    report["throughput"] = {
        "mode": "offline batch (vLLM continuous batching)",
        "examples": len(rows),
        "seconds": elapsed,
        "examples_per_second": len(rows) / elapsed,
        "generated_tokens_per_second": generated_tokens / elapsed,
        "gpu": torch.cuda.get_device_name(0),
    }
    if timings:
        report["latency"] = {
            "batch_size": 1,
            "requests": len(timings),
            "p50_ms": timings[len(timings) // 2],
            "p95_ms": timings[min(len(timings) - 1, int(len(timings) * 0.95))],
        }
    report["predictions"] = [
        {"input_text": row["input_text"], "source": source, "gold": json.loads(g.to_json()),
         "pred": json.loads(p.to_json()), "raw": raw}
        for row, source, g, (p, _), raw in zip(rows, sources, gold, parsed, raw_outputs)
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    view = {k: report[k] for k in ("is_vulgar", "severity_macro_f1", "vulgar_terms", "json_valid_rate",
                                   "throughput", "latency") if k in report}
    print(json.dumps(view, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
