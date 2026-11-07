from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from vulgar_en.config import load_config
from vulgar_en.infer import generate_json_batch, load_detector
from vulgar_en.metrics import full_report
from vulgar_en.nvidia import configure_nvidia_runtime
from vulgar_en.schema import DetectionLabel, label_from_dict, parse_model_json

CLEAN = DetectionLabel(False, "none", ())


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_or_clean(raw: str) -> tuple[DetectionLabel, bool]:
    """Unparseable output counts as a 'clean' prediction (and as a parse failure)."""
    try:
        return parse_model_json(raw), True
    except (ValueError, json.JSONDecodeError):
        return CLEAN, False


def _latency(tokenizer, model, texts: list[str], config, repeats: int) -> dict:
    for text in texts[:3]:
        generate_json_batch(tokenizer, model, [text], config)
    timings = []
    for text in texts[:repeats]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        generate_json_batch(tokenizer, model, [text], config)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    return {
        "batch_size": 1,
        "requests": len(timings),
        "p50_ms": timings[len(timings) // 2],
        "p95_ms": timings[min(len(timings) - 1, int(len(timings) * 0.95))],
    }


def evaluate(
    config_path: str,
    adapter_dir: str | None,
    split_file: str | None,
    use_adapter: bool = True,
    limit: int | None = None,
    batch_size: int = 32,
    model_name: str | None = None,
    latency_requests: int = 50,
    merge: bool = False,
) -> dict:
    config = load_config(config_path)
    if model_name:
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, name=model_name))
    path = Path(split_file or config.data.test_file)
    rows = load_rows(path)
    if limit is not None:
        rows = rows[: max(0, limit)]
    adapter = adapter_dir or config.infer.adapter_dir
    tokenizer, model = load_detector(config, adapter, use_adapter=use_adapter, merge=merge)
    torch.cuda.reset_peak_memory_stats()
    texts = [row["input_text"] for row in rows]
    # length-sorted batching keeps padding small; results are restored to file order
    order = sorted(range(len(rows)), key=lambda i: len(texts[i]))
    raw_outputs: list[str] = [""] * len(rows)
    generate_json_batch(tokenizer, model, texts[:batch_size], config)  # warm-up
    torch.cuda.synchronize()
    started = time.perf_counter()
    for offset in range(0, len(order), batch_size):
        chunk = order[offset : offset + batch_size]
        for index, raw in zip(chunk, generate_json_batch(tokenizer, model, [texts[i] for i in chunk], config)):
            raw_outputs[index] = raw
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    gold = [label_from_dict(row["label"]) for row in rows]
    parsed = [parse_or_clean(raw) for raw in raw_outputs]
    pred = [item[0] for item in parsed]
    sources = [row.get("source", "unknown") for row in rows]
    report = full_report(gold, pred, sources)
    report["split"] = str(path)
    report["model"] = config.model.name
    report["adapter"] = (adapter_dir or config.infer.adapter_dir) if use_adapter else None
    report["weights"] = ("bf16, LoRA merged" if use_adapter else "bf16") if merge or not config.model.use_qlora else "nf4"
    report["parse_failures"] = sum(1 for _, ok in parsed if not ok)
    report["json_valid_rate"] = 1 - report["parse_failures"] / max(len(rows), 1)
    report["throughput"] = {
        "batch_size": batch_size,
        "examples": len(rows),
        "seconds": elapsed,
        "examples_per_second": len(rows) / elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu": torch.cuda.get_device_name(0),
    }
    if latency_requests > 0:
        report["latency"] = _latency(tokenizer, model, texts, config, latency_requests)
    report["predictions"] = [
        {
            "input_text": row["input_text"],
            "source": source,
            "gold": json.loads(g.to_json()),
            "pred": json.loads(p.to_json()),
            "raw": raw,
        }
        for row, source, g, p, raw in zip(rows, sources, gold, pred, raw_outputs)
    ]
    return report


def summary_view(report: dict) -> dict:
    keep = ("examples", "is_vulgar", "severity_macro_f1", "severity_accuracy", "vulgar_terms",
            "ci95", "weights", "parse_failures", "json_valid_rate", "throughput", "latency")
    return {key: report[key] for key in keep if key in report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vulgar-language detector")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, help="override model.name")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-requests", type=int, default=50)
    parser.add_argument("--merge", action="store_true", help="bf16 base with the LoRA merged in, no 4-bit")
    parser.add_argument("--out", default="outputs/eval_report.json")
    args = parser.parse_args()
    configure_nvidia_runtime()
    report = evaluate(
        args.config,
        args.adapter,
        args.split,
        use_adapter=not args.no_adapter,
        limit=args.limit,
        batch_size=args.batch_size,
        model_name=args.model,
        latency_requests=args.latency_requests,
        merge=args.merge,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary_view(report), indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
