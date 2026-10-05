from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from vulgar_en.config import load_config
from vulgar_en.nvidia import configure_nvidia_runtime, probe_nvidia, resolve_attention, torch_dtype


def merge_adapter(config_path: str, adapter_dir: str | None, output_dir: str) -> None:
    configure_nvidia_runtime()
    config = load_config(config_path)
    runtime = probe_nvidia()
    dtype = torch_dtype(runtime) if runtime.cuda_available else torch.float32
    attention = resolve_attention(config.model.attn_implementation)
    source = adapter_dir or config.infer.adapter_dir
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attention,
        device_map="auto" if runtime.cuda_available else None,
    )
    model = PeftModel.from_pretrained(base, source)
    merged = model.merge_and_unload()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    print(f"Merged model saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base weights")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", default="outputs/vulgar-en-merged")
    args = parser.parse_args()
    merge_adapter(args.config, args.adapter, args.out)


if __name__ == "__main__":
    main()
