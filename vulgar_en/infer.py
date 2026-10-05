from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from vulgar_en.config import AppConfig, load_config
from vulgar_en.nvidia import configure_nvidia_runtime, probe_nvidia, resolve_attention, torch_dtype
from vulgar_en.prompts import build_messages
from vulgar_en.schema import DetectionLabel, parse_model_json


def load_tokenizer(model_name: str, adapter_dir: str | None) -> AutoTokenizer:
    source = adapter_dir if adapter_dir and Path(adapter_dir, "tokenizer_config.json").exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_detector(
    config: AppConfig, adapter_dir: str, use_adapter: bool = True
) -> tuple[AutoTokenizer, torch.nn.Module]:
    runtime = probe_nvidia()
    dtype = torch_dtype(runtime)
    attention = resolve_attention(config.model.attn_implementation)
    tokenizer_source = adapter_dir if use_adapter else None
    tokenizer = load_tokenizer(config.model.name, tokenizer_source)
    model_kwargs: dict = {
        "pretrained_model_name_or_path": config.model.name,
        "trust_remote_code": True,
        "attn_implementation": attention,
        "device_map": "auto" if runtime.cuda_available else None,
    }
    if runtime.cuda_available and config.model.use_qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.model.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        model_kwargs["torch_dtype"] = dtype if runtime.cuda_available else torch.float32
    base = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    adapter_path = Path(adapter_dir)
    if use_adapter and (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        model = base
    model.eval()
    return tokenizer, model


def generate_json(tokenizer: AutoTokenizer, model: torch.nn.Module, text: str, config: AppConfig) -> str:
    messages = build_messages(text)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    do_sample = config.infer.temperature > 0
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.infer.max_new_tokens,
            do_sample=do_sample,
            temperature=config.infer.temperature if do_sample else None,
            top_p=config.infer.top_p if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def detect(tokenizer: AutoTokenizer, model: torch.nn.Module, text: str, config: AppConfig) -> DetectionLabel:
    raw = generate_json(tokenizer, model, text, config)
    try:
        return parse_model_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to parse model output: {raw}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vulgar-language detection")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--text", default=None)
    parser.add_argument("--file", default=None, help="Plain text file, one example per line")
    args = parser.parse_args()
    configure_nvidia_runtime()
    config = load_config(args.config)
    adapter_dir = args.adapter or config.infer.adapter_dir
    tokenizer, model = load_detector(config, adapter_dir, use_adapter=not args.no_adapter)
    texts: list[str]
    if args.text:
        texts = [args.text]
    elif args.file:
        texts = [line.strip() for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raise SystemExit("Provide --text or --file")
    for text in texts:
        label = detect(tokenizer, model, text, config)
        print(json.dumps({"input_text": text, **json.loads(label.to_json())}, ensure_ascii=True))


if __name__ == "__main__":
    main()
