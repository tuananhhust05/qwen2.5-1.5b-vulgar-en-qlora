from __future__ import annotations

import argparse
import inspect
from dataclasses import fields as dataclass_fields
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from vulgar_en.config import AppConfig, load_config
from vulgar_en.nvidia import configure_nvidia_runtime, probe_nvidia, resolve_attention, torch_dtype


def _filter_kwargs(callable_obj: object, kwargs: dict) -> dict:
    allowed: set[str] = set()
    try:
        allowed.update(item.name for item in dataclass_fields(callable_obj))
    except TypeError:
        pass
    try:
        allowed.update(inspect.signature(callable_obj.__init__).parameters)  # type: ignore[misc]
    except (TypeError, ValueError):
        pass
    allowed.discard("self")
    filtered = {key: value for key, value in kwargs.items() if key in allowed}
    if "max_length" in filtered and "max_seq_length" in filtered:
        filtered.pop("max_seq_length")
    return filtered


def _render_text(tokenizer: AutoTokenizer):
    def _map(batch: dict) -> dict:
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in batch["messages"]
        ]
        return {"text": texts}

    return _map


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(config: AppConfig, tokenizer: AutoTokenizer) -> AutoModelForCausalLM:
    runtime = probe_nvidia()
    if not runtime.cuda_available:
        raise RuntimeError("Training requires an NVIDIA GPU with CUDA.")
    dtype = torch_dtype(runtime)
    attention = resolve_attention(config.model.attn_implementation)
    model_kwargs: dict = {
        "pretrained_model_name_or_path": config.model.name,
        "trust_remote_code": True,
        "attn_implementation": attention,
        "device_map": "auto",
    }
    if config.model.use_qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.model.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.config.use_cache = False
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if config.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def build_lora(config: AppConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(config.lora.target_modules),
    )


def build_sft_args(config: AppConfig) -> SFTConfig:
    runtime = probe_nvidia()
    kwargs = {
        "output_dir": config.train.output_dir,
        "num_train_epochs": config.train.num_train_epochs,
        "per_device_train_batch_size": config.train.per_device_train_batch_size,
        "per_device_eval_batch_size": config.train.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.train.gradient_accumulation_steps,
        "learning_rate": config.train.learning_rate,
        "warmup_ratio": config.train.warmup_ratio,
        "weight_decay": config.train.weight_decay,
        "lr_scheduler_type": config.train.lr_scheduler_type,
        "logging_steps": config.train.logging_steps,
        "save_strategy": config.train.save_strategy,
        "eval_strategy": config.train.eval_strategy,
        "save_total_limit": config.train.save_total_limit,
        "seed": config.train.seed,
        "gradient_checkpointing": config.train.gradient_checkpointing,
        "optim": config.train.optim,
        "max_grad_norm": config.train.max_grad_norm,
        "dataloader_num_workers": config.train.dataloader_num_workers,
        "report_to": config.train.report_to,
        "bf16": runtime.bf16_supported,
        "fp16": runtime.cuda_available and not runtime.bf16_supported,
        "tf32": runtime.cuda_available,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataset_text_field": "text",
        "max_length": config.model.max_seq_length,
        "max_seq_length": config.model.max_seq_length,
        "packing": False,
        "remove_unused_columns": False,
    }
    return SFTConfig(**_filter_kwargs(SFTConfig, kwargs))


def load_splits(config: AppConfig, tokenizer: AutoTokenizer):
    train_path = Path(config.data.train_file)
    eval_path = Path(config.data.eval_file)
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError("Processed data not found. Run python -m vulgar_en.prepare_data first.")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "eval": str(eval_path),
        },
    )
    mapped = dataset.map(_render_text(tokenizer), batched=True, remove_columns=["messages"])
    return mapped["train"], mapped["eval"]


def train(config_path: str) -> None:
    configure_nvidia_runtime()
    config = load_config(config_path)
    tokenizer = load_tokenizer(config.model.name)
    train_ds, eval_ds = load_splits(config, tokenizer)
    model = load_model(config, tokenizer)
    sft_args = build_sft_args(config)
    trainer_kwargs = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "peft_config": build_lora(config),
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer = SFTTrainer(**_filter_kwargs(SFTTrainer, trainer_kwargs))
    trainer.train()
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved adapter to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune English vulgar-language detector")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
