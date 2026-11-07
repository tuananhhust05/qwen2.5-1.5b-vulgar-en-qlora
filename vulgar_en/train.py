from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import time
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
    if "processing_class" in filtered and "tokenizer" in filtered:
        filtered.pop("tokenizer")
    return filtered


def to_prompt_completion(batch: dict) -> dict:
    """Split chat messages so the loss is computed on the assistant JSON only."""
    prompts = [messages[:-1] for messages in batch["messages"]]
    completions = [messages[-1:] for messages in batch["messages"]]
    for completion in completions:
        if completion[0]["role"] != "assistant":
            raise ValueError("Last message of every training example must be the assistant answer")
    return {"prompt": prompts, "completion": completions}


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
        "device_map": {"": 0},
        "dtype": dtype,
    }
    if config.model.use_qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.model.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.config.use_cache = False
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
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
        "data_seed": config.train.seed,
        "gradient_checkpointing": config.train.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": config.train.optim,
        "max_grad_norm": config.train.max_grad_norm,
        "dataloader_num_workers": config.train.dataloader_num_workers,
        "report_to": config.train.report_to,
        "bf16": runtime.bf16_supported,
        "fp16": runtime.cuda_available and not runtime.bf16_supported,
        "tf32": runtime.cuda_available,
        "max_length": config.model.max_seq_length,
        "max_seq_length": config.model.max_seq_length,
        "packing": False,
        "completion_only_loss": True,
        # transformers >= 5 names it train_sampling_strategy; older versions use group_by_length
        "train_sampling_strategy": "group_by_length" if config.train.group_by_length else "random",
        "group_by_length": config.train.group_by_length,
        "load_best_model_at_end": config.train.save_strategy == config.train.eval_strategy != "no",
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    return SFTConfig(**_filter_kwargs(SFTConfig, kwargs))


def load_splits(config: AppConfig):
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
    columns = dataset["train"].column_names
    mapped = dataset.map(to_prompt_completion, batched=True, remove_columns=columns)
    return mapped["train"], mapped["eval"]


def check_loss_mask(trainer: SFTTrainer, tokenizer: AutoTokenizer) -> dict:
    """Fail fast if the collator would train on prompt tokens."""
    batch = trainer.data_collator([trainer.train_dataset[0]])
    labels = batch["labels"][0]
    supervised = labels[labels != -100]
    text = tokenizer.decode(supervised)
    if "is_vulgar" not in text or "Text:" in text or "vulgar-language detector" in text:
        raise RuntimeError(f"Unexpected supervised tokens: {text!r}")
    info = {
        "sequence_tokens": int(batch["input_ids"].shape[1]),
        "supervised_tokens": int(supervised.numel()),
        "supervised_text": text,
    }
    print(f"loss_mask_check={json.dumps(info)}", flush=True)
    return info


def override(
    config: AppConfig,
    model: str | None = None,
    output_dir: str | None = None,
    seed: int | None = None,
    epochs: float | None = None,
    rank: int | None = None,
) -> AppConfig:
    model_cfg, lora_cfg, train_cfg, infer_cfg = config.model, config.lora, config.train, config.infer
    if model:
        model_cfg = dataclasses.replace(model_cfg, name=model)
    if rank:
        lora_cfg = dataclasses.replace(lora_cfg, r=rank, alpha=2 * rank)
    if output_dir:
        train_cfg = dataclasses.replace(train_cfg, output_dir=output_dir)
        infer_cfg = dataclasses.replace(infer_cfg, adapter_dir=output_dir)
    if seed is not None:
        train_cfg = dataclasses.replace(train_cfg, seed=seed)
    if epochs:
        train_cfg = dataclasses.replace(train_cfg, num_train_epochs=epochs)
    return dataclasses.replace(config, model=model_cfg, lora=lora_cfg, train=train_cfg, infer=infer_cfg)


def train(config: AppConfig) -> None:
    configure_nvidia_runtime()
    tokenizer = load_tokenizer(config.model.name)
    train_ds, eval_ds = load_splits(config)
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
    mask_info = check_loss_mask(trainer, tokenizer)
    trainable, total = trainer.model.get_nb_trainable_parameters()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - started
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    meta = {
        "model": config.model.name,
        "qlora": config.model.use_qlora,
        "lora": dataclasses.asdict(config.lora),
        "train": dataclasses.asdict(config.train),
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds),
        "trainable_params": trainable,
        "total_params": total,
        "train_seconds": elapsed,
        "train_loss": result.training_loss,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu": torch.cuda.get_device_name(0),
        "loss_mask_check": mask_info,
        "log_history": trainer.state.log_history,
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved adapter to {output_dir} in {elapsed:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune English vulgar-language detector")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, help="override model.name")
    parser.add_argument("--output-dir", default=None, help="override train.output_dir")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None, help="override LoRA r (alpha = 2r)")
    args = parser.parse_args()
    config = override(load_config(args.config), args.model, args.output_dir, args.seed, args.epochs, args.rank)
    train(config)


if __name__ == "__main__":
    main()
