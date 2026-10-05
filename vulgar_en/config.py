from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    max_seq_length: int
    attn_implementation: str
    use_qlora: bool
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool


@dataclass(frozen=True)
class LoraConfigValues:
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: str
    logging_steps: int
    save_strategy: str
    eval_strategy: str
    save_total_limit: int
    seed: int
    gradient_checkpointing: bool
    optim: str
    max_grad_norm: float
    dataloader_num_workers: int
    report_to: str


@dataclass(frozen=True)
class DataConfig:
    processed_dir: str
    train_file: str
    eval_file: str
    test_file: str
    labeled_file: str
    include_seed: bool
    val_ratio: float
    test_ratio: float
    seed: int


@dataclass(frozen=True)
class InferConfig:
    adapter_dir: str
    max_new_tokens: int
    temperature: float
    top_p: float


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    lora: LoraConfigValues
    train: TrainConfig
    data: DataConfig
    infer: InferConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing config key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    model = _require(raw, "model")
    lora = _require(raw, "lora")
    train = _require(raw, "train")
    data = _require(raw, "data")
    infer = _require(raw, "infer")
    return AppConfig(
        model=ModelConfig(
            name=str(_require(model, "name")),
            max_seq_length=int(_require(model, "max_seq_length")),
            attn_implementation=str(_require(model, "attn_implementation")),
            use_qlora=bool(_require(model, "use_qlora")),
            bnb_4bit_quant_type=str(_require(model, "bnb_4bit_quant_type")),
            bnb_4bit_use_double_quant=bool(_require(model, "bnb_4bit_use_double_quant")),
        ),
        lora=LoraConfigValues(
            r=int(_require(lora, "r")),
            alpha=int(_require(lora, "alpha")),
            dropout=float(_require(lora, "dropout")),
            target_modules=tuple(str(item) for item in _require(lora, "target_modules")),
        ),
        train=TrainConfig(
            output_dir=str(_require(train, "output_dir")),
            num_train_epochs=float(_require(train, "num_train_epochs")),
            per_device_train_batch_size=int(_require(train, "per_device_train_batch_size")),
            per_device_eval_batch_size=int(_require(train, "per_device_eval_batch_size")),
            gradient_accumulation_steps=int(_require(train, "gradient_accumulation_steps")),
            learning_rate=float(_require(train, "learning_rate")),
            warmup_ratio=float(_require(train, "warmup_ratio")),
            weight_decay=float(_require(train, "weight_decay")),
            lr_scheduler_type=str(_require(train, "lr_scheduler_type")),
            logging_steps=int(_require(train, "logging_steps")),
            save_strategy=str(_require(train, "save_strategy")),
            eval_strategy=str(_require(train, "eval_strategy")),
            save_total_limit=int(_require(train, "save_total_limit")),
            seed=int(_require(train, "seed")),
            gradient_checkpointing=bool(_require(train, "gradient_checkpointing")),
            optim=str(_require(train, "optim")),
            max_grad_norm=float(_require(train, "max_grad_norm")),
            dataloader_num_workers=int(_require(train, "dataloader_num_workers")),
            report_to=str(_require(train, "report_to")),
        ),
        data=DataConfig(
            processed_dir=str(_require(data, "processed_dir")),
            train_file=str(_require(data, "train_file")),
            eval_file=str(_require(data, "eval_file")),
            test_file=str(_require(data, "test_file")),
            labeled_file=str(data.get("labeled_file", "data/raw/labeled.jsonl")),
            include_seed=bool(data.get("include_seed", True)),
            val_ratio=float(_require(data, "val_ratio")),
            test_ratio=float(_require(data, "test_ratio")),
            seed=int(_require(data, "seed")),
        ),
        infer=InferConfig(
            adapter_dir=str(_require(infer, "adapter_dir")),
            max_new_tokens=int(_require(infer, "max_new_tokens")),
            temperature=float(_require(infer, "temperature")),
            top_p=float(_require(infer, "top_p")),
        ),
    )
