from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NvidiaRuntime:
    cuda_available: bool
    device_count: int
    device_name: str | None
    total_memory_gb: float | None
    compute_capability: tuple[int, int] | None
    bf16_supported: bool
    tf32_enabled: bool


def configure_nvidia_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def probe_nvidia() -> NvidiaRuntime:
    if not torch.cuda.is_available():
        return NvidiaRuntime(False, 0, None, None, None, False, False)
    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    return NvidiaRuntime(
        cuda_available=True,
        device_count=torch.cuda.device_count(),
        device_name=torch.cuda.get_device_name(0),
        total_memory_gb=round(props.total_memory / (1024**3), 2),
        compute_capability=capability,
        bf16_supported=capability[0] >= 8,
        tf32_enabled=bool(torch.backends.cuda.matmul.allow_tf32),
    )


def torch_dtype(runtime: NvidiaRuntime) -> torch.dtype:
    if runtime.bf16_supported:
        return torch.bfloat16
    if runtime.cuda_available:
        return torch.float16
    return torch.float32


def resolve_attention(preferred: str) -> str:
    if preferred != "flash_attention_2":
        return preferred
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"
