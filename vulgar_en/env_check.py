from __future__ import annotations

import argparse
import json
from pathlib import Path

from vulgar_en.nvidia import configure_nvidia_runtime, probe_nvidia


def _maybe_nvml() -> list[str]:
    lines: list[str] = []
    try:
        from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit, nvmlShutdown

        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        memory = nvmlDeviceGetMemoryInfo(handle)
        lines.append(f"nvml_free_gb={memory.free / (1024 ** 3):.2f}")
        lines.append(f"nvml_used_gb={memory.used / (1024 ** 3):.2f}")
        nvmlShutdown()
    except Exception as exc:
        lines.append(f"nvml_status={exc.__class__.__name__}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify NVIDIA CUDA training environment")
    parser.parse_args()
    configure_nvidia_runtime()
    runtime = probe_nvidia()
    payload = {
        "cuda_available": runtime.cuda_available,
        "device_count": runtime.device_count,
        "device_name": runtime.device_name,
        "total_memory_gb": runtime.total_memory_gb,
        "compute_capability": runtime.compute_capability,
        "bf16_supported": runtime.bf16_supported,
        "tf32_enabled": runtime.tf32_enabled,
    }
    print(json.dumps(payload, indent=2))
    for line in _maybe_nvml():
        print(line)
    if not runtime.cuda_available:
        raise SystemExit("CUDA is not available. Train on an NVIDIA GPU machine.")
    if runtime.total_memory_gb is not None and runtime.total_memory_gb < 8:
        print("Warning: less than 8GB VRAM. Reduce batch size in configs/default.yaml.")
    Path("outputs").mkdir(exist_ok=True)
    print("NVIDIA environment looks ready.")


if __name__ == "__main__":
    main()
