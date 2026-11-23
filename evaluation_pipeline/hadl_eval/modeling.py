"""Model loading helpers shared by training and evaluation (device, dtype, version quirks)."""
from __future__ import annotations

import gc
import os


def native_bf16() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def resolve_device_dtype(dtype_name: str = "auto"):
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
        # Native bf16 needs compute capability >= 8.0 (A100, L4, H100). torch.cuda.is_bf16_supported()
        # also returns True on T4/P100, where bf16 is only *emulated* and several times slower.
        auto = torch.bfloat16 if native_bf16() else torch.float16
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device, name, auto = "mps", "Apple Silicon MPS", torch.float16
    else:
        device, name, auto = "cpu", "CPU", torch.float32
    table = {"auto": auto, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype_name not in table:
        raise ValueError(f"dtype must be one of {list(table)}")
    return device, name, table[dtype_name]


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def load_causal_lm(model_id: str, dtype, device: str):
    """Load the base model on a single device (or the local rank's GPU under torchrun/DDP).

    device_map="auto" is deliberately NOT used: on a dual-T4 machine it would split the
    model across both GPUs (naive pipeline parallelism) and distort the time measurements.
    """
    import transformers
    from packaging import version
    from transformers import AutoModelForCausalLM

    kwargs = {"low_cpu_mem_usage": True}
    # transformers >= 4.56 renamed torch_dtype -> dtype
    if version.parse(transformers.__version__) >= version.parse("4.56.0"):
        kwargs["dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", **kwargs)
    except (ValueError, ImportError):
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    target = f"cuda:{local_rank()}" if device == "cuda" else device
    return model.to(target)


def free_memory():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize(device: str):
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
