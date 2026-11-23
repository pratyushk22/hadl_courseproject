"""Qwen3-1.7B + LoRA (r=8) fine-tuning with time measurement.

Same hyper-parameters as model_training/train_qwen_lora.py (Aditya Sahu), with these fixes
needed for a correct end-to-end comparison:
* reads the passage from the right column (the synthetic files call it "paragraph";
  reading "context" silently trained on an empty passage);
* truncates only the passage, keeping the answer inside the window (the old code cut the
  prompt tail, i.e. the question and the assistant header);
* LoRA weights are kept in fp32 so fp16 mixed precision works on T4 GPUs;
* the base model is put on one GPU per process (no accidental model-parallel split);
* every micro-batch is timed (CUDA-synchronised) with its (batch, padded length) shape, so
  the time proxy can be evaluated step by step, not only on the epoch total.
"""
from __future__ import annotations

import inspect
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .modeling import free_memory, is_main_process, load_causal_lm, resolve_device_dtype, synchronize
from .prompts import build_training_features, load_tokenizer
from .time_proxy import TimeProxy


def _filter_kwargs(cls, kwargs: dict) -> dict:
    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def tokenize_frame(df: pd.DataFrame, tokenizer, max_seq_len: int):
    from datasets import Dataset

    feats = [build_training_features(p, q, a, tokenizer, max_seq_len=max_seq_len)
             for p, q, a in zip(df["passage"], df["question"], df["answer"])]
    n_bad = sum(all(l == -100 for l in f["labels"]) for f in feats)
    if n_bad:
        raise ValueError(f"{n_bad} examples have no supervised tokens")
    return Dataset.from_list(feats), np.array([len(f["input_ids"]) for f in feats])


def train_lora(train_df: pd.DataFrame, out_dir: str | Path, *, model_id: str = config.MODEL_ID,
               epochs: float = config.EPOCHS, batch_size: int = config.BATCH_SIZE,
               grad_accum: int = config.GRAD_ACCUM_STEPS, lr: float = config.LEARNING_RATE,
               max_seq_len: int = config.MAX_SEQ_LEN, lora_rank: int = config.LORA_RANK,
               lora_alpha: int = config.LORA_ALPHA, lora_dropout: float = config.LORA_DROPOUT,
               warmup_ratio: float = config.WARMUP_RATIO, seed: int = 42, dtype_name: str = "auto",
               max_steps: int = -1, time_steps: bool = True, proxy_json: str | Path = config.TIME_PROXY_JSON,
               tokenizer_id: str | None = None, extra_metadata: dict | None = None) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device, device_name, dtype = resolve_device_dtype(dtype_name)
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))

    tokenizer = load_tokenizer(tokenizer_id or model_id, padding_side="right")
    dataset, seq_lens = tokenize_frame(train_df, tokenizer, max_seq_len)
    proxy = TimeProxy.from_json(proxy_json, max_seq_len=max_seq_len)
    per_sample_ms = proxy.per_sample_ms(seq_lens, batch_size=batch_size)
    is_syn = (train_df["source"].values == "synthetic")

    model = load_causal_lm(model_id, dtype, device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora_cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                          target_modules=config.LORA_TARGET_MODULES, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    for p in model.parameters():          # fp32 adapters -> GradScaler can unscale them
        if p.requires_grad:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    use_cuda = device == "cuda"
    steps_per_epoch = math.ceil(len(dataset) / (batch_size * grad_accum * world_size))
    total_steps = max_steps if max_steps > 0 else max(1, math.ceil(steps_per_epoch * epochs))
    args_kwargs = dict(
        output_dir=str(out_dir / "trainer_tmp"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(round(warmup_ratio * total_steps)),
        max_grad_norm=1.0,
        weight_decay=0.0,
        fp16=use_cuda and dtype == torch.float16,
        bf16=use_cuda and dtype == torch.bfloat16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        optim="adamw_torch",
        seed=seed,
        data_seed=seed,
        dataloader_pin_memory=use_cuda,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        use_cpu=device == "cpu",
    )
    training_args = TrainingArguments(**_filter_kwargs(TrainingArguments, args_kwargs))
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8,
                                      label_pad_token_id=-100)

    step_records: list[dict] = []

    class TimedTrainer(Trainer):
        """Times every micro-batch (forward + backward) with device synchronisation."""

        def training_step(self, *args, **kwargs):
            if not time_steps:
                return super().training_step(*args, **kwargs)
            inputs = args[1] if len(args) > 1 else kwargs.get("inputs")
            synchronize(device)
            t0 = time.perf_counter()
            loss = super().training_step(*args, **kwargs)
            synchronize(device)
            ids = inputs["input_ids"]
            step_records.append({
                "micro_step": len(step_records),
                "global_step": int(self.state.global_step),
                "batch_size": int(ids.shape[0]),
                "padded_seq_len": int(ids.shape[1]),
                "real_tokens": int(inputs["attention_mask"].sum().item()) if "attention_mask" in inputs else int(ids.numel()),
                "measured_ms": (time.perf_counter() - t0) * 1000.0,
            })
            return loss

    trainer = TimedTrainer(model=model, args=training_args, train_dataset=dataset, data_collator=collator)

    free_memory()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    t_start = time.perf_counter()
    train_result = trainer.train()
    synchronize(device)
    wall_sec = time.perf_counter() - t_start

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if use_cuda else None
    epochs_done = float(train_result.metrics.get("epoch", epochs) or epochs)
    steps_df = pd.DataFrame(step_records)
    if len(steps_df):
        steps_df["proxy_predicted_ms"] = proxy.step_ms(steps_df["batch_size"].values, steps_df["padded_seq_len"].values)

    metrics = {
        "model_id": model_id,
        "device": device,
        "device_name": device_name,
        "dtype": str(dtype).replace("torch.", ""),
        "world_size": world_size,
        "num_train_examples": int(len(train_df)),
        "num_synthetic_examples": int(is_syn.sum()),
        "num_real_examples": int((~is_syn).sum()),
        "epochs": epochs,
        "epochs_completed": epochs_done,
        "optimizer_steps": int(train_result.global_step),
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "learning_rate": lr,
        "max_seq_len": max_seq_len,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "trainable_params": int(trainable),
        "total_params": int(total_params),
        "seed": seed,
        "mean_seq_len": float(seq_lens.mean()) if len(seq_lens) else 0.0,
        "total_train_tokens_per_epoch": int(seq_lens.sum()),
        "final_train_loss": float(train_result.training_loss) if np.isfinite(train_result.training_loss) else None,
        # ---- time: actual vs proxy -----------------------------------------------------
        "actual_wall_clock_sec": wall_sec,
        "trainer_runtime_sec": float(train_result.metrics.get("train_runtime", wall_sec)),
        "predicted_training_time_sec": float(per_sample_ms.sum() * epochs_done / 1000.0 / world_size),
        "predicted_synthetic_time_sec": float(per_sample_ms[is_syn].sum() * epochs_done / 1000.0 / world_size),
        "predicted_real_time_sec": float(per_sample_ms[~is_syn].sum() * epochs_done / 1000.0 / world_size),
        "proxy_file": str(proxy_json),
        "proxy_profiled_on": proxy.gpu_name,
        "peak_gpu_memory_gb": peak_mem_gb,
        "throughput_tokens_per_sec": float(seq_lens.sum() * epochs_done / wall_sec) if wall_sec > 0 else None,
    }
    if len(steps_df):
        metrics["measured_microbatch_sum_sec"] = float(steps_df["measured_ms"].sum() / 1000.0)
        metrics["predicted_from_observed_batches_sec"] = float(steps_df["proxy_predicted_ms"].sum() / 1000.0)
        metrics["padding_fraction"] = float(1.0 - steps_df["real_tokens"].sum()
                                            / (steps_df["batch_size"] * steps_df["padded_seq_len"]).sum())
    act = metrics["actual_wall_clock_sec"]
    metrics["proxy_abs_error_sec"] = abs(metrics["predicted_training_time_sec"] - act)
    metrics["proxy_error_pct"] = 100.0 * (metrics["predicted_training_time_sec"] - act) / act if act > 0 else None
    metrics.update(extra_metadata or {})

    if is_main_process():
        adapter_dir = out_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        metrics["adapter_dir"] = str(adapter_dir)
        (out_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
        if len(steps_df):
            steps_df.to_csv(out_dir / "step_timings.csv", index=False)
        (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
        tmp = out_dir / "trainer_tmp"
        if tmp.is_dir() and not any(tmp.iterdir()):
            tmp.rmdir()

    del trainer, model
    free_memory()
    return metrics
