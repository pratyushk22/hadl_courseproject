"""Batched greedy answer generation + scoring on the real Odia QA splits."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .metrics import score_example, summarize_qa
from .modeling import free_memory, load_causal_lm, resolve_device_dtype, synchronize
from .prompts import encode_prompt, load_tokenizer, stop_token_ids
from .text_utils import clean_generation


def load_eval_model(model_id: str = config.MODEL_ID, adapter_dir: str | Path | None = None,
                    dtype_name: str = "auto", tokenizer_id: str | None = None):
    device, device_name, dtype = resolve_device_dtype(dtype_name)
    tokenizer = load_tokenizer(tokenizer_id or model_id, padding_side="left")
    model = load_causal_lm(model_id, dtype, device)
    if adapter_dir is not None:
        from peft import PeftModel

        if not Path(adapter_dir, "adapter_config.json").is_file():
            raise FileNotFoundError(f"No LoRA adapter at {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.config.use_cache = True
    model.eval()
    return model, tokenizer, device, device_name


def generate_answers(model, tokenizer, device: str, df: pd.DataFrame, batch_size: int = config.EVAL_BATCH_SIZE,
                     max_input_tokens: int = config.EVAL_MAX_INPUT_TOKENS,
                     max_new_tokens: int = config.EVAL_MAX_NEW_TOKENS) -> tuple[list[str], list[str], float]:
    """Returns (raw generations, cleaned answers, generation seconds). Batches are formed from
    length-sorted prompts; on CUDA OOM the batch is split in half and retried."""
    import torch
    from transformers import GenerationConfig

    prompts = [encode_prompt(p, q, tokenizer, max_tokens=max_input_tokens)
               for p, q in zip(df["passage"], df["question"])]
    order = np.argsort([-len(p) for p in prompts], kind="stable")
    gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens, num_beams=1,
                               eos_token_id=stop_token_ids(tokenizer), pad_token_id=tokenizer.pad_token_id)
    raw = [""] * len(prompts)
    target = next(model.parameters()).device

    def run(idx: list[int]):
        batch = tokenizer.pad({"input_ids": [prompts[i] for i in idx]}, padding=True, return_tensors="pt")
        batch = {k: v.to(target) for k, v in batch.items()}
        with torch.inference_mode():
            out = model.generate(**batch, generation_config=gen_cfg)
        new_tokens = out[:, batch["input_ids"].shape[1]:]
        for i, toks in zip(idx, new_tokens):
            raw[i] = tokenizer.decode(toks, skip_special_tokens=True)

    def run_safe(idx: list[int]):
        try:
            run(idx)
        except torch.cuda.OutOfMemoryError:
            if len(idx) == 1:
                raise
            free_memory()
            half = len(idx) // 2
            run_safe(idx[:half])
            run_safe(idx[half:])

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **_):
            return x
    synchronize(device)
    t0 = time.perf_counter()
    for s in tqdm(range(0, len(order), batch_size), desc="generating"):
        run_safe([int(i) for i in order[s:s + batch_size]])
    synchronize(device)
    seconds = time.perf_counter() - t0
    return raw, [clean_generation(r) for r in raw], seconds


def evaluate_frame(model, tokenizer, device: str, df: pd.DataFrame, run_id: str, split: str, out_dir: str | Path,
                   n_boot: int = 2000, **gen_kwargs) -> dict:
    """Generate, score and write <split>_predictions.jsonl + <split>_metrics.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, preds, seconds = generate_answers(model, tokenizer, device, df, **gen_kwargs)
    rows = []
    for (_, r), raw_text, pred in zip(df.iterrows(), raw, preds):
        s = score_example(pred, r["answers"])
        rows.append({"example_id": r["example_id"], "passage_id": r["passage_id"], "question": r["question"],
                     "gold_answers": list(r["answers"]), "prediction": pred, "raw_generation": raw_text, **s})
    with open(out_dir / f"{split}_predictions.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize_qa(rows, groups=[r["passage_id"] for r in rows], n_boot=n_boot)
    summary.update({
        "run_id": run_id,
        "split": split,
        "generation_seconds": seconds,
        "seconds_per_example": seconds / max(len(rows), 1),
        "empty_prediction_pct": 100.0 * float(np.mean([r["prediction"] == "" for r in rows])) if rows else 0.0,
    })
    (out_dir / f"{split}_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary
