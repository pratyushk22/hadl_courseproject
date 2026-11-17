import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import gc
import math
import time
import json
import argparse
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen3-1.7B"
SUBSET_DIR = "../data_and_scripts/odia_synth_cleaned/selected_subsets"
CHECKPOINT_ROOT = "./checkpoints"

# Mandated Project Hyperparameters
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

METHOD_FILES = {
    "random": "subset_random.parquet",
    "quality_only": "subset_quality_only.parquet",
    "proposed_knapsack": "subset_proposed_knapsack.parquet",
}


def get_device_and_dtype():
    if torch.cuda.is_available():
        device = "cuda:0"
        gpu_name = torch.cuda.get_device_name(0)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        gpu_name = "Apple Silicon MPS"
        dtype = torch.float16
    else:
        device = "cpu"
        gpu_name = "CPU"
        dtype = torch.float32
    return device, gpu_name, dtype


def empty_memory(device):
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def tokenize_with_answer_only_loss(example, tokenizer, max_seq_len):
    """
    Formats an Odia passage + question + answer using Qwen3 non-thinking mode.
    Guarantees the answer tokens are never truncated off when a passage is long,
    preventing all-(-100) label batches and NaN Cross-Entropy loss.
    """
    passage = str(example.get("context", ""))
    question = str(example.get("question", ""))
    answer = str(example.get("answer", "")).strip()
    if not answer:
        answer = "ଉତ୍ତର ଉପଲବ୍ଧ ନାହିଁ"

    prompt_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions in Odia based strictly on the provided passage."
        },
        {
            "role": "user",
            "content": f"Passage:\n{passage}\n\nQuestion:\n{question}"
        }
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    answer_text = f"{answer}{tokenizer.eos_token}"

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)

    # Cap answer length to at most 128 tokens and reserve space so answer is NEVER truncated out
    answer_ids = answer_ids[:128]
    max_prompt_len = max(64, max_seq_len - len(answer_ids))
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[:max_prompt_len]

    input_ids = prompt_ids + answer_ids
    labels = ([-100] * len(prompt_ids)) + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels
    }


def train_single_method(method_name, args, device, gpu_name, dtype, tokenizer):
    subset_path = os.path.join(args.subset_dir, METHOD_FILES[method_name])
    if not os.path.exists(subset_path):
        raise FileNotFoundError(f"Subset file not found: {subset_path}. Run prepare_training_subsets.py first!")

    print(f"\n======================================================================")
    print(f"Starting Training Run for Baseline: {method_name.upper()}")
    print(f"Hardware: {gpu_name} ({device}) | Precision: {dtype}")
    print(f"======================================================================")

    df = pd.read_parquet(subset_path)
    if args.max_samples > 0 and len(df) > args.max_samples:
        print(f"Subsampling first {args.max_samples} examples (out of {len(df)}) for quick verification...")
        df = df.iloc[:args.max_samples].copy()

    # Predicted training time from Yatika's time proxy (ms -> seconds)
    predicted_epoch_sec = float(df["predicted_time_cost_ms"].sum() / 1000.0)
    predicted_total_sec = predicted_epoch_sec * args.epochs

    raw_dataset = Dataset.from_pandas(df)
    tokenized_dataset = raw_dataset.map(
        lambda ex: tokenize_with_answer_only_loss(ex, tokenizer, args.max_seq_len),
        remove_columns=raw_dataset.column_names,
        desc=f"Tokenizing {method_name}"
    )

    print(f"Loading {MODEL_ID} with LoRA (r={LORA_RANK}, alpha={LORA_ALPHA})...")
    is_cuda = device.startswith("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if is_cuda else None,
        attn_implementation="sdpa"
    )
    if not is_cuda:
        model.to(device)
    model.config.use_cache = False

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    method_out_dir = os.path.join(args.output_dir, method_name)
    os.makedirs(method_out_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=method_out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        max_grad_norm=1.0,
        fp16=(is_cuda and dtype == torch.float16),
        bf16=(is_cuda and dtype == torch.bfloat16),
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        optim="adamw_torch",
        dataloader_pin_memory=is_cuda,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    empty_memory(device)
    if is_cuda:
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    t_start = time.perf_counter()

    train_result = trainer.train()

    if is_cuda:
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    actual_total_sec = time.perf_counter() - t_start

    final_adapter_dir = os.path.join(method_out_dir, "final_adapter")
    trainer.save_model(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    proxy_abs_error_sec = abs(actual_total_sec - predicted_total_sec)
    proxy_rel_error_pct = (proxy_abs_error_sec / max(actual_total_sec, 1e-6)) * 100.0
    final_loss = float(train_result.training_loss)

    metrics_log = {
        "method": method_name,
        "gpu_name": gpu_name,
        "num_train_samples": int(len(df)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "final_train_loss": round(final_loss, 4) if not math.isnan(final_loss) else None,
        "predicted_training_time_sec": round(predicted_total_sec, 3),
        "actual_wall_clock_time_sec": round(actual_total_sec, 3),
        "proxy_time_error_pct": round(proxy_rel_error_pct, 2),
        "adapter_checkpoint_path": final_adapter_dir
    }

    metrics_path = os.path.join(method_out_dir, f"training_metrics_{method_name}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=4)

    print(f"\n--- {method_name.upper()} Run Complete ---")
    print(f"Predicted Training Time : {predicted_total_sec:.2f} s")
    print(f"Actual Wall-Clock Time  : {actual_total_sec:.2f} s")
    print(f"Proxy Method Error      : {proxy_rel_error_pct:.2f}%")
    print(f"Final Training Loss     : {final_loss:.4f}")
    print(f"Saved LoRA Adapter to   : {final_adapter_dir}")

    del trainer, model
    empty_memory(device)

    return metrics_log


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-1.7B with LoRA (r=8) on selected Odia QA subsets.")
    parser.add_argument("--method", type=str, default="proposed_knapsack",
                        choices=["random", "quality_only", "proposed_knapsack", "all"],
                        help="Which baseline subset to train on (or 'all' to run all 3 sequentially).")
    parser.add_argument("--subset_dir", type=str, default=SUBSET_DIR)
    parser.add_argument("--output_dir", type=str, default=CHECKPOINT_ROOT)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=TARGET_BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Optional cap on samples for quick local Mac verification (e.g., --max_samples 8).")
    args = parser.parse_args()

    device, gpu_name, dtype = get_device_and_dtype()

    if device == "mps":
        if args.max_samples == -1:
            args.max_samples = 8
        args.batch_size = 1
        args.grad_accum = 4
        print(f"Local Mac MPS detected: using float16 + gradient checkpointing + micro-batch=1 (grad_accum=4, max_samples={args.max_samples}).")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    methods_to_run = ["random", "quality_only", "proposed_knapsack"] if args.method == "all" else [args.method]

    all_results = []
    for m in methods_to_run:
        res = train_single_method(m, args, device, gpu_name, dtype, tokenizer)
        all_results.append(res)

    summary_path = os.path.join(args.output_dir, "all_training_runs_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved training summary across {len(all_results)} run(s) to '{summary_path}'")


if __name__ == "__main__":
    main()
