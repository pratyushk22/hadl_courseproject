import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import re
import time
import json
import string
import argparse
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_EVAL_FILE = "../data_and_scripts/odia_real_cleaned/validation.jsonl"
DEFAULT_ADAPTER = "./checkpoints/proposed_knapsack/final_adapter"


def normalize_odia_answer(text):
    """
    Normalizes Odia / Indic answer strings by stripping punctuation,
    extra whitespace, and lowercasing ASCII characters.
    """
    if text is None:
        return ""
    text = str(text).strip().lower()
    punct_to_remove = set(string.punctuation) | {"।", "॥", "‘", "’", "“", "”"}
    text = "".join(ch for ch in text if ch not in punct_to_remove)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_exact_match(pred, gold):
    return float(normalize_odia_answer(pred) == normalize_odia_answer(gold))


def compute_token_f1(pred, gold):
    pred_tokens = normalize_odia_answer(pred).split()
    gold_tokens = normalize_odia_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def extract_gold_answers(row):
    """
    Handles both flat {'answer': '...'} and SQuAD/IndicQA style {'answers': {'text': [...]}} schemas.
    """
    if "answer" in row and row["answer"]:
        ans = row["answer"]
        return [ans] if isinstance(ans, str) else list(ans)
    if "answers" in row and isinstance(row["answers"], dict):
        texts = row["answers"].get("text", [])
        if texts:
            return list(texts)
    if "answers" in row and isinstance(row["answers"], list):
        return [str(x) for x in row["answers"]]
    return [""]


def build_truncated_prompt(passage, question, tokenizer, max_input_tokens):
    """
    Truncates only the passage tokens if needed so that the question and
    Qwen3 non-thinking assistant generation header are NEVER cut off.
    """
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    max_passage_tokens = max(64, max_input_tokens - len(question_tokens) - 80)
    passage_tokens = tokenizer.encode(passage, add_special_tokens=False)
    if len(passage_tokens) > max_passage_tokens:
        passage = tokenizer.decode(passage_tokens[:max_passage_tokens], skip_special_tokens=True)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions in Odia based strictly on the provided passage. Reply with only the concise answer."
        },
        {
            "role": "user",
            "content": f"Passage:\n{passage}\n\nQuestion:\n{question}"
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )


def main():
    parser = argparse.ArgumentParser(description="Verify trained Qwen3-1.7B LoRA adapter on real Odia QA split.")
    parser.add_argument("--eval_file", type=str, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--adapter_dir", type=str, default=DEFAULT_ADAPTER)
    parser.add_argument("--output_jsonl", type=str, default="eval_predictions.jsonl")
    parser.add_argument("--max_eval_samples", type=int, default=10,
                        help="Number of validation samples to evaluate (-1 for full split).")
    parser.add_argument("--max_input_tokens", type=int, default=512,
                        help="Max prompt tokens to prevent long Odia passages from swapping RAM on MPS.")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="Max generated answer tokens.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda:0"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Loading base model {MODEL_ID} on {device} ({dtype})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa"
    ).to(device)

    if os.path.exists(args.adapter_dir):
        print(f"Loading and merging trained LoRA adapter from {args.adapter_dir}...", flush=True)
        peft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
        model = peft_model.merge_and_unload()
    else:
        print(f"Adapter path '{args.adapter_dir}' not found; running zero-shot base model.", flush=True)
        model = base_model

    # Silence Qwen3 greedy decoding warnings for sampling flags
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.config.use_cache = True
    model.eval()

    rows = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if args.max_eval_samples > 0:
        rows = rows[:args.max_eval_samples]

    em_scores, f1_scores, predictions_out = [], [], []

    print(f"\nEvaluating on {len(rows)} real Odia QA examples from {args.eval_file}...")
    print("-" * 80, flush=True)

    for idx, row in enumerate(rows):
        t0 = time.perf_counter()
        passage = str(row.get("context", row.get("passage", "")))
        question = str(row.get("question", ""))
        gold_list = extract_gold_answers(row)

        prompt = build_truncated_prompt(passage, question, tokenizer, args.max_input_tokens)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.inference_mode():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        gen_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
        pred_answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        pred_answer = pred_answer.split("\n")[0].strip()

        best_em = max(compute_exact_match(pred_answer, g) for g in gold_list)
        best_f1 = max(compute_token_f1(pred_answer, g) for g in gold_list)
        em_scores.append(best_em)
        f1_scores.append(best_f1)

        elapsed_s = time.perf_counter() - t0
        print(f"[{idx+1:>2}/{len(rows)}] ({elapsed_s:.1f}s) EM={best_em:.0f} F1={best_f1:.2f} | Pred: {pred_answer[:40]} | Gold: {str(gold_list[0])[:40]}", flush=True)

        predictions_out.append({
            "id": row.get("id", idx),
            "question": question,
            "gold_answers": gold_list,
            "predicted_answer": pred_answer,
            "normalized_em": best_em,
            "token_f1": round(best_f1, 4)
        })

    mean_em = (sum(em_scores) / max(len(em_scores), 1)) * 100.0
    mean_f1 = (sum(f1_scores) / max(len(f1_scores), 1)) * 100.0

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in predictions_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n=== Odia QA Evaluation Results ===")
    print(f"Samples Evaluated            : {len(rows)}")
    print(f"Normalized Exact Match (EM)  : {mean_em:.2f}%")
    print(f"Answer Overlap Token F1      : {mean_f1:.2f}%")
    print(f"Saved predictions to         : {args.output_jsonl}")


if __name__ == "__main__":
    main()
