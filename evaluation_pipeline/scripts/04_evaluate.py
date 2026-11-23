"""Step 4 - generate answers on the real Odia QA split(s) and score them (GPU).

Evaluates every trained adapter in outputs/runs/ (or the ones given with --runs) and, unless
--no_base, the untuned base model as a zero-shot reference.

Outputs: outputs/eval/<run_id>/<split>_predictions.jsonl, <split>_metrics.json

    python scripts/04_evaluate.py --splits validation test
    python scripts/04_evaluate.py --runs proposed_knapsack_b600_s42 --splits test
"""
import argparse
import json

import _bootstrap  # noqa: F401

from hadl_eval import config
from hadl_eval.data import load_real_split
from hadl_eval.inference import evaluate_frame, load_eval_model
from hadl_eval.modeling import free_memory


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["validation", "test"], choices=["validation", "test"])
    ap.add_argument("--runs", nargs="*", default=None, help="run ids under outputs/runs (default: all)")
    ap.add_argument("--no_base", action="store_true", help="skip the zero-shot base model")
    ap.add_argument("--model_id", default=config.MODEL_ID)
    ap.add_argument("--tokenizer_id", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--batch_size", type=int, default=config.EVAL_BATCH_SIZE)
    ap.add_argument("--max_input_tokens", type=int, default=config.EVAL_MAX_INPUT_TOKENS)
    ap.add_argument("--max_new_tokens", type=int, default=config.EVAL_MAX_NEW_TOKENS)
    ap.add_argument("--max_eval_samples", type=int, default=-1, help="debug: first N examples per split")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--shard", default=None, help="'i/n': evaluate every n-th model starting at i (one process per GPU)")
    args = ap.parse_args()

    if args.runs is None:
        runs = sorted(p.parent.parent.name for p in config.RUNS_DIR.glob("*/adapter/adapter_config.json"))
    else:
        runs = args.runs
    targets = ([] if args.no_base else [("base_zero_shot", None)]) + \
              [(r, config.RUNS_DIR / r / "adapter") for r in runs]
    if not targets:
        raise SystemExit("Nothing to evaluate: no adapters in outputs/runs and --no_base given")

    splits = {}
    for s in args.splits:
        df = load_real_split(s)
        splits[s] = df.head(args.max_eval_samples) if args.max_eval_samples > 0 else df

    shard_i, shard_n = config.parse_shard(args.shard)
    targets = targets[shard_i::shard_n]
    print(f"{len(targets)} model(s) to evaluate in shard {shard_i}/{shard_n}: {[t[0] for t in targets]}")
    for run_id, adapter in targets:
        todo = [s for s in args.splits
                if args.overwrite or not (config.EVAL_DIR / run_id / f"{s}_metrics.json").is_file()]
        if not todo:
            print(f"[skip] {run_id}: already evaluated")
            continue
        print(f"\n=== Evaluating {run_id} on {todo}")
        model, tok, device, device_name = load_eval_model(args.model_id, adapter, args.dtype, args.tokenizer_id)
        for split in todo:
            summary = evaluate_frame(model, tok, device, splits[split], run_id, split, config.EVAL_DIR / run_id,
                                     n_boot=args.n_boot, batch_size=args.batch_size,
                                     max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens)
            summary["device_name"] = device_name
            (config.EVAL_DIR / run_id / f"{split}_metrics.json").write_text(json.dumps(summary, indent=2))
            print(f"  {split}: EM {summary['exact_match']:.2f}  F1 {summary['token_f1']:.2f}  "
                  f"charF1 {summary['char_f1']:.2f}  ({summary['num_examples']} ex, {summary['generation_seconds']:.0f}s)")
        del model
        free_memory()


if __name__ == "__main__":
    main()
