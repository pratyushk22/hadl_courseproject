"""Central configuration: every path and default hyper-parameter lives here.

All paths are resolved relative to this file, so scripts work from any working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PIPELINE_ROOT = Path(__file__).resolve().parents[1]          # .../evaluation_pipeline
PROJECT_ROOT = PIPELINE_ROOT.parent                          # .../hadl_courseproject

# Inputs produced by the other team members
REAL_DIR = PROJECT_ROOT / "data_and_scripts" / "odia_real_cleaned"
SYNTH_FILTERED = PROJECT_ROOT / "data_and_scripts" / "odia_synth_cleaned" / "or_synthetic_filtered.parquet"
LR_TRAIN_POOL = PROJECT_ROOT / "low_rank_method" / "landmark_losses" / "train_85_percent.parquet"
LR_DRAWS = PROJECT_ROOT / "low_rank_method" / "odia_low_rank_run" / "final_weighted_train_draws.parquet"
CM_OUTPUT_DIR = PROJECT_ROOT / "covariance_matching" / "outputs"
TIME_PROXY_JSON = PROJECT_ROOT / "performance_profiling" / "time_proxy_model.json"
PROFILING_CSV = Path(os.environ.get("HADL_PROFILING_CSV",
                                    PROJECT_ROOT / "performance_profiling" / "profiling_results.csv"))

# Optional (git-ignored *.pt files - only present on the machine that created them)
SYNTH_EMBEDDINGS_PT = PROJECT_ROOT / "embedding_generation" / "odia_embeddings.pt"
REAL_TRAIN_EMBEDDINGS_PT = PROJECT_ROOT / "embedding_generation" / "odia_real_train_embeddings.pt"
LR_POOL_EMBEDDINGS_PT = PROJECT_ROOT / "low_rank_method" / "odia_low_rank_run" / "final_weighted_train_draws_embeddings.pt"

# The *.pt embedding files are git-ignored; they live in the team Google Drive folder and are
# downloaded on demand (gdown) by the covariance diagnostic. Drive file ids per local path:
TEAM_DRIVE_FOLDER = "https://drive.google.com/drive/folders/11sPbsrO8gDA4w0HTX4zh_WlH3ocpadjd"
DRIVE_FILE_IDS = {
    SYNTH_EMBEDDINGS_PT: "1sXuHpqZ-M-_5QHjhSsCRqevTlfpoZBGZ",      # embedding_generation/odia_embeddings.pt
    LR_POOL_EMBEDDINGS_PT: "1x_53L13_aqcVgoHSligzWMbOW7zhcDfJ",    # low_rank_method/odia_low_rank_run/...
}

# A copy of the Qwen3 tokenizer is committed with the teammates' adapters; it is used as
# an offline fallback when the Hugging Face Hub is unreachable.
LOCAL_TOKENIZER_FALLBACK = PROJECT_ROOT / "model_training" / "checkpoints" / "proposed_knapsack" / "final_adapter"

# Outputs of this pipeline
# HADL_OUTPUT_ROOT lets tests / experiments write somewhere else without touching real results.
OUTPUT_ROOT = Path(os.environ.get("HADL_OUTPUT_ROOT", PIPELINE_ROOT / "outputs")).resolve()
POOL_DIR = OUTPUT_ROOT / "pool"
SELECTION_DIR = OUTPUT_ROOT / "selections"
RUNS_DIR = OUTPUT_ROOT / "runs"
EVAL_DIR = OUTPUT_ROOT / "eval"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
OVERHEAD_JSON = PIPELINE_ROOT / "configs" / "overhead_times.json"

# --------------------------------------------------------------------------------------
# Model / training defaults (mandated by the project: Qwen3-1.7B, LoRA rank 8)
# --------------------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-1.7B"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
EPOCHS = 1.0
WARMUP_RATIO = 0.03
# Odia text costs ~2 Qwen tokens per character: the median synthetic training sequence is
# ~600 tokens and the median real IndicQA passage ~1,300 tokens. 2048 matches the cap used
# by the time proxy (performance_profiling/assign_time_costs.py).
MAX_SEQ_LEN = 2048
MAX_ANSWER_TOKENS = 128

# Evaluation defaults. Real test passages reach ~3,700 tokens, so 4096 keeps every passage
# intact (truncating would silently delete the answer span from the prompt).
EVAL_MAX_INPUT_TOKENS = 4096
EVAL_MAX_NEW_TOKENS = 128
EVAL_BATCH_SIZE = 8

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions in Odia based strictly on the "
    "provided passage. Reply with only the short answer span copied from the passage."
)

# --------------------------------------------------------------------------------------
# Selection defaults
# --------------------------------------------------------------------------------------
# The budget is "proxy-predicted synthetic training seconds per epoch". Absolute seconds depend
# on the GPU the proxy was profiled on, so by default the budget is expressed as a multiple of
# the proxy-predicted time of one epoch over the real IndicQA train split (886 rows):
# x = 1.0 means "the synthetic data may cost as much training time as the real data".
# Pass --budget_sec to scripts/02_select_subsets.py to use absolute seconds instead.
DEFAULT_BUDGET_X_REAL = [1.0]
BUDGET_SWEEP_X_REAL = [0.25, 0.5, 1.0, 2.0, 4.0]
DEFAULT_SEEDS = [42]
DEFAULT_ALPHA = 0.5                  # weight of the low-rank score in the composite utility
RANDOM_POOL_SIZE = 30_000            # rows of the 185k synthetic train pool costed for the random baseline
BUDGETS_JSON_NAME = "budgets.json"   # written by step 2, read by steps 3, 5, 6

# Methods. The first three are the baselines named in the mid-project report.
METHODS = ["random", "quality_only", "proposed_knapsack", "low_rank_only", "cov_match_only"]
HEADLINE_METHODS = ["random", "quality_only", "proposed_knapsack"]
REFERENCE_RUNS = ["base_zero_shot", "real_only"]

METHOD_LABELS = {
    "base_zero_shot": "Base model (zero-shot)",
    "real_only": "Real data only",
    "random": "Random selection",
    "low_rank_only": "Low-rank only",
    "cov_match_only": "Covariance matching only",
    "quality_only": "LR + CM (no time proxy)",
    "proposed_knapsack": "Proposed: LR + CM + time knapsack",
}

# Fixed method -> colour mapping (colour follows the entity, never its rank).
# Categorical slots of the validated reference palette, in fixed order.
METHOD_COLORS = {
    "random": "#2a78d6",            # blue
    "quality_only": "#eb6834",      # orange
    "proposed_knapsack": "#1baf7a", # aqua
    "low_rank_only": "#eda100",     # yellow
    "cov_match_only": "#e87ba4",    # magenta
    "real_only": "#6250d6",         # violet
    "base_zero_shot": "#8a8984",    # neutral grey (reference, not a competing series)
}


def method_of_run(run_id: str) -> str:
    """'proposed_knapsack_b600_s42' -> 'proposed_knapsack'."""
    for m in sorted(list(METHOD_LABELS), key=len, reverse=True):
        if run_id == m or run_id.startswith(m + "_"):
            return m
    return run_id


def run_id_for(method: str, budget_sec: float | None, seed: int) -> str:
    if method in REFERENCE_RUNS:
        return method if method == "base_zero_shot" else f"{method}_s{seed}"
    return f"{method}_b{budget_tag(budget_sec)}_s{seed}"


def resolved_budgets(cli_budgets: list[float] | None = None) -> list[float]:
    """Budgets chosen in step 2 (outputs/selections/budgets.json) unless given explicitly."""
    import json

    if cli_budgets:
        return [float(b) for b in cli_budgets]
    path = SELECTION_DIR / BUDGETS_JSON_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found - run scripts/02_select_subsets.py first (or pass --budget_sec)")
    return [float(b) for b in json.loads(path.read_text())["budgets_sec"]]


def budget_tag(budget_sec: float | None) -> str:
    if budget_sec is None:
        return "none"
    return str(int(budget_sec)) if float(budget_sec).is_integer() else str(budget_sec).replace(".", "p")


def parse_shard(spec: str | None) -> tuple[int, int]:
    """'0/2' -> (0, 2). Lets several single-GPU processes split the job list (one per GPU)."""
    if not spec:
        return 0, 1
    try:
        i, n = (int(x) for x in spec.split("/"))
    except ValueError as exc:
        raise ValueError(f"--shard must look like 'i/n', got {spec!r}") from exc
    if not 0 <= i < n:
        raise ValueError(f"--shard index must be in [0, n), got {spec!r}")
    return i, n
