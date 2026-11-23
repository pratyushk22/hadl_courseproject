"""Loading the real and synthetic data into one common schema.

Common columns:
    example_id   unique id ("real:<review_id>" or "syn:<_source_row_idx>")
    source       "real" | "synthetic"
    passage_id   group id (passage for real data, wiki article for synthetic data)
    passage      context paragraph
    question
    answer       single training target (cleaned)
    answers      list of acceptable gold answers (for evaluation)

The teammates' files use different column names (real: passage/answers[list],
synthetic: paragraph/answer[str]); normalising them here removes a whole class of
silent bugs (e.g. reading a non-existent "context" column returns an empty passage).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .text_utils import clean_answer_target

COMMON_COLUMNS = ["example_id", "source", "passage_id", "passage", "question", "answer", "answers"]
REAL_SPLITS = ("train", "validation", "test")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def _as_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):                       # SQuAD style {"text": [...]}
        return [str(t) for t in x.get("text", [])]
    try:
        return [str(t) for t in list(x)]
    except TypeError:
        return [str(x)]


def standardize_real(df: pd.DataFrame) -> pd.DataFrame:
    answers = df["answers"].map(lambda a: [s.strip() for s in _as_list(a) if str(s).strip()])
    out = pd.DataFrame({
        "example_id": "real:" + df["review_id"].astype(str),
        "source": "real",
        "passage_id": df["passage_id"].astype(str),
        "passage": df["passage"].astype(str),
        "question": df["question"].astype(str),
        "answer": answers.map(lambda a: clean_answer_target(a[0]) if a else ""),
        "answers": answers,
    })
    return out.reset_index(drop=True)


def load_real_split(split: str, real_dir: str | Path = config.REAL_DIR) -> pd.DataFrame:
    if split not in REAL_SPLITS:
        raise ValueError(f"split must be one of {REAL_SPLITS}, got {split!r}")
    return standardize_real(pd.read_json(Path(real_dir) / f"{split}.jsonl", lines=True))


def standardize_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps every extra column (scores, costs, ...) and adds the common columns."""
    out = df.copy()
    passage_col = "paragraph" if "paragraph" in out.columns else ("passage" if "passage" in out.columns else "context")
    if passage_col not in out.columns:
        raise KeyError("synthetic data needs a 'paragraph', 'passage' or 'context' column")
    if "_source_row_idx" not in out.columns:
        raise KeyError("synthetic data needs '_source_row_idx' (row index into or_synthetic_filtered.parquet)")
    group_col = "_wiki_group" if "_wiki_group" in out.columns else ("wiki_id" if "wiki_id" in out.columns else None)
    raw_answer = out["answer"].astype(str)
    out["example_id"] = "syn:" + out["_source_row_idx"].astype(str)
    out["source"] = "synthetic"
    out["passage_id"] = ("wiki:" + out[group_col].astype(str)) if group_col else out["example_id"]
    out["passage"] = out[passage_col].astype(str)
    out["question"] = out["question"].astype(str)
    out["answer"] = raw_answer.map(clean_answer_target)
    out["answers"] = out["answer"].map(lambda a: [a])
    return out


def training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Only the columns needed to train, with empty targets removed."""
    df = df[COMMON_COLUMNS].copy()
    df = df[(df["answer"].str.len() > 0) & (df["passage"].str.len() > 0) & (df["question"].str.len() > 0)]
    return df.reset_index(drop=True)
