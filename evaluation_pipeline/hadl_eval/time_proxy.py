"""Training-time proxy (Yatika's fitted model) and per-example cost computation.

T_step(B, L) [ms] = b0 + bB*B + bBL*B*L + bBL2*B*L^2       (time_proxy_model.json)

Per-example cost at the training batch size B:  c(L) = T_step(B, L) / B.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import config
from .prompts import build_training_features


@dataclass
class TimeProxy:
    beta_0: float
    beta_B: float
    beta_BL: float
    beta_BL2: float
    max_seq_len: int = config.MAX_SEQ_LEN
    source: str = ""
    gpu_name: str = ""

    @classmethod
    def from_json(cls, path: str | Path = config.TIME_PROXY_JSON, max_seq_len: int = config.MAX_SEQ_LEN) -> "TimeProxy":
        meta = json.loads(Path(path).read_text())
        c = meta["coefficients"]
        return cls(float(c["beta_0"]), float(c["beta_B"]), float(c["beta_BL"]), float(c["beta_BL2"]),
                   max_seq_len=max_seq_len, source=str(path), gpu_name=str(meta.get("gpu_name", "")))

    def step_ms(self, batch_size, seq_len):
        B = np.asarray(batch_size, dtype=np.float64)
        L = np.minimum(np.asarray(seq_len, dtype=np.float64), self.max_seq_len)
        return self.beta_0 + self.beta_B * B + self.beta_BL * B * L + self.beta_BL2 * B * L ** 2

    def per_sample_ms(self, seq_len, batch_size: int = config.BATCH_SIZE):
        return self.step_ms(batch_size, seq_len) / float(batch_size)

    def padded_batches_ms(self, lengths: Sequence[int], batch_size: int = config.BATCH_SIZE,
                          pad_to_multiple_of: int = 8) -> float:
        """Prediction that accounts for padding when the examples are batched in the given order."""
        lengths = np.asarray(lengths, dtype=np.int64)
        total = 0.0
        for s in range(0, len(lengths), batch_size):
            chunk = lengths[s:s + batch_size]
            L = int(np.ceil(chunk.max() / pad_to_multiple_of) * pad_to_multiple_of)
            total += float(self.step_ms(len(chunk), L))
        return total


def compute_sequence_lengths(df: pd.DataFrame, tokenizer, max_seq_len: int = config.MAX_SEQ_LEN,
                             desc: str = "token lengths") -> np.ndarray:
    """Exact training-sequence length (prompt + answer, after truncation) of every row.

    Needs the common columns (passage, question, answer) - see data.standardize_*.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **_):
            return x
    lengths = np.empty(len(df), dtype=np.int64)
    for i, (p, q, a) in enumerate(tqdm(zip(df["passage"], df["question"], df["answer"]), total=len(df),
                                       desc=desc, mininterval=5)):
        lengths[i] = len(build_training_features(p, q, a, tokenizer, max_seq_len=max_seq_len)["input_ids"])
    return lengths


def annotate_costs(df: pd.DataFrame, tokenizer, proxy: TimeProxy, batch_size: int = config.BATCH_SIZE,
                   max_seq_len: int = config.MAX_SEQ_LEN, desc: str = "token lengths") -> tuple[pd.DataFrame, float]:
    """Adds token_length and predicted_time_cost_ms. Returns (df, seconds spent)."""
    t0 = time.perf_counter()
    df = df.copy()
    df["token_length"] = compute_sequence_lengths(df, tokenizer, max_seq_len=max_seq_len, desc=desc)
    df["predicted_time_cost_ms"] = proxy.per_sample_ms(df["token_length"].values, batch_size=batch_size)
    return df, time.perf_counter() - t0
