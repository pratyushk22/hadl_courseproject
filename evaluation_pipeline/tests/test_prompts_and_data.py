
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hadl_eval import config  # noqa: E402
from hadl_eval.data import load_real_split, standardize_synthetic, training_frame  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tok():
    from hadl_eval.prompts import load_tokenizer
    try:
        return load_tokenizer()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"tokenizer unavailable: {exc}")


def test_real_split_schema():
    df = load_real_split("test")
    assert len(df) == 176
    assert set(["example_id", "passage", "question", "answer", "answers", "passage_id"]) <= set(df.columns)
    assert df["answers"].map(len).min() >= 1
    assert all(a in p for p, a in zip(df["passage"], df["answer"]))   # IndicQA answers are extractive


def test_synthetic_schema_uses_paragraph():
    import pandas as pd
    raw = pd.read_parquet(config.LR_DRAWS).head(20)
    df = standardize_synthetic(raw)
    assert (df["passage"] == raw["paragraph"]).all()      # not an empty 'context' column
    assert not df["answer"].str.endswith("।").any()       # trailing danda stripped
    assert len(training_frame(df)) == 20


def test_training_features_keep_answer_and_question(tok):
    from hadl_eval.prompts import build_training_features
    df = load_real_split("train")
    row = df.loc[df["passage"].str.len().idxmax()]           # longest passage -> forces truncation
    f = build_training_features(row["passage"], row["question"], row["answer"], tok, max_seq_len=512)
    assert len(f["input_ids"]) <= 512
    text = tok.decode(f["input_ids"])
    assert row["question"] in text                            # question never truncated
    assert text.rstrip().endswith("<|im_end|>")
    target = tok.decode([t for t in f["labels"] if t != -100])
    assert target.replace("<|im_end|>", "").strip() == row["answer"]
    prompt_part = tok.decode([t for t, l in zip(f["input_ids"], f["labels"]) if l == -100])
    assert row["answer"] in prompt_part                       # answer-containing window kept
    assert "<think>\n\n</think>" in prompt_part               # Qwen3 non-thinking mode


def test_eval_prompt_truncates_only_passage(tok):
    from hadl_eval.prompts import encode_prompt
    df = load_real_split("test")
    row = df.loc[df["passage"].str.len().idxmax()]
    ids = encode_prompt(row["passage"], row["question"], tok, max_tokens=300)
    assert len(ids) <= 300
    text = tok.decode(ids)
    assert row["question"] in text and text.endswith("<think>\n\n</think>\n\n")
    full = encode_prompt(row["passage"], row["question"], tok, max_tokens=config.EVAL_MAX_INPUT_TOKENS)
    assert row["passage"] in tok.decode(full)                 # default eval budget keeps the full passage
