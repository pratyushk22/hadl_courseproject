"""Prompt construction shared by training, time-cost estimation and evaluation.

Using one function everywhere guarantees that the model is evaluated with exactly the
prompt format it was trained on (Qwen3 chat template, non-thinking mode).

Truncation rules
* evaluation: only the passage is shortened, from the end; question, system prompt and the
  assistant header are never cut (the gold answer is never used to choose the window).
* training: if the passage is too long, a token window that contains the answer is kept
  (standard extractive-QA practice), so the supervision target stays answerable.
"""
from __future__ import annotations

from pathlib import Path

from . import config

_TOKENIZER_CACHE: dict = {}


def load_tokenizer(model_id: str = config.MODEL_ID, padding_side: str = "right"):
    """Load the Qwen3 tokenizer; fall back to the copy committed in the repo when offline."""
    from transformers import AutoTokenizer

    key = (model_id, padding_side)
    if key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[key]
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # network blocked / not cached
        fallback = Path(config.LOCAL_TOKENIZER_FALLBACK)
        if model_id != config.MODEL_ID or not fallback.is_dir():
            raise
        print(f"[tokenizer] Could not load '{model_id}' ({type(exc).__name__}); "
              f"using the local Qwen3 tokenizer copy at {fallback}")
        tok = AutoTokenizer.from_pretrained(str(fallback))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    _TOKENIZER_CACHE[key] = tok
    return tok


_SPECIAL_CACHE: dict = {}


def _token_id(tokenizer, token: str | None):
    """Id of a single (special) token, or None if the tokenizer does not know it.
    (get_vocab() rebuilds a 150k-entry dict on every call, so it must not be used per row.)"""
    if token is None:
        return None
    key = (id(tokenizer), token)
    if key not in _SPECIAL_CACHE:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is None or (tokenizer.unk_token_id is not None and tid == tokenizer.unk_token_id):
            tid = None
        _SPECIAL_CACHE[key] = tid
    return _SPECIAL_CACHE[key]


def end_of_turn_token(tokenizer) -> str:
    """Token that closes an assistant turn in the chat template (<|im_end|> for Qwen3)."""
    return "<|im_end|>" if _token_id(tokenizer, "<|im_end|>") is not None else tokenizer.eos_token


def stop_token_ids(tokenizer) -> list[int]:
    ids = [_token_id(tokenizer, t) for t in ("<|im_end|>", tokenizer.eos_token, "<|endoftext|>")]
    return sorted({i for i in ids if i is not None})


def build_prompt_text(passage: str, question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": f"Passage:\n{passage}\n\nQuestion:\n{question}"},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:  # tokenizers whose template does not take enable_thinking
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _encode(text: str, tokenizer) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def encode_prompt(passage: str, question: str, tokenizer, max_tokens: int,
                  answer_hint: str | None = None) -> list[int]:
    """Prompt ids of length <= max_tokens. Only the passage is shortened.

    answer_hint (training only): keep a passage window containing this string.
    """
    passage = str(passage)
    question = str(question)
    ids = _encode(build_prompt_text(passage, question, tokenizer), tokenizer)
    if len(ids) <= max_tokens:
        return ids

    passage_ids = _encode(passage, tokenizer)
    overhead = len(ids) - len(passage_ids)
    start = 0
    for _ in range(4):  # decode/re-encode can shift the count by a few tokens; iterate
        keep = max(0, max_tokens - overhead - 2)
        if keep >= len(passage_ids):
            keep = max(0, len(passage_ids) - 1)
        if answer_hint:
            start = _answer_window_start(passage, passage_ids, answer_hint, keep, tokenizer)
        window = passage_ids[start:start + keep]
        # byte-level BPE windows can cut a multi-byte Odia character -> drop the U+FFFD stubs
        short_passage = tokenizer.decode(window, skip_special_tokens=True).replace("�", "")
        ids = _encode(build_prompt_text(short_passage, question, tokenizer), tokenizer)
        if len(ids) <= max_tokens:
            return ids
        overhead += len(ids) - max_tokens   # tighten and retry
    # Extremely long question: keep the tail (question end + assistant header).
    return ids[-max_tokens:]


def _answer_window_start(passage: str, passage_ids: list[int], answer: str, keep: int, tokenizer) -> int:
    """Start token index of a window of `keep` tokens that contains the answer (centred)."""
    char_pos = passage.find(answer)
    if char_pos < 0 or keep <= 0:
        return 0
    prefix_tokens = len(_encode(passage[:char_pos], tokenizer))
    answer_tokens = len(_encode(answer, tokenizer))
    centre = prefix_tokens + answer_tokens // 2
    start = max(0, centre - keep // 2)
    return min(start, max(0, len(passage_ids) - keep))


def build_training_features(passage: str, question: str, answer: str, tokenizer,
                            max_seq_len: int = config.MAX_SEQ_LEN,
                            max_answer_tokens: int = config.MAX_ANSWER_TOKENS) -> dict:
    """input_ids / attention_mask / labels with loss on the answer tokens only."""
    answer = str(answer).strip()
    eot = end_of_turn_token(tokenizer)
    answer_ids = _encode(answer + eot, tokenizer)
    if len(answer_ids) > max_answer_tokens:
        answer_ids = answer_ids[:max_answer_tokens - 1] + [_token_id(tokenizer, eot)]
    prompt_ids = encode_prompt(passage, question, tokenizer, max_tokens=max_seq_len - len(answer_ids),
                               answer_hint=answer)
    input_ids = prompt_ids + answer_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }
