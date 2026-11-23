"""Odia-aware text normalisation shared by data loading, metrics and post-processing."""
from __future__ import annotations

import re
import unicodedata

# Zero-width characters are common in Odia Wikipedia text (e.g. ZWNJ inside conjuncts) and
# carry no meaning for answer matching.
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)

# Odia (U+0B66..U+0B6F) and Devanagari (U+0966..U+096F) digits -> ASCII digits, so that
# "୨୦୧୪" and "2014" compare equal.
_DIGITS = {}
for _base in (0x0B66, 0x0966):
    for _i in range(10):
        _DIGITS[_base + _i] = str(_i)

_WS = re.compile(r"\s+")
_TRAILING_PUNCT = "।॥.,;:!?'\"“”‘’ \t\n"


def _strip_punctuation(text: str) -> str:
    # Remove every Unicode punctuation (P*) and symbol (S*) character, incl. danda "।" (Po).
    return "".join(ch if not unicodedata.category(ch)[0] in ("P", "S") else " " for ch in text)


def normalize_answer(text) -> str:
    """Normalisation used by EM / F1 (SQuAD-style, adapted to Odia).

    NFC -> drop zero-width chars -> unify digits -> lowercase (Latin) -> remove punctuation
    and symbols -> collapse whitespace.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.translate(_ZERO_WIDTH).translate(_DIGITS).lower()
    text = _strip_punctuation(text)
    return _WS.sub(" ", text).strip()


def clean_answer_target(text) -> str:
    """Training target cleaning: synthetic answers always end with a danda ("।"), real
    IndicQA answers are bare extractive spans. Strip trailing sentence punctuation so the
    model learns span-style answers."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text)).strip()
    return text.rstrip(_TRAILING_PUNCT).strip()


def clean_generation(text: str) -> str:
    """Post-process a generated answer: drop any leaked think block / role tags, keep the
    first non-empty line and strip trailing punctuation."""
    if text is None:
        return ""
    text = re.sub(r"<think>.*?</think>", " ", str(text), flags=re.S)
    text = text.replace("<think>", " ").replace("</think>", " ")
    for prefix in ("Answer:", "answer:", "ଉତ୍ତର:", "ଉତ୍ତର :"):
        text = text.strip()
        if text.startswith(prefix):
            text = text[len(prefix):]
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return clean_answer_target(lines[0]) if lines else ""


def word_shingles(text: str, n: int = 8) -> set[str]:
    """Set of n-word shingles of the normalised text (used for contamination checks)."""
    words = normalize_answer(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}
