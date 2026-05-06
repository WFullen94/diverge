"""Answer matching strategies for comparing model outputs.

Four strategies with increasing tolerance:
  exact      — case-insensitive equality
  normalized — strip punctuation + whitespace, lowercase
  letter     — extract first A/B/C/D letter (for multiple-choice)
  semantic   — cosine similarity via sentence-transformers (optional dep)

All functions follow the signature (a: str, b: str) -> bool so they drop
directly into diverge(match_fn=...).
"""

from __future__ import annotations

import re
import string
from typing import Callable


def exact_match(a: str, b: str) -> bool:
    """Case-insensitive exact match."""
    return a.strip().lower() == b.strip().lower()


def normalized_match(a: str, b: str) -> bool:
    """Strip punctuation and normalize whitespace before comparing."""
    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = s.translate(str.maketrans("", "", string.punctuation))
        return " ".join(s.split())
    return _norm(a) == _norm(b)


def letter_match(a: str, b: str) -> bool:
    """Extract the first A/B/C/D letter from each answer and compare.

    Handles responses like 'A', 'A)', 'The answer is A', '(A) Paris', etc.
    Falls back to normalized_match if no letter is found in either response.
    """
    def _extract(s: str) -> str | None:
        s = s.strip()
        # Leading letter: "A", "A)", "A."
        m = re.match(r"^([A-Da-d])[).\s]", s)
        if m:
            return m.group(1).upper()
        # "answer is A" or "answer: A"
        m = re.search(r"\b(?:answer\s*(?:is|:)\s*)?([A-Da-d])\b", s, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Bare single letter
        if len(s) == 1 and s.upper() in "ABCD":
            return s.upper()
        return None

    la, lb = _extract(a), _extract(b)
    if la is None or lb is None:
        return normalized_match(a, b)
    return la == lb


def semantic_similarity(a: str, b: str) -> float:
    """Cosine similarity between sentence embeddings in [0, 1].

    Requires: pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError as e:
        raise ImportError("pip install sentence-transformers to use semantic matching") from e

    _model = _get_embed_model()
    vecs = _model.encode([a, b], normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))


def semantic_match(threshold: float = 0.85) -> Callable[[str, str], bool]:
    """Return a match function that agrees when cosine similarity >= threshold.

    Usage:
        diverge(model_a, model_b, prompts, match_fn=semantic_match(0.9))
    """
    def _match(a: str, b: str) -> bool:
        return semantic_similarity(a, b) >= threshold
    _match.__name__ = f"semantic_match(threshold={threshold})"
    return _match


_embed_model_cache: object = None

def _get_embed_model():
    global _embed_model_cache
    if _embed_model_cache is None:
        from sentence_transformers import SentenceTransformer
        _embed_model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model_cache
