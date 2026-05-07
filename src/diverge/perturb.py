"""Prompt perturbation generators for adversarial search.

Four types at three semantic levels:
  Level 1 — surface:    typo, format  (spelling/spacing/punctuation)
  Level 2 — lexical:    synonym       (word-level substitution)
  Level 3 — structural: reorder       (clause-level word shuffle)

Each generator is deterministic given an rng, so searches are reproducible.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


PERTURBATION_LEVELS: dict[str, int] = {
    "typo":    1,
    "format":  1,
    "synonym": 2,
    "reorder": 3,
}

_SYNONYMS: dict[str, list[str]] = {
    "what": ["which", "what exactly"],
    "how": ["in what way", "by what means"],
    "why": ["for what reason", "what is the reason"],
    "where": ["in what place", "at what location"],
    "when": ["at what time", "on what occasion"],
    "is": ["are", "was"],
    "are": ["is", "were"],
    "big": ["large", "great", "significant"],
    "small": ["little", "tiny", "minor"],
    "good": ["beneficial", "positive", "favorable"],
    "bad": ["poor", "negative", "unfavorable"],
    "true": ["correct", "accurate", "right"],
    "false": ["incorrect", "wrong", "inaccurate"],
    "fast": ["quick", "rapid", "swift"],
    "slow": ["gradual", "unhurried", "leisurely"],
    "important": ["significant", "crucial", "key"],
    "difficult": ["challenging", "hard", "complex"],
    "easy": ["simple", "straightforward", "uncomplicated"],
    "many": ["numerous", "multiple", "several"],
    "few": ["several", "some", "a number of"],
    "first": ["primary", "initial", "earliest"],
    "last": ["final", "ultimate", "concluding"],
    "best": ["optimal", "top", "finest"],
    "worst": ["poorest", "lowest", "least effective"],
    "show": ["demonstrate", "illustrate", "reveal"],
    "use": ["utilize", "employ", "apply"],
    "make": ["create", "produce", "generate"],
    "get": ["obtain", "acquire", "receive"],
    "give": ["provide", "offer", "supply"],
    "find": ["discover", "identify", "locate"],
    "know": ["understand", "recognize", "be aware"],
    "think": ["believe", "consider", "suppose"],
    "say": ["state", "mention", "indicate"],
    "call": ["name", "refer to", "term"],
}

_KEYBOARD_NEIGHBORS: dict[str, str] = {
    "a": "sqwz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wrsdf",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "uojkl", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhij", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}


@dataclass
class PerturbedPrompt:
    original: str
    perturbed: str
    perturbation_type: str
    level: int


def inject_typo(text: str, rng: random.Random, n: int = 1) -> str:
    chars = list(text)
    indices = [i for i, c in enumerate(chars) if c.isalpha()]
    if not indices:
        return text
    for _ in range(min(n, len(indices))):
        idx = rng.choice(indices)
        c = chars[idx].lower()
        op = rng.choice(["swap", "sub", "delete", "insert"])
        if op == "swap" and idx + 1 < len(chars) and chars[idx + 1].isalpha():
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif op == "sub" and c in _KEYBOARD_NEIGHBORS:
            repl = rng.choice(_KEYBOARD_NEIGHBORS[c])
            chars[idx] = repl if chars[idx].islower() else repl.upper()
        elif op == "delete":
            chars[idx] = ""
        elif op == "insert" and c in _KEYBOARD_NEIGHBORS:
            chars.insert(idx, rng.choice(_KEYBOARD_NEIGHBORS[c]))
    return "".join(chars)


def format_variant(text: str, rng: random.Random) -> str:
    variants = [
        text.upper(),
        text.lower(),
        text.rstrip("?") + ".",
        text.rstrip(".") + "?",
        "  " + text + "  ",
        text.replace(",", " ,").replace(".", " ."),
    ]
    result = rng.choice(variants)
    return result if result != text else text.upper()


def synonym_swap(text: str, rng: random.Random) -> str:
    words = text.split()
    candidates = [(i, w) for i, w in enumerate(words) if w.lower().rstrip("?,. ") in _SYNONYMS]
    if not candidates:
        return text
    idx, word = rng.choice(candidates)
    clean = word.lower().rstrip("?,. ")
    replacement = rng.choice(_SYNONYMS[clean])
    punct = word[len(clean):]
    words[idx] = replacement + punct
    return " ".join(words)


def reorder_words(text: str, rng: random.Random) -> str:
    clauses = re.split(r"([,;])", text)
    result = []
    for clause in clauses:
        if clause in (",", ";"):
            result.append(clause)
            continue
        words = clause.split()
        if len(words) <= 2:
            result.append(clause)
            continue
        middle = words[1:-1]
        rng.shuffle(middle)
        result.append(" ".join([words[0]] + middle + [words[-1]]))
    return "".join(result)


def perturb(
    prompt: str,
    types: list[str] | None = None,
    n_per_type: int = 1,
    seed: int = 42,
) -> list[PerturbedPrompt]:
    """Generate perturbations for a single prompt.

    Args:
        prompt: The input string to perturb.
        types: Perturbation types to apply. Defaults to all four.
        n_per_type: Number of variants per type.
        seed: Random seed.

    Returns:
        List of PerturbedPrompt objects, one per (type × n_per_type).
    """
    if types is None:
        types = list(PERTURBATION_LEVELS.keys())

    unknown = [t for t in types if t not in PERTURBATION_LEVELS]
    if unknown:
        raise ValueError(f"Unknown perturbation types: {unknown}. "
                         f"Choose from {list(PERTURBATION_LEVELS.keys())}")

    rng = random.Random(seed)
    results: list[PerturbedPrompt] = []

    for ptype in types:
        for _ in range(n_per_type):
            if ptype == "typo":
                perturbed = inject_typo(prompt, rng, n=rng.randint(1, 3))
            elif ptype == "format":
                perturbed = format_variant(prompt, rng)
            elif ptype == "synonym":
                perturbed = synonym_swap(prompt, rng)
            elif ptype == "reorder":
                perturbed = reorder_words(prompt, rng)
            else:
                raise ValueError(f"Unknown perturbation type: {ptype!r}")

            results.append(PerturbedPrompt(
                original=prompt,
                perturbed=perturbed,
                perturbation_type=ptype,
                level=PERTURBATION_LEVELS[ptype],
            ))

    return results
