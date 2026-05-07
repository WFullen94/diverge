"""Tests for perturbation generators."""

import pytest
import random
from diverge.perturb import (
    inject_typo, format_variant, synonym_swap, reorder_words,
    perturb, PerturbedPrompt, PERTURBATION_LEVELS,
)

TEXT = "What is the best approach to solve this difficult problem?"


def test_inject_typo_changes_text():
    rng = random.Random(0)
    result = inject_typo("hello world", rng, n=2)
    assert result != "hello world"


def test_inject_typo_preserves_length_approx():
    rng = random.Random(0)
    original = "the quick brown fox"
    result = inject_typo(original, rng, n=1)
    assert abs(len(result) - len(original)) <= 2


def test_format_variant_changes_text():
    rng = random.Random(0)
    result = format_variant(TEXT, rng)
    assert result != TEXT or result == TEXT.upper()


def test_synonym_swap_replaces_known_word():
    rng = random.Random(0)
    result = synonym_swap("What is the best approach?", rng)
    assert result != "What is the best approach?"


def test_synonym_swap_no_known_words():
    rng = random.Random(0)
    result = synonym_swap("xyzzy qwerty asdfgh", rng)
    assert result == "xyzzy qwerty asdfgh"


def test_reorder_preserves_words():
    rng = random.Random(42)
    original = "the quick brown fox jumped over"
    result = reorder_words(original, rng)
    assert sorted(result.split()) == sorted(original.split())


# ---------------------------------------------------------------------------
# perturb()
# ---------------------------------------------------------------------------

def test_perturb_returns_list_of_perturbed_prompts():
    results = perturb(TEXT, types=["typo"])
    assert all(isinstance(r, PerturbedPrompt) for r in results)


def test_perturb_correct_count():
    results = perturb(TEXT, types=["typo", "synonym"], n_per_type=2)
    assert len(results) == 4  # 2 types × 2 per type


def test_perturb_levels_correct():
    results = perturb(TEXT, types=["typo", "synonym", "reorder"], n_per_type=1)
    level_map = {r.perturbation_type: r.level for r in results}
    assert level_map["typo"] == 1
    assert level_map["synonym"] == 2
    assert level_map["reorder"] == 3


def test_perturb_all_types_default():
    results = perturb(TEXT)
    types_produced = {r.perturbation_type for r in results}
    assert types_produced == set(PERTURBATION_LEVELS.keys())


def test_perturb_reproducible():
    r1 = perturb(TEXT, seed=42)
    r2 = perturb(TEXT, seed=42)
    assert [r.perturbed for r in r1] == [r.perturbed for r in r2]


def test_perturb_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown perturbation types"):
        perturb(TEXT, types=["nonexistent"])


def test_perturb_original_stored():
    results = perturb(TEXT, types=["typo"])
    assert all(r.original == TEXT for r in results)
