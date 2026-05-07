"""Tests for Phase 2 adversarial search."""

import pytest
from diverge.search import (
    find_divergence_inputs,
    AdversarialSearchResult,
    SearchResult,
    DivergenceCandidate,
)
from diverge.match import exact_match, letter_match

PROMPTS = [
    "What is the capital of France?",
    "How many planets are in the solar system?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water?",
]


# ---------------------------------------------------------------------------
# DivergenceCandidate properties
# ---------------------------------------------------------------------------

def _make_candidate(agreed_original, agreed_perturbed):
    return DivergenceCandidate(
        original="original", perturbed="perturbed",
        perturbation_type="typo", level=1,
        a_original="A", b_original="A" if agreed_original else "B",
        a_perturbed="A", b_perturbed="A" if agreed_perturbed else "B",
        agreed_original=agreed_original,
        agreed_perturbed=agreed_perturbed,
    )


def test_is_new_divergence_true():
    c = _make_candidate(agreed_original=True, agreed_perturbed=False)
    assert c.is_new_divergence is True


def test_is_new_divergence_false_when_already_disagreed():
    c = _make_candidate(agreed_original=False, agreed_perturbed=False)
    assert c.is_new_divergence is False


def test_is_new_agreement():
    c = _make_candidate(agreed_original=False, agreed_perturbed=True)
    assert c.is_new_agreement is True


# ---------------------------------------------------------------------------
# SearchResult.mdd
# ---------------------------------------------------------------------------

def test_search_result_mdd_none_when_no_divergence():
    sr = SearchResult("prompt", "A", "A", True)
    assert sr.mdd is None


def test_search_result_mdd_minimum_level():
    sr = SearchResult("prompt", "A", "A", True)
    sr.candidates = [
        _make_candidate(True, True),  # no divergence
        _make_candidate(True, False),  # divergence at level 1 (default)
    ]
    sr.candidates[1].level = 2
    # Only one divergence at level 2
    assert sr.mdd == 2


def test_search_result_mdd_picks_minimum():
    sr = SearchResult("prompt", "A", "A", True)
    c1 = _make_candidate(True, False)
    c1.level = 3
    c2 = _make_candidate(True, False)
    c2.level = 1
    sr.candidates = [c1, c2]
    assert sr.mdd == 1


# ---------------------------------------------------------------------------
# find_divergence_inputs — stable model (always agree)
# ---------------------------------------------------------------------------

def test_stable_models_no_divergence():
    result = find_divergence_inputs(
        lambda p: "Paris",
        lambda p: "Paris",
        PROMPTS,
        strategy="mdd",
    )
    assert isinstance(result, AdversarialSearchResult)
    assert result.divergence_found_rate == 0.0
    assert result.mean_mdd is None


def test_stable_models_mdd_none():
    result = find_divergence_inputs(
        lambda p: "same answer",
        lambda p: "same answer",
        PROMPTS[:2],
        strategy="mdd",
    )
    for r in result.results:
        assert r.mdd is None


# ---------------------------------------------------------------------------
# find_divergence_inputs — models that differ based on input noise
# ---------------------------------------------------------------------------

def test_sensitive_model_finds_divergence():
    """Model B flips answer on ANY input that's not exactly the original."""
    originals = set(PROMPTS)

    def model_a(p):
        return "Paris"

    def model_b(p):
        return "Paris" if p in originals else "London"

    result = find_divergence_inputs(
        model_a, model_b, PROMPTS[:2],
        strategy="mdd",
        match_fn=exact_match,
    )
    assert result.divergence_found_rate > 0.0
    assert result.mean_mdd is not None


def test_mdd_level_1_detected():
    """Model B changes answer when there's a typo."""
    originals = set(PROMPTS)

    def model_b(p):
        return "Paris" if p in originals else "London"

    result = find_divergence_inputs(
        lambda p: "Paris",
        model_b,
        PROMPTS[:2],
        strategy="mdd",
        perturbation_types=["typo"],
        match_fn=exact_match,
    )
    found = [r for r in result.results if r.mdd is not None]
    if found:
        assert all(r.mdd == 1 for r in found)


# ---------------------------------------------------------------------------
# AdversarialSearchResult properties
# ---------------------------------------------------------------------------

def test_result_shape():
    result = find_divergence_inputs(
        lambda p: "A", lambda p: "A", PROMPTS, strategy="mdd",
    )
    assert len(result.results) == len(PROMPTS)
    assert isinstance(result.results[0], SearchResult)


def test_mdd_distribution_keys():
    result = find_divergence_inputs(
        lambda p: "A", lambda p: "A", PROMPTS, strategy="mdd",
    )
    dist = result.mdd_distribution
    assert set(dist.keys()) == {"1", "2", "3", "none"}


def test_mdd_distribution_sums_to_n_prompts():
    result = find_divergence_inputs(
        lambda p: "A", lambda p: "B", PROMPTS[:2],
        strategy="mdd",
        match_fn=exact_match,
    )
    dist = result.mdd_distribution
    assert sum(dist.values()) == len(PROMPTS[:2])


def test_top_divergences_are_new_divergences():
    originals = set(PROMPTS)
    result = find_divergence_inputs(
        lambda p: "Paris",
        lambda p: "Paris" if p in originals else "London",
        PROMPTS[:2],
        strategy="mdd",
        match_fn=exact_match,
    )
    for c in result.top_divergences():
        assert c.is_new_divergence


def test_model_names_stored():
    result = find_divergence_inputs(
        lambda p: "A", lambda p: "A", PROMPTS[:1],
        model_a_name="GPT", model_b_name="Claude",
        strategy="mdd",
    )
    assert result.model_a_name == "GPT"
    assert result.model_b_name == "Claude"


# ---------------------------------------------------------------------------
# Beam strategy
# ---------------------------------------------------------------------------

def test_beam_strategy_runs():
    result = find_divergence_inputs(
        lambda p: "A", lambda p: "A", PROMPTS[:2],
        strategy="beam",
        n_per_type=1,
        beam_width=2,
    )
    assert isinstance(result, AdversarialSearchResult)
    assert result.strategy == "beam"


def test_beam_finds_divergence_for_sensitive_model():
    originals = set(PROMPTS)
    result = find_divergence_inputs(
        lambda p: "Paris",
        lambda p: "Paris" if p in originals else "London",
        PROMPTS[:2],
        strategy="beam",
        match_fn=exact_match,
        n_per_type=2,
        beam_width=3,
    )
    assert result.divergence_found_rate > 0.0


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        find_divergence_inputs(lambda p: "A", lambda p: "A", PROMPTS[:1], strategy="unknown")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_report_runs():
    result = find_divergence_inputs(
        lambda p: "A", lambda p: "A", PROMPTS[:2], strategy="mdd",
    )
    report = result.report()
    assert "Adversarial Search Report" in report
    assert "MDD" in report


def test_report_shows_model_names():
    result = find_divergence_inputs(
        lambda p: "A", lambda p: "A", PROMPTS[:1],
        model_a_name="GPT", model_b_name="Claude",
        strategy="mdd",
    )
    report = result.report()
    assert "GPT" in report
    assert "Claude" in report


def test_report_no_divergence_message():
    result = find_divergence_inputs(
        lambda p: "same", lambda p: "same", PROMPTS[:2], strategy="mdd",
    )
    report = result.report()
    assert "N/A" in report or "no divergences" in report.lower()


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

def test_importable_from_top_level():
    from diverge import find_divergence_inputs as fdi, AdversarialSearchResult as ASR
    assert callable(fdi)
    assert ASR is AdversarialSearchResult
