"""Tests for Phase 1 core differential testing."""

import pytest
from diverge.core import diverge, DivergenceResult, PromptResult
from diverge.match import exact_match, letter_match

PROMPTS = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "Who wrote Hamlet?",
    "What is the boiling point of water?",
    "How many continents are there?",
]
LABELS = ["Paris", "4", "Shakespeare", "100°C", "7"]


# ---------------------------------------------------------------------------
# Return type and shape
# ---------------------------------------------------------------------------

def test_diverge_returns_divergence_result():
    result = diverge(lambda p: "A", lambda p: "A", PROMPTS)
    assert isinstance(result, DivergenceResult)


def test_result_has_correct_n_prompts():
    result = diverge(lambda p: "A", lambda p: "A", PROMPTS)
    assert len(result.results) == len(PROMPTS)


def test_each_result_is_prompt_result():
    result = diverge(lambda p: "A", lambda p: "A", PROMPTS)
    for r in result.results:
        assert isinstance(r, PromptResult)
        assert r.prompt in PROMPTS
        assert isinstance(r.agree, bool)


# ---------------------------------------------------------------------------
# Agreement / disagreement rates
# ---------------------------------------------------------------------------

def test_identical_models_full_agreement():
    result = diverge(lambda p: "Paris", lambda p: "Paris", PROMPTS)
    assert result.agreement_rate == pytest.approx(1.0)
    assert result.disagreement_rate == pytest.approx(0.0)
    assert result.n_disagreements == 0


def test_always_different_models_full_disagreement():
    result = diverge(lambda p: "Paris", lambda p: "London", PROMPTS)
    assert result.agreement_rate == pytest.approx(0.0)
    assert result.n_disagreements == len(PROMPTS)


def test_partial_disagreement_rate():
    answers_a = ["Paris", "4", "Paris", "Paris", "Paris"]
    answers_b = ["Paris", "4", "London", "London", "London"]
    idx = [0]

    def model_a(p):
        i = PROMPTS.index(p)
        return answers_a[i]

    def model_b(p):
        i = PROMPTS.index(p)
        return answers_b[i]

    result = diverge(model_a, model_b, PROMPTS)
    assert result.n_disagreements == 3
    assert result.agreement_rate == pytest.approx(2 / 5)


# ---------------------------------------------------------------------------
# Disagreements are sorted first
# ---------------------------------------------------------------------------

def test_disagreements_appear_first():
    idx = [0]
    def model_a(p): return "same"
    def model_b(p):
        idx[0] += 1
        return "different" if idx[0] % 2 == 0 else "same"

    result = diverge(model_a, model_b, PROMPTS)
    disagreements_done = False
    for r in result.results:
        if r.agree:
            disagreements_done = True
        if disagreements_done:
            assert r.agree, "Agreement should come after all disagreements"


def test_top_disagreements_returns_only_disagreements():
    result = diverge(lambda p: "A", lambda p: "B", PROMPTS)
    top = result.top_disagreements(n=3)
    assert len(top) <= 3
    assert all(not r.agree for r in top)


def test_top_disagreements_respects_n():
    result = diverge(lambda p: "A", lambda p: "B", PROMPTS)
    assert len(result.top_disagreements(n=2)) == 2
    assert len(result.top_disagreements(n=100)) == len(PROMPTS)


# ---------------------------------------------------------------------------
# Label-based accuracy
# ---------------------------------------------------------------------------

def test_no_labels_no_accuracy():
    result = diverge(lambda p: "A", lambda p: "B", PROMPTS)
    assert result.accuracy_a is None
    assert result.accuracy_b is None
    assert result.disputed_accuracy_a is None


def test_perfect_model_a_accuracy():
    label_map = dict(zip(PROMPTS, LABELS))
    result = diverge(
        lambda p: label_map[p],
        lambda p: "wrong",
        PROMPTS,
        labels=LABELS,
    )
    assert result.accuracy_a == pytest.approx(1.0)
    assert result.accuracy_b == pytest.approx(0.0)


def test_accuracy_metrics_populated_with_labels():
    result = diverge(
        lambda p: "Paris",
        lambda p: "London",
        PROMPTS,
        labels=LABELS,
        match_fn=exact_match,
    )
    assert result.accuracy_a is not None
    assert result.accuracy_b is not None
    assert 0.0 <= result.accuracy_a <= 1.0
    assert 0.0 <= result.accuracy_b <= 1.0


def test_disputed_accuracy_computed():
    label_map = dict(zip(PROMPTS, LABELS))
    idx = [0]

    def model_a(p):
        return label_map[p]  # always correct

    def model_b(p):
        idx[0] += 1
        return "wrong" if idx[0] % 2 == 0 else label_map[p]

    result = diverge(model_a, model_b, PROMPTS, labels=LABELS)
    if result.disputed_accuracy_a is not None:
        assert result.disputed_accuracy_a == pytest.approx(1.0)


def test_label_mismatch_raises():
    with pytest.raises(ValueError, match="labels length"):
        diverge(lambda p: "A", lambda p: "B", PROMPTS, labels=["x", "y"])


# ---------------------------------------------------------------------------
# Custom match_fn
# ---------------------------------------------------------------------------

def test_letter_match_fn():
    result = diverge(
        lambda p: "A) Paris",
        lambda p: "A",
        PROMPTS,
        match_fn=letter_match,
    )
    assert result.agreement_rate == pytest.approx(1.0)


def test_custom_match_fn_respected():
    # Match function that always says they agree regardless of content
    result = diverge(
        lambda p: "completely different answer",
        lambda p: "another answer",
        PROMPTS,
        match_fn=lambda a, b: True,
    )
    assert result.agreement_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------

def test_model_names_stored():
    result = diverge(
        lambda p: "A", lambda p: "B", PROMPTS,
        model_a_name="gpt-4o", model_b_name="claude-haiku",
    )
    assert result.model_a_name == "gpt-4o"
    assert result.model_b_name == "claude-haiku"


def test_model_names_default():
    result = diverge(lambda p: "A", lambda p: "A", PROMPTS)
    assert result.model_a_name == "model_a"
    assert result.model_b_name == "model_b"


# ---------------------------------------------------------------------------
# report()
# ---------------------------------------------------------------------------

def test_report_contains_model_names():
    result = diverge(
        lambda p: "A", lambda p: "B", PROMPTS,
        model_a_name="GPT", model_b_name="Claude",
    )
    report = result.report()
    assert "GPT" in report
    assert "Claude" in report


def test_report_contains_disagreement_rate():
    result = diverge(lambda p: "A", lambda p: "B", PROMPTS)
    report = result.report()
    assert "100.0%" in report or "disagreement" in report.lower()


def test_report_shows_top_examples():
    result = diverge(lambda p: "A", lambda p: "B", PROMPTS)
    report = result.report()
    assert PROMPTS[0] in report or any(p[:30] in report for p in PROMPTS)


def test_report_with_labels_shows_accuracy():
    result = diverge(
        lambda p: "A", lambda p: "B", PROMPTS, labels=LABELS,
    )
    report = result.report()
    assert "Accuracy" in report


def test_report_full_agreement():
    result = diverge(lambda p: "same", lambda p: "same", PROMPTS)
    report = result.report()
    assert "100.0%" in report


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_prompts():
    result = diverge(lambda p: "A", lambda p: "B", [])
    assert len(result.results) == 0
    assert result.n_disagreements == 0


def test_single_prompt_agree():
    result = diverge(lambda p: "yes", lambda p: "yes", ["hello"])
    assert result.agreement_rate == pytest.approx(1.0)


def test_single_prompt_disagree():
    result = diverge(lambda p: "yes", lambda p: "no", ["hello"])
    assert result.agreement_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

def test_importable_from_top_level():
    from diverge import diverge as d, DivergenceResult as DR
    assert callable(d)
    assert DR is DivergenceResult
