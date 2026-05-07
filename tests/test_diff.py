"""Tests for Phase 3 prompt version diff."""

import pytest
from diverge.diff import prompt_diff, measure_baseline, PromptDiffResult, _binomial_p, _mcnemar_p

INPUTS = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "Who wrote Hamlet?",
    "What is the boiling point of water in Celsius?",
    "How many continents are there?",
    "What is the chemical formula for water?",
    "Who painted the Mona Lisa?",
    "What is the largest planet in the solar system?",
]
LABELS = ["Paris", "4", "Shakespeare", "100", "7", "H2O", "Leonardo da Vinci", "Jupiter"]

TEMPLATE_A = "Answer this question: {input}"
TEMPLATE_B = "Answer this question concisely: {input}"


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def test_binomial_p_zero_changes():
    # 0 out of 10 disagree with baseline=0 → p=1 (no evidence of change)
    p = _binomial_p(0, 10, 0.0)
    assert p == pytest.approx(1.0)


def test_binomial_p_all_change_against_zero_baseline():
    p = _binomial_p(10, 10, 0.0)
    assert p < 0.01


def test_binomial_p_above_baseline():
    # 8/10 disagree vs 50% baseline — not as extreme
    p_low = _binomial_p(10, 10, 0.0)
    p_high = _binomial_p(10, 10, 0.5)
    assert p_low < p_high


def test_mcnemar_no_change():
    # Both always correct → p=1 (no difference)
    a = [True] * 8
    b = [True] * 8
    p = _mcnemar_p(a, b)
    assert p == pytest.approx(1.0)


def test_mcnemar_always_switches():
    # A always right, B always wrong → strong effect
    a = [True] * 8
    b = [False] * 8
    p = _mcnemar_p(a, b)
    assert p < 0.05


def test_mcnemar_symmetric():
    a = [True, False, True, False]
    b = [False, True, False, True]
    p = _mcnemar_p(a, b)
    assert p > 0.05  # balanced off-diagonal — no significant difference


# ---------------------------------------------------------------------------
# prompt_diff — return type and shape
# ---------------------------------------------------------------------------

def test_returns_prompt_diff_result():
    result = prompt_diff(lambda p: "Paris", TEMPLATE_A, TEMPLATE_B, INPUTS[:3])
    assert isinstance(result, PromptDiffResult)


def test_inputs_stored():
    result = prompt_diff(lambda p: "Paris", TEMPLATE_A, TEMPLATE_B, INPUTS[:4])
    assert result.inputs == INPUTS[:4]


def test_labels_mismatch_raises():
    with pytest.raises(ValueError, match="labels length"):
        prompt_diff(lambda p: "x", TEMPLATE_A, TEMPLATE_B, INPUTS[:3], labels=["a", "b"])


# ---------------------------------------------------------------------------
# Identical templates — no change expected
# ---------------------------------------------------------------------------

def test_identical_templates_no_change():
    result = prompt_diff(
        lambda p: "Paris",
        "Answer: {input}", "Answer: {input}",
        INPUTS[:5],
    )
    assert result.change_rate == pytest.approx(0.0)
    assert result.divergence.n_disagreements == 0


def test_identical_templates_not_significant():
    result = prompt_diff(
        lambda p: "Paris",
        "Answer: {input}", "Answer: {input}",
        INPUTS,
    )
    assert not result.significant


# ---------------------------------------------------------------------------
# Different templates that trigger model output change
# ---------------------------------------------------------------------------

def test_different_templates_cause_change():
    # Model echoes the prompt prefix back — different prefixes = different answers
    def echo_prefix(p: str) -> str:
        return p.split(":")[0]  # returns "Answer this question" vs "Answer concisely"

    result = prompt_diff(echo_prefix, TEMPLATE_A, TEMPLATE_B, INPUTS[:4])
    assert result.change_rate > 0.0


def test_significant_change_detected():
    # All inputs change → very significant
    call_count = [0]

    def alternating(p: str) -> str:
        call_count[0] += 1
        return "A" if call_count[0] % 2 == 1 else "B"

    result = prompt_diff(alternating, TEMPLATE_A, TEMPLATE_B, INPUTS)
    assert result.significant


# ---------------------------------------------------------------------------
# Accuracy metrics with labels
# ---------------------------------------------------------------------------

def test_no_accuracy_without_labels():
    result = prompt_diff(lambda p: "x", TEMPLATE_A, TEMPLATE_B, INPUTS[:4])
    assert result.accuracy_a is None
    assert result.accuracy_b is None
    assert result.accuracy_delta is None
    assert result.mcnemar_p_value is None


def test_accuracy_computed_with_labels():
    result = prompt_diff(
        lambda p: "Paris",
        TEMPLATE_A, TEMPLATE_B, INPUTS[:5],
        labels=LABELS[:5],
    )
    assert result.accuracy_a is not None
    assert result.accuracy_b is not None
    assert 0.0 <= result.accuracy_a <= 1.0


def test_accuracy_delta_positive_when_b_improves():
    label_map = dict(zip(INPUTS, LABELS))
    call_count = [0]

    def model(p: str) -> str:
        call_count[0] += 1
        # Even calls (template B) get the right answer; odd calls (A) get wrong
        inp = p.split("\n\n")[-1] if "\n\n" in p else p.split(": ")[-1]
        return label_map.get(inp, "wrong") if call_count[0] % 2 == 0 else "wrong"

    result = prompt_diff(model, TEMPLATE_A, TEMPLATE_B, INPUTS[:4], labels=LABELS[:4])
    if result.accuracy_delta is not None:
        assert result.accuracy_delta >= 0.0


def test_mcnemar_p_value_populated():
    result = prompt_diff(
        lambda p: "Paris",
        TEMPLATE_A, TEMPLATE_B, INPUTS,
        labels=LABELS,
    )
    assert result.mcnemar_p_value is not None
    assert 0.0 <= result.mcnemar_p_value <= 1.0


# ---------------------------------------------------------------------------
# Template formatting
# ---------------------------------------------------------------------------

def test_template_with_placeholder():
    received = []

    def capture(p: str) -> str:
        received.append(p)
        return "answer"

    prompt_diff(capture, "Q: {input}", "Question: {input}", ["hello"])
    assert "Q: hello" in received[0]
    assert "Question: hello" in received[1]


def test_template_prefix_fallback():
    received = []

    def capture(p: str) -> str:
        received.append(p)
        return "answer"

    prompt_diff(capture, "Be concise.", "Be verbose.", ["hello"])
    assert "Be concise." in received[0]
    assert "hello" in received[0]


# ---------------------------------------------------------------------------
# Labels and metadata
# ---------------------------------------------------------------------------

def test_model_name_stored():
    result = prompt_diff(
        lambda p: "x", TEMPLATE_A, TEMPLATE_B, INPUTS[:2],
        model_name="gpt-4o",
    )
    assert result.model_name == "gpt-4o"


def test_prompt_labels_stored():
    result = prompt_diff(
        lambda p: "x", TEMPLATE_A, TEMPLATE_B, INPUTS[:2],
        prompt_a_label="system-v1", prompt_b_label="system-v2",
    )
    assert result.prompt_a_label == "system-v1"
    assert result.prompt_b_label == "system-v2"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_report_runs():
    result = prompt_diff(lambda p: "Paris", TEMPLATE_A, TEMPLATE_B, INPUTS[:3])
    report = result.report()
    assert "Prompt Diff Report" in report
    assert "v1" in report and "v2" in report


def test_report_with_labels():
    result = prompt_diff(
        lambda p: "Paris", TEMPLATE_A, TEMPLATE_B, INPUTS[:5], labels=LABELS[:5],
    )
    report = result.report()
    assert "Accuracy" in report
    assert "McNemar" in report


def test_report_no_change_message():
    result = prompt_diff(
        lambda p: "same", "A: {input}", "A: {input}", INPUTS[:3],
    )
    report = result.report()
    assert "NO" in report or "within noise" in report.lower()


# ---------------------------------------------------------------------------
# measure_baseline
# ---------------------------------------------------------------------------

def test_measure_baseline_deterministic_model():
    # Temperature=0 model → always same answer → baseline = 0
    rate = measure_baseline(lambda p: "Paris", TEMPLATE_A, INPUTS[:4])
    assert rate == pytest.approx(0.0)


def test_measure_baseline_stochastic_model():
    call_count = [0]

    def stochastic(p: str) -> str:
        call_count[0] += 1
        return "A" if call_count[0] % 2 == 1 else "B"

    rate = measure_baseline(stochastic, TEMPLATE_A, INPUTS[:4])
    assert rate > 0.0


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

def test_importable_from_top_level():
    from diverge import prompt_diff as pd, PromptDiffResult as PDR
    assert callable(pd)
    assert PDR is PromptDiffResult


# ---------------------------------------------------------------------------
# Helper property
# ---------------------------------------------------------------------------

# Add a convenience property so tests can access count directly
def test_n_disagreements_via_divergence():
    result = prompt_diff(lambda p: "same", "A: {input}", "A: {input}", INPUTS[:4])
    assert result.divergence.n_disagreements == 0
