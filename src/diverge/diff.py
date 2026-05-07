"""Prompt version diff — statistical regression detection for prompt updates.

Answers: "did this prompt change meaningfully alter model behavior?"

Two testing modes:
  Without labels — measures behavioral change rate (fraction of inputs where
    the response changed) and tests whether it's significantly above a baseline
    using a one-sided binomial test. At temperature=0, any change is real.

  With labels — adds McNemar's test on (a_correct, b_correct) pairs, which
    directly answers "did accuracy change?" with a p-value. Also reports
    which version wins on disputed inputs (where they gave different answers).

Research Idea C: A developer can gate prompt deployments with a p-value.
  If disagreement_rate >> baseline_rate → significant behavioral change detected.
  If accuracy_delta is negative and significant → this is a regression, block it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from diverge.core import DivergenceResult, diverge
from diverge.match import normalized_match


@dataclass
class PromptDiffResult:
    """Result of comparing two prompt versions on the same model."""

    inputs: list[str]
    model_name: str
    prompt_a_label: str
    prompt_b_label: str

    # Full behavioral diff — reuses DivergenceResult from Phase 1
    divergence: DivergenceResult

    # Statistical test on behavioral change rate
    change_rate: float          # fraction of inputs where v2 produced a different answer
    baseline_rate: float        # expected noise floor (0.0 for deterministic models)
    p_value: float              # one-sided binomial: H0 = change_rate ≤ baseline_rate
    significant: bool           # p_value < alpha
    alpha: float

    # Accuracy metrics (only when labels provided)
    accuracy_a: float | None = None
    accuracy_b: float | None = None
    accuracy_delta: float | None = None        # accuracy_b - accuracy_a (+ = improvement)
    mcnemar_p_value: float | None = None       # McNemar's test on paired correctness
    accuracy_change_significant: bool | None = None

    def report(self) -> str:
        lines = [
            "═" * 62,
            "  Prompt Diff Report (Phase 3)",
            "═" * 62,
            f"  Model:    {self.model_name}",
            f"  Prompt A: {self.prompt_a_label}",
            f"  Prompt B: {self.prompt_b_label}",
            f"  Inputs:   {len(self.inputs)}",
            "─" * 62,
            f"  Behavioral change rate:  {self.change_rate:.1%}  "
            f"({self.divergence.n_disagreements}/{len(self.inputs)} inputs changed)",
            f"  Baseline (noise floor):  {self.baseline_rate:.1%}",
            f"  p-value (binomial):      {self.p_value:.4f}",
            f"  Significant (α={self.alpha}):    {'YES — behavior changed' if self.significant else 'NO — within noise'}",
        ]

        if self.accuracy_a is not None:
            delta_str = f"{self.accuracy_delta:+.1%}" if self.accuracy_delta is not None else "N/A"
            lines += [
                "",
                f"  Accuracy ({self.prompt_a_label}):        {self.accuracy_a:.1%}",
                f"  Accuracy ({self.prompt_b_label}):        {self.accuracy_b:.1%}",
                f"  Accuracy delta (B−A):    {delta_str}",
            ]
            if self.mcnemar_p_value is not None:
                sig = self.accuracy_change_significant
                lines.append(
                    f"  McNemar p-value:         {self.mcnemar_p_value:.4f}  "
                    f"({'significant' if sig else 'not significant'})"
                )
            if self.accuracy_delta is not None:
                if self.accuracy_delta < -0.02 and self.accuracy_change_significant:
                    lines.append(f"\n  ⚠  REGRESSION DETECTED: accuracy dropped {self.accuracy_delta:.1%}")
                elif self.accuracy_delta > 0.02 and self.accuracy_change_significant:
                    lines.append(f"\n  ✓  IMPROVEMENT: accuracy gained {self.accuracy_delta:+.1%}")

        # Who wins on disputed inputs
        disp = self.divergence
        if disp.disputed_accuracy_a is not None:
            lines += [
                "",
                "  On changed inputs (where v1 ≠ v2 response):",
                f"    {self.prompt_a_label} correct: {disp.disputed_accuracy_a:.1%}",
                f"    {self.prompt_b_label} correct: {disp.disputed_accuracy_b:.1%}",
            ]

        # Top changed inputs
        top = disp.top_disagreements(n=3)
        if top:
            lines += ["", "─" * 62, "  Sample changed inputs:", ""]
            for i, r in enumerate(top, 1):
                lines.append(f"  [{i}] {r.prompt[:70]}")
                lines.append(f"      {self.prompt_a_label}: {r.answer_a[:60]}")
                lines.append(f"      {self.prompt_b_label}: {r.answer_b[:60]}")
                if r.a_correct is not None:
                    verdict = (
                        f"{self.prompt_a_label}✓" if r.a_correct
                        else f"{self.prompt_b_label}✓" if r.b_correct
                        else "both wrong"
                    )
                    lines.append(f"      → {verdict}")
                lines.append("")

        lines.append("═" * 62)
        return "\n".join(lines)


def _format_prompt(template: str, inp: str) -> str:
    """Apply template to an input. Supports {input} placeholder or prefix mode."""
    if "{input}" in template:
        return template.format(input=inp)
    return f"{template}\n\n{inp}"


def _binomial_p(k: int, n: int, p0: float) -> float:
    """One-sided binomial test: P(X >= k) under H0: p = p0."""
    from scipy.stats import binom
    if n == 0:
        return 1.0
    # P(X >= k) = 1 - P(X <= k-1)
    return float(1.0 - binom.cdf(k - 1, n, max(p0, 1e-9)))


def _mcnemar_p(a_correct: list[bool], b_correct: list[bool]) -> float:
    """McNemar's test on paired binary outcomes.

    Tests H0: the marginal probabilities of correctness are equal.
    Only pairs where exactly one is correct (off-diagonal) carry information.
    Returns exact binomial p-value for small n, normal approximation for large n.
    """
    from scipy.stats import binom
    n01 = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)
    n10 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    n = n01 + n10
    if n == 0:
        return 1.0
    # Exact: under H0, each off-diagonal cell is equally likely (Binomial(n, 0.5))
    k = max(n01, n10)
    return float(2 * binom.cdf(n - k, n, 0.5))  # two-sided


def prompt_diff(
    model_fn: Callable[[str], str],
    template_a: str,
    template_b: str,
    inputs: list[str],
    labels: list[str] | None = None,
    alpha: float = 0.05,
    baseline_rate: float = 0.0,
    match_fn: Callable[[str, str], bool] | None = None,
    model_name: str = "model",
    prompt_a_label: str = "v1",
    prompt_b_label: str = "v2",
) -> PromptDiffResult:
    """Compare two prompt versions on the same model.

    Runs model_fn on template_a(input) and template_b(input) for every input,
    then tests whether the behavioral change rate exceeds the baseline.

    Args:
        model_fn: The model to test (same model for both versions).
        template_a: Prompt template for version A. Use {input} as a placeholder,
                    e.g. "Answer this question: {input}". If no {input}, the
                    template is used as a prefix.
        template_b: Prompt template for version B.
        inputs: The variable input strings (questions, documents, etc.)
        labels: Optional ground-truth answers for accuracy comparison.
        alpha: Significance threshold (default 0.05).
        baseline_rate: Expected disagreement from model noise. Use 0.0 for
                       deterministic (temperature=0) models; measure empirically
                       for stochastic models by running the same prompt twice.
        match_fn: How to compare answers. Defaults to normalized_match.
        model_name: Display name for the model.
        prompt_a_label: Label for version A in reports (default: "v1").
        prompt_b_label: Label for version B in reports (default: "v2").

    Returns:
        PromptDiffResult with behavioral change rate, p-value, and optional
        accuracy metrics.
    """
    if labels is not None and len(labels) != len(inputs):
        raise ValueError(
            f"labels length ({len(labels)}) must match inputs length ({len(inputs)})"
        )
    if match_fn is None:
        match_fn = normalized_match

    # Build the two model callables: same model, different prompt wrapping
    def model_a(inp: str) -> str:
        return model_fn(_format_prompt(template_a, inp))

    def model_b(inp: str) -> str:
        return model_fn(_format_prompt(template_b, inp))

    # Reuse Phase 1 diverge() for per-input comparison
    div = diverge(
        model_a, model_b, inputs,
        labels=labels,
        match_fn=match_fn,
        model_a_name=prompt_a_label,
        model_b_name=prompt_b_label,
    )

    change_rate = div.disagreement_rate
    k = div.n_disagreements
    n = len(inputs)
    p_value = _binomial_p(k, n, baseline_rate)
    significant = p_value < alpha

    # McNemar's test on accuracy (only if labels provided)
    accuracy_a: float | None = div.accuracy_a
    accuracy_b: float | None = div.accuracy_b
    accuracy_delta: float | None = None
    mcnemar_p: float | None = None
    accuracy_sig: bool | None = None

    if accuracy_a is not None and accuracy_b is not None:
        accuracy_delta = accuracy_b - accuracy_a
        a_correct = [r.a_correct for r in div.results]
        b_correct = [r.b_correct for r in div.results]
        mcnemar_p = _mcnemar_p(a_correct, b_correct)
        accuracy_sig = mcnemar_p < alpha

    return PromptDiffResult(
        inputs=inputs,
        model_name=model_name,
        prompt_a_label=prompt_a_label,
        prompt_b_label=prompt_b_label,
        divergence=div,
        change_rate=change_rate,
        baseline_rate=baseline_rate,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        accuracy_a=accuracy_a,
        accuracy_b=accuracy_b,
        accuracy_delta=accuracy_delta,
        mcnemar_p_value=mcnemar_p,
        accuracy_change_significant=accuracy_sig,
    )


def measure_baseline(
    model_fn: Callable[[str], str],
    template: str,
    inputs: list[str],
    n_pairs: int = 3,
    match_fn: Callable[[str, str], bool] | None = None,
    seed: int = 42,
) -> float:
    """Measure the noise floor disagreement rate for a stochastic model.

    Runs the same prompt through the model twice and measures how often
    it gives different answers. Use this as baseline_rate when the model
    has temperature > 0.

    Args:
        model_fn: Model to measure (should have temperature > 0 for this to matter).
        template: Prompt template to test.
        inputs: Sample inputs to use for measurement.
        n_pairs: How many times to re-run each input (uses first two runs).
        match_fn: Answer comparison function.

    Returns:
        Measured baseline disagreement rate in [0, 1].
    """
    if match_fn is None:
        match_fn = normalized_match

    disagreements = 0
    total = 0
    for inp in inputs:
        prompt = _format_prompt(template, inp)
        answers = [model_fn(prompt).strip() for _ in range(2)]
        if not match_fn(answers[0], answers[1]):
            disagreements += 1
        total += 1

    return disagreements / total if total > 0 else 0.0
