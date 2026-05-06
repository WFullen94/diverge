"""Core differential testing — run two models on the same prompts, rank disagreements.

The central question diverge answers: given model A and model B, which prompts
cause them to diverge, and what does that pattern reveal about their capability
differences?

Key design choices:
- match_fn is pluggable: swap between exact/normalized/letter/semantic matching
- labels are optional: without them you still get disagreement rates; with them
  you get disputed_accuracy (who was right on contested questions?)
- results are sorted by disagreement first for easy inspection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from diverge.match import normalized_match


@dataclass
class PromptResult:
    """Outcome for a single prompt tested against both models."""
    prompt: str
    answer_a: str
    answer_b: str
    agree: bool
    # Set when ground-truth labels are provided
    a_correct: bool | None = None
    b_correct: bool | None = None


@dataclass
class DivergenceResult:
    """Aggregate result from comparing two models across a prompt set."""

    results: list[PromptResult]
    model_a_name: str
    model_b_name: str

    # Aggregate metrics
    agreement_rate: float

    # Accuracy metrics — only populated when labels are provided
    accuracy_a: float | None = None
    accuracy_b: float | None = None
    # Accuracy restricted to prompts where the two models *disagreed*
    # Answers: "when they disagree, which one is right more often?"
    disputed_accuracy_a: float | None = None
    disputed_accuracy_b: float | None = None

    @property
    def disagreement_rate(self) -> float:
        return 1.0 - self.agreement_rate

    @property
    def n_disagreements(self) -> int:
        return sum(1 for r in self.results if not r.agree)

    def top_disagreements(self, n: int = 10) -> list[PromptResult]:
        """Return up to n prompts where the models disagreed, as-is."""
        return [r for r in self.results if not r.agree][:n]

    def report(self) -> str:
        lines = [
            "═" * 60,
            "  Divergence Report",
            "═" * 60,
            f"  Model A:  {self.model_a_name}",
            f"  Model B:  {self.model_b_name}",
            f"  Prompts:  {len(self.results)}",
            "─" * 60,
            f"  Agreement rate:     {self.agreement_rate:.1%}",
            f"  Disagreement rate:  {self.disagreement_rate:.1%}  "
            f"({self.n_disagreements} / {len(self.results)} prompts)",
        ]

        if self.accuracy_a is not None and self.accuracy_b is not None:
            lines += [
                "",
                f"  Accuracy (A):  {self.accuracy_a:.1%}",
                f"  Accuracy (B):  {self.accuracy_b:.1%}",
                f"  Gap (A−B):     {self.accuracy_a - self.accuracy_b:+.1%}",
            ]
            if self.disputed_accuracy_a is not None:
                lines += [
                    "",
                    "  On disputed prompts (where A ≠ B):",
                    f"    A correct:  {self.disputed_accuracy_a:.1%}",
                    f"    B correct:  {self.disputed_accuracy_b:.1%}",
                ]
                if self.disputed_accuracy_a > self.disputed_accuracy_b:
                    lines.append(
                        f"    → A wins disputed cases "
                        f"({self.disputed_accuracy_a:.1%} vs {self.disputed_accuracy_b:.1%})"
                    )
                elif self.disputed_accuracy_b > self.disputed_accuracy_a:
                    lines.append(
                        f"    → B wins disputed cases "
                        f"({self.disputed_accuracy_b:.1%} vs {self.disputed_accuracy_a:.1%})"
                    )
                else:
                    lines.append("    → Equal on disputed cases")

        # Top disagreements
        top = self.top_disagreements(n=5)
        if top:
            lines += ["", "─" * 60, "  Top disagreements:", ""]
            for i, r in enumerate(top, 1):
                lines.append(f"  [{i}] {r.prompt[:80]}")
                lines.append(f"      A: {r.answer_a[:60]}")
                lines.append(f"      B: {r.answer_b[:60]}")
                if r.a_correct is not None:
                    verdict = "A✓" if r.a_correct else ("B✓" if r.b_correct else "both wrong")
                    lines.append(f"      → {verdict}")
                lines.append("")

        lines.append("═" * 60)
        return "\n".join(lines)


def diverge(
    model_a: Callable[[str], str],
    model_b: Callable[[str], str],
    prompts: list[str],
    labels: list[str] | None = None,
    match_fn: Callable[[str, str], bool] | None = None,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
) -> DivergenceResult:
    """Compare two models across a set of prompts and rank their disagreements.

    Args:
        model_a: First model callable (str) -> str.
        model_b: Second model callable (str) -> str.
        prompts: Input prompts to test.
        labels: Optional ground-truth answers. When provided, accuracy metrics
                and disputed_accuracy are computed.
        match_fn: How to compare answers. Defaults to normalized_match.
                  Use letter_match for multiple-choice, semantic_match() for
                  open-ended prose.
        model_a_name: Display name for model A.
        model_b_name: Display name for model B.

    Returns:
        DivergenceResult with per-prompt outcomes and aggregate metrics,
        with disagreements listed first.
    """
    if labels is not None and len(labels) != len(prompts):
        raise ValueError(
            f"labels length ({len(labels)}) must match prompts length ({len(prompts)})"
        )

    if match_fn is None:
        match_fn = normalized_match

    results: list[PromptResult] = []

    for i, prompt in enumerate(prompts):
        ans_a = model_a(prompt).strip()
        ans_b = model_b(prompt).strip()
        agree = match_fn(ans_a, ans_b)

        a_correct: bool | None = None
        b_correct: bool | None = None
        if labels is not None:
            a_correct = match_fn(ans_a, labels[i])
            b_correct = match_fn(ans_b, labels[i])

        results.append(PromptResult(
            prompt=prompt,
            answer_a=ans_a,
            answer_b=ans_b,
            agree=agree,
            a_correct=a_correct,
            b_correct=b_correct,
        ))

    # Sort: disagreements first, then agreements
    results.sort(key=lambda r: (r.agree, 0))

    agreement_rate = float(np.mean([r.agree for r in results])) if results else 1.0

    # Accuracy metrics
    accuracy_a: float | None = None
    accuracy_b: float | None = None
    disputed_accuracy_a: float | None = None
    disputed_accuracy_b: float | None = None

    if labels is not None and results:
        accuracy_a = float(np.mean([r.a_correct for r in results]))
        accuracy_b = float(np.mean([r.b_correct for r in results]))

        disputed = [r for r in results if not r.agree]
        if disputed:
            disputed_accuracy_a = float(np.mean([r.a_correct for r in disputed]))
            disputed_accuracy_b = float(np.mean([r.b_correct for r in disputed]))

    return DivergenceResult(
        results=results,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        agreement_rate=agreement_rate,
        accuracy_a=accuracy_a,
        accuracy_b=accuracy_b,
        disputed_accuracy_a=disputed_accuracy_a,
        disputed_accuracy_b=disputed_accuracy_b,
    )
