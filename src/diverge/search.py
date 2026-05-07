"""Adversarial search — find inputs that maximally diverge two models.

Two strategies:

  mdd    — Minimum Divergence Distance search.
           For each seed prompt (where A and B currently agree), tries
           perturbations at increasing semantic levels (1→2→3) and records
           the minimum level that causes disagreement. Mean MDD across prompts
           is the Semantic Divergence Threshold — a single number describing
           how similar the two models are in their perturbation sensitivity.

  beam   — Beam search for maximal disagreement.
           Does not require seeds where models agree. Expands the top-K most
           disagreeing variants at each depth, surfacing the hardest cases
           for model comparison regardless of starting agreement. Useful for
           finding capability gap examples (Idea A).

Research Idea B: Minimum Divergence Distance (MDD).
  Higher MDD = models are more similar (hard to make them disagree via perturbation).
  Lower MDD = models diverge easily at the surface level (different surface sensitivity).
  MDD = None for a prompt = no perturbation in the search space caused divergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from diverge.match import normalized_match
from diverge.perturb import PERTURBATION_LEVELS, PerturbedPrompt, perturb


@dataclass
class DivergenceCandidate:
    """A single perturbation that was tested during search."""
    original: str
    perturbed: str
    perturbation_type: str
    level: int
    a_original: str      # model A's answer to original
    b_original: str      # model B's answer to original
    a_perturbed: str     # model A's answer to perturbed
    b_perturbed: str     # model B's answer to perturbed
    agreed_original: bool
    agreed_perturbed: bool

    @property
    def is_new_divergence(self) -> bool:
        """True if they agreed on the original but not the perturbed version."""
        return self.agreed_original and not self.agreed_perturbed

    @property
    def is_new_agreement(self) -> bool:
        """True if they disagreed on the original but agreed on the perturbed version."""
        return not self.agreed_original and self.agreed_perturbed


@dataclass
class SearchResult:
    """Adversarial search result for a single seed prompt."""
    seed_prompt: str
    seed_answer_a: str
    seed_answer_b: str
    seed_agree: bool
    candidates: list[DivergenceCandidate] = field(default_factory=list)

    @property
    def mdd(self) -> int | None:
        """Minimum perturbation level that caused new divergence. None if not found."""
        levels = [c.level for c in self.candidates if c.is_new_divergence]
        return min(levels) if levels else None

    @property
    def first_divergence(self) -> DivergenceCandidate | None:
        """The lowest-level candidate that caused new divergence."""
        divs = [c for c in self.candidates if c.is_new_divergence]
        return min(divs, key=lambda c: c.level) if divs else None

    @property
    def n_divergences_found(self) -> int:
        return sum(1 for c in self.candidates if c.is_new_divergence)


@dataclass
class AdversarialSearchResult:
    """Aggregate result from adversarial search over a prompt set."""
    results: list[SearchResult]
    strategy: str
    model_a_name: str
    model_b_name: str

    @property
    def mean_mdd(self) -> float | None:
        """Mean MDD across prompts where divergence was found."""
        mdds = [r.mdd for r in self.results if r.mdd is not None]
        return float(np.mean(mdds)) if mdds else None

    @property
    def divergence_found_rate(self) -> float:
        """Fraction of seed prompts where any new divergence was found."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.mdd is not None) / len(self.results)

    @property
    def mdd_distribution(self) -> dict[str, int]:
        """Count of prompts at each MDD level, plus 'none' for no divergence found."""
        dist: dict[str, int] = {"1": 0, "2": 0, "3": 0, "none": 0}
        for r in self.results:
            key = str(r.mdd) if r.mdd is not None else "none"
            dist[key] = dist.get(key, 0) + 1
        return dist

    def top_divergences(self, n: int = 5) -> list[DivergenceCandidate]:
        """Return up to n new-divergence candidates, lowest level first."""
        all_divs = [
            c for r in self.results for c in r.candidates if c.is_new_divergence
        ]
        all_divs.sort(key=lambda c: c.level)
        return all_divs[:n]

    def report(self) -> str:
        lines = [
            "═" * 62,
            "  Adversarial Search Report (Phase 2)",
            "═" * 62,
            f"  Model A:   {self.model_a_name}",
            f"  Model B:   {self.model_b_name}",
            f"  Strategy:  {self.strategy}",
            f"  Seeds:     {len(self.results)}",
            "─" * 62,
            f"  Divergence found rate:  {self.divergence_found_rate:.1%}  "
            f"({sum(1 for r in self.results if r.mdd is not None)}/{len(self.results)} prompts)",
        ]

        if self.mean_mdd is not None:
            lines.append(
                f"  Mean MDD:               {self.mean_mdd:.2f}  "
                "  (1=surface, 2=lexical, 3=structural)"
            )
            if self.mean_mdd <= 1.2:
                lines.append("  → Models diverge at surface level — high sensitivity to typos/format.")
            elif self.mean_mdd <= 2.0:
                lines.append("  → Models diverge at lexical level — synonym swaps trigger disagreement.")
            else:
                lines.append("  → Models are structurally robust — only word reorder causes divergence.")
        else:
            lines.append("  Mean MDD:               N/A  (no divergences found in search space)")

        dist = self.mdd_distribution
        lines += [
            "",
            "  MDD distribution:",
            f"    Level 1 (typo/format):  {dist['1']} prompts",
            f"    Level 2 (synonym):      {dist['2']} prompts",
            f"    Level 3 (reorder):      {dist['3']} prompts",
            f"    Not found:              {dist['none']} prompts",
        ]

        top = self.top_divergences(n=3)
        if top:
            lines += ["", "─" * 62, "  Top divergences found:", ""]
            for i, c in enumerate(top, 1):
                lines.append(f"  [{i}] Level {c.level} ({c.perturbation_type})")
                lines.append(f"      Original:  {c.original[:70]}")
                lines.append(f"      Perturbed: {c.perturbed[:70]}")
                lines.append(f"      A said: '{c.a_perturbed[:50]}'")
                lines.append(f"      B said: '{c.b_perturbed[:50]}'")
                lines.append("")

        lines.append("═" * 62)
        return "\n".join(lines)


def find_divergence_inputs(
    model_a: Callable[[str], str],
    model_b: Callable[[str], str],
    prompts: list[str],
    match_fn: Callable[[str, str], bool] | None = None,
    strategy: str = "mdd",
    perturbation_types: list[str] | None = None,
    n_per_type: int = 3,
    beam_width: int = 5,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
    seed: int = 42,
) -> AdversarialSearchResult:
    """Search for prompts that cause two models to diverge.

    Args:
        model_a: First model callable.
        model_b: Second model callable.
        prompts: Seed prompts to start from.
        match_fn: How to compare answers. Defaults to normalized_match.
        strategy: "mdd" (level-by-level MDD search) or "beam" (beam search
                  for maximal disagreement regardless of starting agreement).
        perturbation_types: Which types to apply. Defaults to all four.
        n_per_type: Perturbation variants per type per seed.
        beam_width: Number of candidates to expand per level (beam strategy only).
        model_a_name: Display name for model A.
        model_b_name: Display name for model B.
        seed: Random seed.

    Returns:
        AdversarialSearchResult with per-seed MDD values and aggregate metrics.
    """
    if match_fn is None:
        match_fn = normalized_match

    if strategy == "mdd":
        results = _mdd_search(
            model_a, model_b, prompts, match_fn,
            perturbation_types, n_per_type, seed,
        )
    elif strategy == "beam":
        results = _beam_search(
            model_a, model_b, prompts, match_fn,
            perturbation_types, n_per_type, beam_width, seed,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Choose 'mdd' or 'beam'.")

    return AdversarialSearchResult(
        results=results,
        strategy=strategy,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
    )


def _run_candidate(
    model_a: Callable,
    model_b: Callable,
    p: PerturbedPrompt,
    a_original: str,
    b_original: str,
    agreed_original: bool,
    match_fn: Callable,
) -> DivergenceCandidate:
    a_perturbed = model_a(p.perturbed).strip()
    b_perturbed = model_b(p.perturbed).strip()
    agreed_perturbed = match_fn(a_perturbed, b_perturbed)
    return DivergenceCandidate(
        original=p.original,
        perturbed=p.perturbed,
        perturbation_type=p.perturbation_type,
        level=p.level,
        a_original=a_original,
        b_original=b_original,
        a_perturbed=a_perturbed,
        b_perturbed=b_perturbed,
        agreed_original=agreed_original,
        agreed_perturbed=agreed_perturbed,
    )


def _mdd_search(
    model_a: Callable,
    model_b: Callable,
    prompts: list[str],
    match_fn: Callable,
    perturbation_types: list[str] | None,
    n_per_type: int,
    seed: int,
) -> list[SearchResult]:
    """Level-by-level search: try level 1 first, then 2, then 3.

    Stops for a given seed once a divergence is found (we have the MDD).
    All perturbations are still tried at the found level for completeness,
    but higher levels are skipped once MDD is established.
    """
    types = perturbation_types or list(PERTURBATION_LEVELS.keys())
    results = []

    # Sort types by level so we search cheapest first
    types_by_level: dict[int, list[str]] = {}
    for t in types:
        lvl = PERTURBATION_LEVELS[t]
        types_by_level.setdefault(lvl, []).append(t)

    for i, prompt in enumerate(prompts):
        a_orig = model_a(prompt).strip()
        b_orig = model_b(prompt).strip()
        agreed_orig = match_fn(a_orig, b_orig)

        sr = SearchResult(
            seed_prompt=prompt,
            seed_answer_a=a_orig,
            seed_answer_b=b_orig,
            seed_agree=agreed_orig,
        )

        found_mdd = False
        for level in sorted(types_by_level.keys()):
            level_types = types_by_level[level]
            level_perturbations = perturb(
                prompt, types=level_types, n_per_type=n_per_type,
                seed=seed + i * 100 + level,
            )
            for p in level_perturbations:
                cand = _run_candidate(model_a, model_b, p, a_orig, b_orig, agreed_orig, match_fn)
                sr.candidates.append(cand)
                if cand.is_new_divergence:
                    found_mdd = True

            if found_mdd:
                break  # MDD established at this level; skip higher levels

        results.append(sr)

    return results


def _beam_search(
    model_a: Callable,
    model_b: Callable,
    prompts: list[str],
    match_fn: Callable,
    perturbation_types: list[str] | None,
    n_per_type: int,
    beam_width: int,
    seed: int,
) -> list[SearchResult]:
    """Beam search: expand top-K most diverging variants at each depth.

    Unlike MDD search, this does NOT require seeds where models agree —
    it actively searches for the most disagreeing variants starting from
    any prompt. Useful for finding capability gap examples (Idea A).
    """
    types = perturbation_types or list(PERTURBATION_LEVELS.keys())
    results = []

    for i, prompt in enumerate(prompts):
        a_orig = model_a(prompt).strip()
        b_orig = model_b(prompt).strip()
        agreed_orig = match_fn(a_orig, b_orig)

        sr = SearchResult(
            seed_prompt=prompt,
            seed_answer_a=a_orig,
            seed_answer_b=b_orig,
            seed_agree=agreed_orig,
        )

        # Beam: list of (prompt_text, a_answer, b_answer, agreed)
        beam: list[tuple[str, str, str, bool]] = [(prompt, a_orig, b_orig, agreed_orig)]

        for depth in range(3):  # expand up to 3 levels deep
            candidates_this_level: list[tuple[DivergenceCandidate, str]] = []

            for beam_prompt, beam_a, beam_b, beam_agree in beam:
                perturbations = perturb(
                    beam_prompt, types=types, n_per_type=n_per_type,
                    seed=seed + i * 1000 + depth * 100,
                )
                for p in perturbations:
                    cand = _run_candidate(
                        model_a, model_b, p, beam_a, beam_b, beam_agree, match_fn
                    )
                    sr.candidates.append(cand)
                    # Score: prefer candidates that disagree on perturbed
                    score_key = (0 if not cand.agreed_perturbed else 1)
                    candidates_this_level.append((cand, p.perturbed))

            # Keep top beam_width most disagreeing variants for next level
            candidates_this_level.sort(key=lambda x: (x[0].agreed_perturbed, 0))
            beam = []
            for cand, perturbed_text in candidates_this_level[:beam_width]:
                beam.append((
                    perturbed_text,
                    cand.a_perturbed,
                    cand.b_perturbed,
                    cand.agreed_perturbed,
                ))
            if not beam:
                break

        results.append(sr)

    return results
