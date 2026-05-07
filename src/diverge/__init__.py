"""diverge — LLM differential tester.

Find inputs where two models (or two prompt versions) disagree maximally.
"""

from __future__ import annotations

__version__ = "0.1.0"

from diverge.core import (
    PromptResult,
    DivergenceResult,
    diverge,
)

from diverge.match import (
    exact_match,
    normalized_match,
    letter_match,
    semantic_match,
    semantic_similarity,
)

from diverge.diff import (
    PromptDiffResult,
    prompt_diff,
    measure_baseline,
)

from diverge.perturb import (
    PerturbedPrompt,
    perturb,
    PERTURBATION_LEVELS,
)

from diverge.search import (
    DivergenceCandidate,
    SearchResult,
    AdversarialSearchResult,
    find_divergence_inputs,
)

from diverge.adapters import (
    ModelAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    OllamaAdapter,
    HuggingFaceAdapter,
)

__all__ = [
    "PromptResult",
    "PromptDiffResult",
    "prompt_diff",
    "measure_baseline",
    "PerturbedPrompt",
    "perturb",
    "PERTURBATION_LEVELS",
    "DivergenceCandidate",
    "SearchResult",
    "AdversarialSearchResult",
    "find_divergence_inputs",
    "DivergenceResult",
    "diverge",
    "exact_match",
    "normalized_match",
    "letter_match",
    "semantic_match",
    "semantic_similarity",
    "ModelAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "OllamaAdapter",
    "HuggingFaceAdapter",
]
