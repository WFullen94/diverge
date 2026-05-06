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

from diverge.adapters import (
    ModelAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    OllamaAdapter,
    HuggingFaceAdapter,
)

__all__ = [
    "PromptResult",
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
