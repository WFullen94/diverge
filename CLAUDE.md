# diverge

LLM differential tester — finds inputs where two models or prompt versions disagree maximally.

**GitHub:** `github.com/wfullen/diverge`
**Part of:** Wayne's tooling portfolio. Full backlog at `~/computer-vision-notebooks/BACKLOG.md`.

## What this project does

Given two models or two prompt versions, automatically searches for inputs where they disagree maximally — surfacing failure modes, capability gaps, and behavioral regressions that standard evals miss. Uses adversarial search to find the most informative disagreement cases rather than random sampling. Answers "what exactly changed between model A and model B?" with concrete examples, not aggregate metrics.

The gap this fills: Evals measure average performance; `diverge` finds the specific inputs where behavior changed. Academic differential testing work exists but no production tool. Invaluable for debugging prompt regressions, comparing model versions, and understanding fine-tuning side effects.

## Phased Roadmap

- **Phase 1 — Random Differential Sampling:** Given two model endpoints and a prompt template, sample inputs and flag high-disagreement cases by embedding distance
- **Phase 2 — Adversarial Search:** Use beam search / genetic algorithm to actively find inputs that maximize disagreement between the two models
- **Phase 3 — Disagreement Taxonomy:** Cluster disagreement cases by type (factuality, tone, refusal behavior, format, reasoning)
- **Phase 4 — Prompt Version Diff:** Given prompt_v1 and prompt_v2 against the same model, find inputs where the behavior changed most
- **Phase 5 — Regression Mode:** Given a baseline run and a current run, identify new failure modes introduced by a model update
- **Phase 6 — CLI:** `diverge compare --model-a gpt-4o --model-b gpt-4o-mini --dataset prompts.jsonl`
- **Phase 7 — Report:** HTML/Markdown report with disagreement examples, cluster summaries, severity ranking
- **Phase 8 — Distribution:** PyPI package, pairs naturally with `drift`

## Conventions

- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`
- Tag each phase: `v0.1.0`, etc.
- Model-agnostic: OpenAI, Anthropic, HuggingFace, local ollama via unified interface
- sentence-transformers for embedding-based disagreement scoring

## Current Status

Not started. Natural companion to `drift` (prompt regression framework).
