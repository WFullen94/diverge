# Research Directions

Open research questions that emerged from building diverge. Each one is a potential
workshop paper, arXiv preprint, or full venue submission.

---

## Idea A — Disagreement as a Capability Signal

**Hypothesis:** The *pattern* of where two models disagree is more informative than
aggregate accuracy gaps. Disagreement clusters by task type reveal specific capability
deltas — not just "A is better" but "A is better at multi-step reasoning, B is better
at factual recall."

**Novel claim:** Current model comparison work reports Δaccuracy as a scalar. Nobody
has studied the *topography* of disagreement — which inputs produce it, whether it
clusters predictably, and whether those clusters map onto interpretable capability
dimensions.

**What we need:**
- Phase 1: diverge() core — run A and B on MMLU/TruthfulQA/HellaSwag
- Analysis: cluster prompts by disagreement type using embeddings; measure whether
  clusters align with task categories
- Metric: "capability gap profile" — a vector of per-category disagreement rates
  rather than a single number

**Potential finding:** Two models with identical overall accuracy can have opposite
capability gap profiles (A wins on reasoning, B wins on recall). The profile is
more actionable for practitioners selecting models than any single number.

**Target venue:** ACL Findings, EMNLP, or arXiv
**Status:** Unblocked once Phase 1 complete

---

## Idea B — Minimum Divergence Distance (MDD)

**Hypothesis:** The minimum semantic perturbation required to cause two models to
*diverge* (disagree on an answer they previously agreed on) is a novel inter-model
sensitivity metric. Analogous to llm-reliability's Semantic Stability Distance, but
measuring inter-model sensitivity rather than single-model robustness.

**Novel claim:** Current robustness metrics (SSD in llm-reliability, adversarial
consistency) measure how stable one model is under perturbation. MDD measures how
*different* two models are in their perturbation sensitivity — which tells you when
they will and won't disagree in production.

**What we need:**
- Phase 2 (adversarial search): perturbation-based disagreement search
- For each prompt where A and B agree, find the minimum perturbation that causes
  them to disagree
- Mean MDD across a test set = the inter-model divergence threshold

**Potential finding:** MDD correlates with overall accuracy gap — models that are
closer in capability have higher MDD (harder to find divergence inputs). MDD is a
better proxy for "will these models give different answers in prod?" than Δaccuracy.

**Target venue:** EACL, EMNLP Findings, arXiv
**Status:** Blocked on Phase 2 (adversarial search)

---

## Idea C — Prompt Version Regression Testing

**Hypothesis:** When a prompt is updated in a production system, the statistical
signature of a behavioral regression is distinguishable from an improvement — and
from noise — using disagreement rate with a statistical baseline.

**Novel claim:** Prompt A/B testing in production currently uses human eval or
task accuracy. Disagreement rate (between old and new prompt on a fixed model)
is a faster, cheaper signal. The key question: what disagreement rate is "too much
change" vs. expected variation?

**What we need:**
- Phase 3: prompt version diff mode
- Baseline: capture disagreement rate of prompt_v1 vs. itself (N runs, measures noise floor)
- Test: compare disagreement rate of prompt_v1 vs prompt_v2 to baseline
- Statistical test: is the observed disagreement significantly above the noise floor?

**Practical implication:** Developers can gate prompt deployments with
`diverge check --prompt-a v1.txt --prompt-b v2.txt --model gpt-4o` and get a
p-value before shipping.

**Target venue:** EMNLP System Demonstrations, or ACL industry track
**Status:** Blocked on Phase 3 (prompt version diff)

---

## Dependency map

```
Phase 1 (core)           ──► Idea A (disagreement pattern analysis)
Phase 2 (adversarial)    ──► Idea B (Minimum Divergence Distance)
Phase 3 (prompt diff)    ──► Idea C (prompt regression testing)
```

**Natural first paper: Idea A** — produced directly from running Phase 1 on real
models across MMLU/TruthfulQA/HellaSwag. Compute disagreement rates per task
category and check whether they cluster by capability type.
