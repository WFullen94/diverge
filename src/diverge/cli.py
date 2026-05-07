"""diverge CLI — compare models, find failures, diff prompt versions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable

import click
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_list(path: str) -> list[str]:
    """Load a JSON array or newline-delimited text file as a list of strings."""
    text = Path(path).read_text().strip()
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise click.ClickException(f"{path}: expected a JSON array")
        return [str(x) for x in data]
    return [line for line in text.splitlines() if line.strip()]


_SHORTHANDS = {
    "gpt-4o-mini":      ("openai",     "gpt-4o-mini"),
    "gpt-4o":           ("openai",     "gpt-4o"),
    "claude-haiku":     ("anthropic",  "claude-haiku-4-5-20251001"),
    "claude-haiku-4-5": ("anthropic",  "claude-haiku-4-5-20251001"),
    "claude-sonnet":    ("anthropic",  "claude-sonnet-4-6"),
}


def _build_adapter(spec: str) -> tuple[str, Callable]:
    """Parse 'provider:model' or shorthand → (display_name, callable adapter)."""
    from diverge.adapters import OpenAIAdapter, AnthropicAdapter, OllamaAdapter

    if spec in _SHORTHANDS:
        provider, model = _SHORTHANDS[spec]
    elif ":" in spec:
        provider, model = spec.split(":", 1)
    else:
        raise click.ClickException(
            f"Unknown model spec: {spec!r}. Use 'provider:model' "
            f"(e.g. openai:gpt-4o-mini) or a shorthand: {list(_SHORTHANDS)}"
        )

    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise click.ClickException("OPENAI_API_KEY not set.")
        return spec, OpenAIAdapter(model=model, api_key=key)
    elif provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise click.ClickException("ANTHROPIC_API_KEY not set.")
        return spec, AnthropicAdapter(model=model, api_key=key)
    elif provider == "ollama":
        return spec, OllamaAdapter(model=model)
    else:
        raise click.ClickException(f"Unknown provider: {provider!r}. Use openai, anthropic, or ollama.")


def _get_match_fn(match: str) -> Callable:
    from diverge.match import exact_match, normalized_match, letter_match, semantic_match
    return {
        "exact": exact_match,
        "normalized": normalized_match,
        "letter": letter_match,
        "semantic": semantic_match(threshold=0.85),
    }[match]


def _save_json(path: str, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2))
    console.print(f"[dim]Results saved → {path}[/]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
@click.version_option()
def main() -> None:
    """diverge — find inputs where two models or prompt versions disagree."""


@main.command()
@click.argument("model_a_spec")
@click.argument("model_b_spec")
@click.option("--prompts", "-p", required=True, type=click.Path(exists=True),
              help="JSON array or text file (one prompt per line).")
@click.option("--labels", "-l", type=click.Path(exists=True), default=None,
              help="JSON array or text file with ground-truth answers.")
@click.option("--match", "-m",
              type=click.Choice(["exact", "normalized", "letter", "semantic"]),
              default="normalized", show_default=True,
              help="Answer comparison strategy.")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Save results to JSON.")
@click.option("--top", default=10, show_default=True,
              help="Number of top disagreements to show.")
def compare(model_a_spec, model_b_spec, prompts, labels, match, output, top):
    """Compare MODEL_A_SPEC and MODEL_B_SPEC on a prompt set.

    Shows agreement rate, top disagreements, and accuracy comparison
    when ground-truth labels are provided.

    Model specs: gpt-4o-mini, claude-haiku, ollama:llama3, openai:gpt-4o, etc.

    \b
    Examples:
      diverge compare gpt-4o-mini claude-haiku --prompts q.json --labels a.json --match letter
      diverge compare openai:gpt-4o anthropic:claude-sonnet -p prompts.txt
    """
    from diverge.core import diverge as _diverge

    console.print(f"\n[bold cyan]diverge compare[/]  "
                  f"[yellow]{model_a_spec}[/] vs [yellow]{model_b_spec}[/]\n")

    name_a, adapter_a = _build_adapter(model_a_spec)
    name_b, adapter_b = _build_adapter(model_b_spec)

    prompt_list = _load_list(prompts)
    label_list = _load_list(labels) if labels else None

    console.print(f"  prompts: {len(prompt_list)}  "
                  f"labels: {'yes' if label_list else 'no'}  "
                  f"match: {match}\n")

    with console.status("Running both models…"):
        result = _diverge(
            adapter_a, adapter_b, prompt_list,
            labels=label_list,
            match_fn=_get_match_fn(match),
            model_a_name=name_a,
            model_b_name=name_b,
        )

    console.print(result.report())

    if output:
        _save_json(output, {
            "model_a": name_a,
            "model_b": name_b,
            "n_prompts": len(prompt_list),
            "agreement_rate": result.agreement_rate,
            "disagreement_rate": result.disagreement_rate,
            "accuracy_a": result.accuracy_a,
            "accuracy_b": result.accuracy_b,
            "disputed_accuracy_a": result.disputed_accuracy_a,
            "disputed_accuracy_b": result.disputed_accuracy_b,
            "top_disagreements": [
                {
                    "prompt": r.prompt,
                    "answer_a": r.answer_a,
                    "answer_b": r.answer_b,
                    "a_correct": r.a_correct,
                    "b_correct": r.b_correct,
                }
                for r in result.top_disagreements(top)
            ],
        })

    sys.exit(0)


@main.command("find-failures")
@click.argument("model_a_spec")
@click.argument("model_b_spec")
@click.option("--prompts", "-p", required=True, type=click.Path(exists=True),
              help="Seed prompts — ideally ones where the models currently agree.")
@click.option("--strategy", "-s",
              type=click.Choice(["mdd", "beam"]), default="mdd", show_default=True,
              help="mdd: level-by-level MDD search. beam: expand top-K most diverging.")
@click.option("--match", "-m",
              type=click.Choice(["exact", "normalized", "letter", "semantic"]),
              default="normalized", show_default=True)
@click.option("--n-per-type", default=3, show_default=True,
              help="Perturbation variants per type per seed.")
@click.option("--beam-width", default=5, show_default=True,
              help="Beam width (beam strategy only).")
@click.option("--output", "-o", type=click.Path(), default=None)
def find_failures(model_a_spec, model_b_spec, prompts, strategy, match, n_per_type, beam_width, output):
    """Search for inputs that cause MODEL_A and MODEL_B to diverge.

    Uses adversarial perturbation search to find the minimum semantic
    distance needed to trigger disagreement (MDD strategy) or the most
    informative disagreement examples (beam strategy).

    \b
    Examples:
      diverge find-failures gpt-4o-mini claude-haiku -p prompts.json
      diverge find-failures gpt-4o-mini claude-haiku -p prompts.json --strategy beam --beam-width 5
    """
    from diverge.search import find_divergence_inputs

    console.print(f"\n[bold cyan]diverge find-failures[/]  "
                  f"[yellow]{model_a_spec}[/] vs [yellow]{model_b_spec}[/]  "
                  f"strategy=[cyan]{strategy}[/]\n")

    name_a, adapter_a = _build_adapter(model_a_spec)
    name_b, adapter_b = _build_adapter(model_b_spec)
    prompt_list = _load_list(prompts)

    console.print(f"  seeds: {len(prompt_list)}  n_per_type: {n_per_type}\n")

    with console.status(f"Running {strategy} search…"):
        result = find_divergence_inputs(
            adapter_a, adapter_b, prompt_list,
            match_fn=_get_match_fn(match),
            strategy=strategy,
            n_per_type=n_per_type,
            beam_width=beam_width,
            model_a_name=name_a,
            model_b_name=name_b,
        )

    console.print(result.report())

    if output:
        dist = result.mdd_distribution
        _save_json(output, {
            "model_a": name_a,
            "model_b": name_b,
            "strategy": strategy,
            "n_seeds": len(prompt_list),
            "divergence_found_rate": result.divergence_found_rate,
            "mean_mdd": result.mean_mdd,
            "mdd_distribution": dist,
            "top_divergences": [
                {
                    "original": c.original,
                    "perturbed": c.perturbed,
                    "perturbation_type": c.perturbation_type,
                    "level": c.level,
                    "answer_a": c.a_perturbed,
                    "answer_b": c.b_perturbed,
                }
                for c in result.top_divergences(10)
            ],
        })

    sys.exit(0)


@main.command()
@click.argument("model_spec")
@click.option("--template-a", "-A", required=True,
              help="Prompt template for version A. Use {input} as placeholder.")
@click.option("--template-b", "-B", required=True,
              help="Prompt template for version B.")
@click.option("--inputs", "-i", required=True, type=click.Path(exists=True),
              help="JSON array or text file of input strings.")
@click.option("--labels", "-l", type=click.Path(exists=True), default=None,
              help="Ground-truth answers for accuracy comparison.")
@click.option("--match", "-m",
              type=click.Choice(["exact", "normalized", "letter", "semantic"]),
              default="normalized", show_default=True)
@click.option("--alpha", default=0.05, show_default=True,
              help="Significance threshold for statistical tests.")
@click.option("--baseline-rate", default=0.0, show_default=True,
              help="Expected noise-floor disagreement rate (0.0 for temp=0 models).")
@click.option("--prompt-a-label", default="v1", show_default=True)
@click.option("--prompt-b-label", default="v2", show_default=True)
@click.option("--output", "-o", type=click.Path(), default=None)
def diff(model_spec, template_a, template_b, inputs, labels, match,
         alpha, baseline_rate, prompt_a_label, prompt_b_label, output):
    """Test whether a prompt update caused a significant behavioral change.

    Runs MODEL on template A and template B for each input, then tests
    whether the change rate exceeds the baseline using a binomial test.
    With --labels, also runs McNemar's test on accuracy changes.

    \b
    Examples:
      diverge diff gpt-4o-mini \\
        --template-a "Answer this: {input}" \\
        --template-b "Answer concisely: {input}" \\
        --inputs questions.json --labels answers.json --match letter

      # Detect regressions in CI (exit 1 if significant change detected)
      diverge diff gpt-4o-mini -A "v1.txt" -B "v2.txt" -i inputs.json
    """
    from diverge.diff import prompt_diff as _prompt_diff

    # Template args can be file paths or inline strings
    def _resolve_template(t: str) -> str:
        p = Path(t)
        return p.read_text().strip() if p.exists() else t

    tmpl_a = _resolve_template(template_a)
    tmpl_b = _resolve_template(template_b)

    console.print(f"\n[bold cyan]diverge diff[/]  [yellow]{model_spec}[/]")
    console.print(f"  {prompt_a_label}: {tmpl_a[:60]}{'…' if len(tmpl_a) > 60 else ''}")
    console.print(f"  {prompt_b_label}: {tmpl_b[:60]}{'…' if len(tmpl_b) > 60 else ''}\n")

    _, adapter = _build_adapter(model_spec)
    input_list = _load_list(inputs)
    label_list = _load_list(labels) if labels else None

    with console.status("Running prompt diff…"):
        result = _prompt_diff(
            adapter,
            tmpl_a, tmpl_b,
            input_list,
            labels=label_list,
            alpha=alpha,
            baseline_rate=baseline_rate,
            match_fn=_get_match_fn(match),
            model_name=model_spec,
            prompt_a_label=prompt_a_label,
            prompt_b_label=prompt_b_label,
        )

    console.print(result.report())

    if output:
        _save_json(output, {
            "model": model_spec,
            "prompt_a_label": prompt_a_label,
            "prompt_b_label": prompt_b_label,
            "n_inputs": len(input_list),
            "change_rate": result.change_rate,
            "p_value": result.p_value,
            "significant": result.significant,
            "alpha": alpha,
            "accuracy_a": result.accuracy_a,
            "accuracy_b": result.accuracy_b,
            "accuracy_delta": result.accuracy_delta,
            "mcnemar_p_value": result.mcnemar_p_value,
        })

    # Exit 1 if significant change detected — useful for CI gating
    sys.exit(1 if result.significant else 0)
