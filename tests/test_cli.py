"""Tests for Phase 4 CLI commands."""

import json
import pytest
from click.testing import CliRunner
from diverge.cli import main

PROMPTS = [
    "What is the capital of France?",
    "How many planets are in the solar system?",
    "Who wrote Hamlet?",
]
LABELS = ["Paris", "8", "Shakespeare"]


def _write_json(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


def _write_text(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines))
    return str(p)


# ---------------------------------------------------------------------------
# Helpers: _load_list
# ---------------------------------------------------------------------------

def test_load_list_json(tmp_path):
    from diverge.cli import _load_list
    f = _write_json(tmp_path, "p.json", PROMPTS)
    assert _load_list(f) == PROMPTS


def test_load_list_text(tmp_path):
    from diverge.cli import _load_list
    f = _write_text(tmp_path, "p.txt", PROMPTS)
    assert _load_list(f) == PROMPTS


def test_load_list_skips_blank_lines(tmp_path):
    from diverge.cli import _load_list
    f = _write_text(tmp_path, "p.txt", ["a", "", "b", ""])
    assert _load_list(f) == ["a", "b"]


# ---------------------------------------------------------------------------
# compare command
# ---------------------------------------------------------------------------

def test_compare_stable_models(tmp_path, monkeypatch):
    pf = _write_json(tmp_path, "p.json", PROMPTS)
    runner = CliRunner()

    # Patch _build_adapter to return mock callables without API calls
    def mock_build(spec):
        return spec, lambda p: "Paris"

    monkeypatch.setattr("diverge.cli._build_adapter", mock_build)

    result = runner.invoke(main, ["compare", "mock-a", "mock-b", "--prompts", pf])
    assert result.exit_code == 0
    assert "Agreement" in result.output or "agreement" in result.output.lower()


def test_compare_with_labels(tmp_path, monkeypatch):
    pf = _write_json(tmp_path, "p.json", PROMPTS)
    lf = _write_json(tmp_path, "l.json", LABELS)

    monkeypatch.setattr("diverge.cli._build_adapter",
                        lambda spec: (spec, lambda p: "Paris"))

    result = CliRunner().invoke(
        main, ["compare", "a", "b", "--prompts", pf, "--labels", lf]
    )
    assert result.exit_code == 0
    assert "Accuracy" in result.output


def test_compare_outputs_json(tmp_path, monkeypatch):
    pf = _write_json(tmp_path, "p.json", PROMPTS)
    out = str(tmp_path / "out.json")

    monkeypatch.setattr("diverge.cli._build_adapter",
                        lambda spec: (spec, lambda p: "Paris"))

    CliRunner().invoke(main, ["compare", "a", "b", "--prompts", pf, "--output", out])
    data = json.loads((tmp_path / "out.json").read_text())
    assert "agreement_rate" in data
    assert data["model_a"] == "a"


def test_compare_text_file_prompts(tmp_path, monkeypatch):
    pf = _write_text(tmp_path, "p.txt", PROMPTS)
    monkeypatch.setattr("diverge.cli._build_adapter",
                        lambda spec: (spec, lambda p: "answer"))
    result = CliRunner().invoke(main, ["compare", "a", "b", "--prompts", pf])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# find-failures command
# ---------------------------------------------------------------------------

def test_find_failures_mdd_strategy(tmp_path, monkeypatch):
    pf = _write_json(tmp_path, "p.json", PROMPTS[:2])
    monkeypatch.setattr("diverge.cli._build_adapter",
                        lambda spec: (spec, lambda p: "answer"))

    result = CliRunner().invoke(
        main, ["find-failures", "a", "b", "--prompts", pf, "--strategy", "mdd"]
    )
    assert result.exit_code == 0
    assert "Search Report" in result.output


def test_find_failures_beam_strategy(tmp_path, monkeypatch):
    pf = _write_json(tmp_path, "p.json", PROMPTS[:2])
    monkeypatch.setattr("diverge.cli._build_adapter",
                        lambda spec: (spec, lambda p: "answer"))

    result = CliRunner().invoke(
        main, ["find-failures", "a", "b", "--prompts", pf,
               "--strategy", "beam", "--beam-width", "2", "--n-per-type", "1"]
    )
    assert result.exit_code == 0


def test_find_failures_output_json(tmp_path, monkeypatch):
    pf = _write_json(tmp_path, "p.json", PROMPTS[:2])
    out = str(tmp_path / "failures.json")
    monkeypatch.setattr("diverge.cli._build_adapter",
                        lambda spec: (spec, lambda p: "answer"))

    CliRunner().invoke(main, ["find-failures", "a", "b", "--prompts", pf, "--output", out])
    data = json.loads((tmp_path / "failures.json").read_text())
    assert "mean_mdd" in data
    assert "mdd_distribution" in data


# ---------------------------------------------------------------------------
# diff command
# ---------------------------------------------------------------------------

def test_diff_no_change(tmp_path, monkeypatch):
    inf = _write_json(tmp_path, "i.json", PROMPTS)
    monkeypatch.setattr("diverge.cli._build_adapter",
                        lambda spec: (spec, lambda p: "Paris"))

    result = CliRunner().invoke(main, [
        "diff", "model",
        "--template-a", "Answer: {input}",
        "--template-b", "Answer: {input}",
        "--inputs", inf,
    ])
    assert result.exit_code == 0  # not significant → exit 0
    assert "NO" in result.output or "within noise" in result.output.lower()


def test_diff_significant_change_exits_1(tmp_path, monkeypatch):
    inf = _write_json(tmp_path, "i.json", PROMPTS)
    call_count = [0]

    def flip(p):
        call_count[0] += 1
        return "A" if call_count[0] % 2 == 1 else "B"

    monkeypatch.setattr("diverge.cli._build_adapter", lambda spec: (spec, flip))

    result = CliRunner().invoke(main, [
        "diff", "model",
        "--template-a", "v1: {input}",
        "--template-b", "v2: {input}",
        "--inputs", inf,
        "--alpha", "0.05",
    ])
    assert result.exit_code == 1  # significant → exit 1


def test_diff_with_labels(tmp_path, monkeypatch):
    inf = _write_json(tmp_path, "i.json", PROMPTS)
    lf = _write_json(tmp_path, "l.json", LABELS)
    monkeypatch.setattr("diverge.cli._build_adapter",
                        lambda spec: (spec, lambda p: "Paris"))

    result = CliRunner().invoke(main, [
        "diff", "model",
        "--template-a", "Q: {input}",
        "--template-b", "Q: {input}",
        "--inputs", inf, "--labels", lf,
    ])
    assert result.exit_code == 0
    assert "McNemar" in result.output


def test_diff_template_from_file(tmp_path, monkeypatch):
    inf = _write_json(tmp_path, "i.json", PROMPTS[:2])
    tf = tmp_path / "tmpl.txt"
    tf.write_text("Please answer: {input}")
    monkeypatch.setattr("diverge.cli._build_adapter",
                        lambda spec: (spec, lambda p: "x"))

    result = CliRunner().invoke(main, [
        "diff", "model",
        "--template-a", str(tf),
        "--template-b", "Answer: {input}",
        "--inputs", inf,
    ])
    assert result.exit_code in (0, 1)  # just check it runs


def test_diff_output_json(tmp_path, monkeypatch):
    inf = _write_json(tmp_path, "i.json", PROMPTS[:2])
    out = str(tmp_path / "diff.json")
    monkeypatch.setattr("diverge.cli._build_adapter",
                        lambda spec: (spec, lambda p: "Paris"))

    CliRunner().invoke(main, [
        "diff", "model",
        "--template-a", "A: {input}", "--template-b", "A: {input}",
        "--inputs", inf, "--output", out,
    ])
    data = json.loads((tmp_path / "diff.json").read_text())
    assert "p_value" in data
    assert "change_rate" in data


def test_diff_custom_labels(tmp_path, monkeypatch):
    inf = _write_json(tmp_path, "i.json", PROMPTS[:2])
    monkeypatch.setattr("diverge.cli._build_adapter",
                        lambda spec: (spec, lambda p: "x"))

    result = CliRunner().invoke(main, [
        "diff", "model",
        "--template-a", "A: {input}", "--template-b", "B: {input}",
        "--inputs", inf,
        "--prompt-a-label", "system-v1",
        "--prompt-b-label", "system-v2",
    ])
    assert "system-v1" in result.output
    assert "system-v2" in result.output


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_compare_missing_prompts_file():
    result = CliRunner().invoke(
        main, ["compare", "a", "b", "--prompts", "/nonexistent/path.json"]
    )
    assert result.exit_code != 0


def test_build_adapter_unknown_spec():
    from diverge.cli import _build_adapter
    import click
    with pytest.raises(click.ClickException):
        _build_adapter("unknownprovider:model")


def test_build_adapter_missing_openai_key(monkeypatch):
    from diverge.cli import _build_adapter
    import click
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(click.ClickException, match="OPENAI_API_KEY"):
        _build_adapter("openai:gpt-4o-mini")
