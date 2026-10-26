"""
Live end-to-end smoke test.

PREREQUISITES — this test is SKIPPED by default.
To run it, all of the following must be true:
  1. Ollama is running locally (ollama serve)
  2. The configured model is installed (e.g. ollama pull gemma4)
  3. The LAB_RUN_SMOKE_TEST=1 environment variable is set

Run with:
    LAB_RUN_SMOKE_TEST=1 pytest tests/smoke/ -v -s

This test performs a real research run and is expected to take 1–5 minutes
depending on model size.  It does NOT fake any model responses — the
assertions verify structural correctness, not specific content.

IMPORTANT: Never claim this test passed unless it was actually executed with
Ollama running and a model installed.  The default skip ensures the CI test
suite never falsely reports a live pass.
"""
from __future__ import annotations

import os
import uuid

import pytest

from lab.config import load_config
from lab.data.loader import load_csv
from lab.pipeline.runner import PipelineRunner


pytestmark = pytest.mark.skipif(
    os.environ.get("LAB_RUN_SMOKE_TEST") != "1",
    reason="Live smoke test — set LAB_RUN_SMOKE_TEST=1 to run",
)


def test_live_research_run():
    """
    Full end-to-end smoke test: PipelineRunner → Ollama → typed result.

    Asserts structural correctness of the result; does not assert specific
    hypothesis content or agent wording (those depend on the model).
    """
    cfg = load_config()
    load_result = load_csv(cfg.data_path)
    assert len(load_result.candles) > 50, "Not enough candles for smoke test"

    run_id = str(uuid.uuid4())
    runner = PipelineRunner(cfg, load_result.candles)
    result = runner.run(run_id)

    # Structural assertions — do not assert specific content
    assert result.run_id == run_id

    hyp = result.hypothesis
    assert hyp.get("hypothesis_statement")
    assert hyp.get("experiment_type") == "sma_crossover_followthrough"
    assert "experiment_params" in hyp

    exp = result.experiment_result
    assert "event_count" in exp
    assert "p_value" in exp
    assert "statistically_significant" in exp
    assert isinstance(exp["event_count"], int)
    assert 0.0 <= exp.get("win_rate", -1) <= 1.0

    assert result.devils_advocate.get("objection")
    assert result.researcher_rebuttal.get("rebuttal") is not None

    assert result.detractor_arguments.get("primary_challenge")

    judge = result.judge_decision
    assert judge.get("decision") in ("accept", "reject", "inconclusive")
    assert judge.get("rationale")

    # Integrity check — hardcoded False by architecture (Judge has no numerical output field)
    assert result.result_integrity_check.get("judge_altered_result") is False

    assert result.recommendation.get("action")
    assert result.summarizer_output.get("summary")

    print(f"\n✓ Smoke test passed.")
    print(f"  Decision:   {judge['decision'].upper()} (confidence: {judge.get('confidence')})")
    print(f"  Hypothesis: {hyp['hypothesis_statement'][:80]}")
    print(f"  Events:     {exp['event_count']}, p={exp['p_value']:.4f}")
    print(f"  Calls used: {result.ollama_calls_used} / {cfg.max_ollama_calls}")
    print(f"  Wall clock: {result.wall_clock_seconds}s")
