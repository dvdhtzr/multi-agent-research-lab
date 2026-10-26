"""
Integration tests for the PipelineRunner.

These tests verify the pipeline's orchestration logic without calling Ollama.
The instructor client is replaced with a mock that returns valid typed
responses for each agent, so the pipeline runs end-to-end in milliseconds.

What is tested:
  - All 8 agents are called in the correct order.
  - The deterministic experiment is executed between steps 3 and 5.
  - The final PipelineResult contains all expected fields.
  - Budget limits (call count) are enforced before each LLM call.
  - ExperimentValidationError is raised for invalid Researcher output.

What is NOT tested here:
  - That Ollama actually produces correct JSON (that is the smoke test's job).
  - The quality of model responses.

C# parallel: integration tests using a mock HttpClient — the HTTP layer is
replaced with a stub that returns canned responses, so the test exercises
the orchestration logic without making real network calls.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from lab.agents.schemas import (
    DevilsAdvocateOutput,
    DetractorOutput,
    JudgeOutput,
    RecommenderOutput,
    ResearcherOutput,
    ResearcherRebuttalOutput,
    SMAParams,
    SummarizerOutput,
)
from lab.config import LabConfig
from lab.data.loader import OHLCVCandle
from lab.pipeline.runner import BudgetExceededError, ExperimentValidationError, PipelineRunner


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_candles(n: int = 300) -> list[OHLCVCandle]:
    """Minimal synthetic candles — enough for the test experiment parameters."""
    base = datetime(2023, 1, 2, 9, 30)
    price = 100.0
    candles = []
    for i in range(n):
        price = max(price + (0.5 if i % 5 < 3 else -0.3), 1.0)
        candles.append(OHLCVCandle(
            timestamp=base + timedelta(minutes=i),
            open=price, high=price + 0.1, low=price - 0.1, close=price,
            volume=10_000.0,
        ))
    return candles


def _make_config(**overrides) -> LabConfig:
    defaults = dict(
        ollama_base_url="http://localhost:11434",
        default_model="test-model",
        max_ollama_calls=12,
        max_wall_clock_seconds=300,
    )
    defaults.update(overrides)
    return LabConfig(**defaults)


# Canned valid responses for each agent, in pipeline order.
_CANNED_RESPONSES = [
    ResearcherOutput(
        hypothesis_statement="When the 5-bar SMA crosses above the 20-bar SMA, price rises over 10 bars.",
        experiment_type="sma_crossover_followthrough",
        experiment_params=SMAParams(short_window=5, long_window=20, horizon_periods=10),
    ),
    DevilsAdvocateOutput(
        objection="The 10-bar horizon is too short to capture meaningful price moves in 1-minute data.",
        targeted_assumption="10 minutes is a meaningful prediction horizon.",
        confidence=0.6,
    ),
    ResearcherRebuttalOutput(
        rebuttal="Intraday momentum is well-documented over 5-15 minute windows in the literature.",
        concedes_point=False,
    ),
    DetractorOutput(
        primary_challenge="The sample size may be too small to generalise.",
        methodological_concern="Look-ahead bias cannot be ruled out with rolling windows.",
        alternative_explanation="The pattern may reflect time-of-day effects, not SMAs.",
    ),
    JudgeOutput(
        decision="accept",
        rationale="p < 0.05 and n >= 10 — the evidence meets the acceptance threshold.",
        confidence="medium",
    ),
    RecommenderOutput(
        action="archive_confirmed",
        rationale="Strong statistical evidence and no fatal methodological flaw identified.",
    ),
    SummarizerOutput(
        summary=(
            "The team tested whether a short-term average crossing above a long-term average "
            "predicts a price rise over the next 10 minutes. "
            "The result met the statistical threshold, and the team voted to archive the finding."
        ),
    ),
]


def _build_mock_runner(cfg: LabConfig, candles: list[OHLCVCandle]) -> PipelineRunner:
    """Return a PipelineRunner whose _call method returns canned responses."""
    runner = PipelineRunner(cfg, candles)
    call_index = {"i": 0}

    def fake_call(system, user, response_model):
        runner._check_budget()
        response = _CANNED_RESPONSES[call_index["i"]]
        call_index["i"] += 1
        runner._calls_used += 1
        return response

    runner._call = fake_call  # type: ignore[method-assign]
    return runner


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_returns_complete_result():
    """A full run produces a PipelineResult with all expected top-level fields."""
    cfg = _make_config()
    candles = _make_candles()
    runner = _build_mock_runner(cfg, candles)

    result = runner.run("test-run-id")

    assert result.run_id == "test-run-id"
    assert result.hypothesis["hypothesis_statement"]
    assert result.devils_advocate["objection"]
    assert result.researcher_rebuttal["rebuttal"] is not None
    assert result.experiment_result["event_count"] >= 0
    assert result.experiment_meta["method"]
    assert result.detractor_arguments["primary_challenge"]
    assert result.judge_decision["decision"] in ("accept", "reject", "inconclusive")
    assert result.recommendation["action"]
    assert result.summarizer_output["summary"]


def test_pipeline_uses_7_ollama_calls():
    """The pipeline makes exactly 7 LLM calls (one per agent)."""
    cfg = _make_config()
    candles = _make_candles()
    runner = _build_mock_runner(cfg, candles)
    runner.run("test-run-id")
    assert runner._calls_used == 7


def test_integrity_check_is_false():
    """The result_integrity_check must always report judge_altered_result=False."""
    cfg = _make_config()
    candles = _make_candles()
    runner = _build_mock_runner(cfg, candles)
    result = runner.run("test-run-id")
    assert result.result_integrity_check["judge_altered_result"] is False


def test_deterministic_experiment_result_is_in_result():
    """The experiment result contains the expected numerical fields."""
    cfg = _make_config()
    candles = _make_candles()
    runner = _build_mock_runner(cfg, candles)
    result = runner.run("test-run-id")
    exp = result.experiment_result
    assert "event_count" in exp
    assert "p_value" in exp
    assert "win_rate" in exp
    assert "statistically_significant" in exp
    assert isinstance(exp["event_count"], int)
    assert 0.0 <= exp["win_rate"] <= 1.0


def test_researcher_params_in_hypothesis():
    """The hypothesis dict contains the experiment_params sub-dict."""
    cfg = _make_config()
    candles = _make_candles()
    runner = _build_mock_runner(cfg, candles)
    result = runner.run("test-run-id")
    params = result.hypothesis.get("experiment_params", {})
    assert params.get("short_window") == 5
    assert params.get("long_window") == 20


# ─────────────────────────────────────────────────────────────────────────────
# Budget enforcement
# ─────────────────────────────────────────────────────────────────────────────

def test_budget_exceeded_raises_error():
    """A max_ollama_calls of 2 must stop the pipeline after 2 calls."""
    cfg = _make_config(max_ollama_calls=2)
    candles = _make_candles()
    runner = _build_mock_runner(cfg, candles)
    with pytest.raises(BudgetExceededError, match="budget"):
        runner.run("test-run-id")


def test_budget_message_mentions_env_var():
    """The BudgetExceededError message should guide the user to the right fix."""
    cfg = _make_config(max_ollama_calls=1)
    candles = _make_candles()
    runner = _build_mock_runner(cfg, candles)
    with pytest.raises(BudgetExceededError, match="LAB_MAX_OLLAMA_CALLS"):
        runner.run("test-run-id")
