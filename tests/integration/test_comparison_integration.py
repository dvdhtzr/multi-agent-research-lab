"""
Integration tests for Milestone 5: comparative evaluation.

Tests:
  - SingleAgentRunner produces correct results when Researcher is mocked.
  - SingleAgentRunner respects budget limits.
  - ComparisonRunner orchestrates all three approaches correctly.
  - Comparison results are persisted to SQLite.
  - DB CRUD for comparisons table.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from lab.agents.schemas import ResearcherOutput, SMAParams
from lab.campaign.runner import CampaignResult
from lab.config import LabConfig
from lab.data.loader import OHLCVCandle
from lab.db.repository import (
    Comparison,
    get_comparison,
    get_recent_comparisons,
    new_id,
    now_iso,
    save_comparison,
)
from lab.db.schema import init_db
from lab.evaluation.baseline import BaselineEvaluator, BaselineResult
from lab.evaluation.runner import ComparisonResult, ComparisonRunner
from lab.evaluation.single_agent import SingleAgentResult, SingleAgentRunner
from lab.pipeline.runner import BudgetExceededError, PipelineResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(**kwargs) -> LabConfig:
    defaults = dict(
        ollama_base_url="http://localhost:11434",
        default_model="test-model",
        max_ollama_calls=12,
        max_wall_clock_seconds=300,
        max_campaign_runs=3,
        max_campaign_calls=60,
        compare_n_samples=5,
        compare_llm_budget=12,
    )
    defaults.update(kwargs)
    return LabConfig(**defaults)


def _make_candles(n: int = 500) -> list[OHLCVCandle]:
    base = datetime(2023, 1, 2, 9, 30)
    return [
        OHLCVCandle(
            timestamp=base + timedelta(minutes=i),
            open=100.0, high=100.1, low=99.9, close=100.0, volume=10_000.0,
        )
        for i in range(n)
    ]


def _make_in_memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    from lab.db import schema as _schema
    with conn:
        for stmt in _schema._CREATE_STATEMENTS:
            conn.execute(stmt)
    return conn


def _researcher_output(short: int = 5, long: int = 20, horizon: int = 10) -> ResearcherOutput:
    return ResearcherOutput(
        hypothesis_statement=f"Test hypothesis short={short} long={long}",
        experiment_type="sma_crossover_followthrough",
        experiment_params=SMAParams(
            short_window=short,
            long_window=long,
            horizon_periods=horizon,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: SingleAgentRunner
# ─────────────────────────────────────────────────────────────────────────────

def test_single_agent_returns_correct_run_count():
    """SingleAgentRunner completes n_runs when budget allows."""
    cfg = _make_config(compare_llm_budget=10)
    candles = _make_candles()

    call_count = [0]

    def _fake_create(**kw):
        call_count[0] += 1
        return _researcher_output()

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _fake_create

    with patch("lab.evaluation.single_agent.instructor") as mock_instructor, \
         patch("lab.evaluation.single_agent.openai"):
        mock_instructor.from_openai.return_value = mock_client
        mock_instructor.Mode.JSON = "JSON"
        runner = SingleAgentRunner(cfg, candles, budget=10)
        result = runner.run(n_runs=5)

    assert isinstance(result, SingleAgentResult)
    assert result.n_runs == 5


def test_single_agent_stops_when_budget_exhausted():
    """SingleAgentRunner stops early when budget < n_runs × 1."""
    cfg = _make_config()
    candles = _make_candles()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _researcher_output()

    with patch("lab.evaluation.single_agent.instructor") as mock_instructor, \
         patch("lab.evaluation.single_agent.openai"):
        mock_instructor.from_openai.return_value = mock_client
        mock_instructor.Mode.JSON = "JSON"
        runner = SingleAgentRunner(cfg, candles, budget=3)
        result = runner.run(n_runs=10)

    # With budget=3 and 1 call/run, at most 3 runs
    assert result.n_runs <= 3
    assert result.total_calls_used <= 3


def test_single_agent_records_calls_used():
    """calls_used increments by 1 per run (no retries in this test)."""
    cfg = _make_config()
    candles = _make_candles()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _researcher_output()

    with patch("lab.evaluation.single_agent.instructor") as mock_instructor, \
         patch("lab.evaluation.single_agent.openai"):
        mock_instructor.from_openai.return_value = mock_client
        mock_instructor.Mode.JSON = "JSON"
        runner = SingleAgentRunner(cfg, candles, budget=5)
        result = runner.run(n_runs=5)

    assert result.total_calls_used == result.n_runs


def test_single_agent_result_has_experiment_fields():
    """Each run result contains the deterministic experiment output."""
    cfg = _make_config()
    candles = _make_candles()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _researcher_output(5, 20, 10)

    with patch("lab.evaluation.single_agent.instructor") as mock_instructor, \
         patch("lab.evaluation.single_agent.openai"):
        mock_instructor.from_openai.return_value = mock_client
        mock_instructor.Mode.JSON = "JSON"
        runner = SingleAgentRunner(cfg, candles, budget=5)
        result = runner.run(n_runs=1)

    assert result.n_runs == 1
    r = result.all_results[0]
    assert r.short_window == 5
    assert r.long_window == 20
    assert r.horizon_periods == 10
    assert 0.0 <= r.p_value <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: ComparisonRunner
# ─────────────────────────────────────────────────────────────────────────────

def _make_fake_pipeline_result(run_id: str, decision: str = "reject", action: str = "archive_rejected") -> PipelineResult:
    return PipelineResult(
        run_id=run_id,
        hypothesis={
            "hypothesis_statement": "Test",
            "experiment_type": "sma_crossover_followthrough",
            "experiment_params": {"short_window": 5, "long_window": 20, "horizon_periods": 10},
        },
        devils_advocate={"objection": "test", "targeted_assumption": "test", "confidence": 0.5},
        researcher_rebuttal={"rebuttal": "test", "concedes_point": False},
        experiment_result={
            "event_count": 0, "positive_count": 0, "win_rate": 0.0, "p_value": 1.0,
            "statistically_significant": False, "minimum_events_met": False, "candles_used": 500,
            "calculation_method": "test",
        },
        experiment_meta={"method": "test", "candles_used": 500},
        detractor_arguments={"primary_challenge": "test", "methodological_concern": "test", "alternative_explanation": "test"},
        judge_decision={"decision": decision, "rationale": "test rationale for judge", "confidence": "medium"},
        recommendation={"action": action, "rationale": "test"},
        summarizer_output={"summary": "Test summary."},
        ollama_calls_used=7,
        wall_clock_seconds=1.0,
    )


def test_comparison_runner_produces_result():
    """ComparisonRunner returns a ComparisonResult with all three arms."""
    cfg = _make_config(compare_n_samples=5, compare_llm_budget=12)
    candles = _make_candles()
    conn = _make_in_memory_db()

    # Mock single-agent's instructor client
    sa_mock_client = MagicMock()
    sa_mock_client.chat.completions.create.return_value = _researcher_output()

    # Mock campaign pipeline runner
    from lab.pipeline.runner import PipelineResult
    fake_pipeline_result = _make_fake_pipeline_result("r1", "reject", "archive_rejected")

    class FakePipelineRunner:
        def run(self, run_id, prior_context=None):
            return fake_pipeline_result

    with patch("lab.evaluation.single_agent.instructor") as sa_instr, \
         patch("lab.evaluation.single_agent.openai"), \
         patch("lab.campaign.runner.PipelineRunner", return_value=FakePipelineRunner()):
        sa_instr.from_openai.return_value = sa_mock_client
        sa_instr.Mode.JSON = "JSON"

        runner = ComparisonRunner(cfg, candles, conn)
        result = runner.run()

    assert isinstance(result, ComparisonResult)
    assert isinstance(result.baseline, BaselineResult)
    assert isinstance(result.single_agent, SingleAgentResult)
    assert isinstance(result.multi_agent, CampaignResult)
    assert result.n_samples == 5
    assert result.llm_budget == 12


def test_comparison_summaries_have_three_entries():
    """ComparisonResult.summaries returns baseline, single_agent, multi_agent."""
    cfg = _make_config(compare_n_samples=3, compare_llm_budget=8)
    candles = _make_candles()
    conn = _make_in_memory_db()

    sa_mock_client = MagicMock()
    sa_mock_client.chat.completions.create.return_value = _researcher_output()

    class FakePipelineRunner:
        def run(self, run_id, prior_context=None):
            return _make_fake_pipeline_result(run_id, "reject", "archive_rejected")

    with patch("lab.evaluation.single_agent.instructor") as sa_instr, \
         patch("lab.evaluation.single_agent.openai"), \
         patch("lab.campaign.runner.PipelineRunner", return_value=FakePipelineRunner()):
        sa_instr.from_openai.return_value = sa_mock_client
        sa_instr.Mode.JSON = "JSON"

        runner = ComparisonRunner(cfg, candles, conn)
        result = runner.run()

    summaries = result.summaries
    assert len(summaries) == 3
    approaches = [s.approach for s in summaries]
    assert "baseline" in approaches
    assert "single_agent" in approaches
    assert "multi_agent" in approaches


def test_comparison_to_dict_has_all_keys():
    """to_dict() produces a serialisable summary for the DB."""
    cfg = _make_config(compare_n_samples=2, compare_llm_budget=8)
    candles = _make_candles()
    conn = _make_in_memory_db()

    sa_mock_client = MagicMock()
    sa_mock_client.chat.completions.create.return_value = _researcher_output()

    class FakePipelineRunner:
        def run(self, run_id, prior_context=None):
            return _make_fake_pipeline_result(run_id, "reject", "archive_rejected")

    with patch("lab.evaluation.single_agent.instructor") as sa_instr, \
         patch("lab.evaluation.single_agent.openai"), \
         patch("lab.campaign.runner.PipelineRunner", return_value=FakePipelineRunner()):
        sa_instr.from_openai.return_value = sa_mock_client
        sa_instr.Mode.JSON = "JSON"

        runner = ComparisonRunner(cfg, candles, conn)
        result = runner.run()

    d = result.to_dict()
    assert "comparison_id" in d
    assert "baseline" in d
    assert "single_agent" in d
    assert "multi_agent" in d
    assert "n_samples" in d
    assert "llm_budget" in d


# ─────────────────────────────────────────────────────────────────────────────
# Tests: DB persistence
# ─────────────────────────────────────────────────────────────────────────────

def test_save_and_retrieve_comparison():
    conn = _make_in_memory_db()
    cmp_id = new_id()
    ts = now_iso()

    with conn:
        save_comparison(conn, Comparison(
            id=cmp_id,
            created_at=ts,
            n_samples=50,
            llm_budget=60,
            baseline_json={"n_samples": 50, "significant_count": 3, "best_p_value": 0.03},
            single_json={"n_runs": 50, "significant_count": 2, "total_calls_used": 50},
            multi_json={"n_runs": 8, "significant_count": 1, "total_calls_used": 56},
            completed_at=ts,
        ))

    retrieved = get_comparison(conn, cmp_id)
    assert retrieved is not None
    assert retrieved.n_samples == 50
    assert retrieved.llm_budget == 60
    assert retrieved.baseline_json["significant_count"] == 3
    assert retrieved.single_json["n_runs"] == 50


def test_get_comparison_returns_none_for_unknown():
    conn = _make_in_memory_db()
    assert get_comparison(conn, "00000000-0000-0000-0000-000000000000") is None


def test_get_recent_comparisons_returns_newest_first():
    conn = _make_in_memory_db()
    for i in range(3):
        with conn:
            save_comparison(conn, Comparison(
                id=new_id(),
                created_at=f"2023-01-0{i + 1}T12:00:00+00:00",
                n_samples=10, llm_budget=20,
                baseline_json={}, single_json={}, multi_json={},
                completed_at=None,
            ))

    rows = get_recent_comparisons(conn, limit=3)
    assert len(rows) == 3
    # newest first
    assert rows[0]["created_at"] > rows[1]["created_at"]
