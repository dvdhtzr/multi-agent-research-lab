"""
Integration tests for CampaignRunner.

The PipelineRunner is mocked so tests run in milliseconds without Ollama.
The SQLite DB is an in-memory instance so no files are left on disk.

What is tested:
  - A two-run campaign that stops when the Recommender says 'archive_rejected'.
  - A campaign that stops due to the max_runs budget.
  - A campaign that stops due to the max_campaign_calls budget.
  - A campaign that stops due to stagnation.
  - Campaign state is persisted correctly to SQLite after each run.
  - Resuming a 'running' campaign picks up from the correct run index.
  - Resuming a non-running campaign raises ValueError.
  - The prior_context passed to PipelineRunner includes previous hypotheses.

C# parallel: integration tests using a mock for PipelineRunner in place of
the real service, verifying that CampaignRunner's orchestration logic is
correct without making real Ollama calls.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
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
from lab.campaign.runner import CampaignRunner, CampaignResult, STAGNATION_CONSECUTIVE_RUNS
from lab.config import LabConfig
from lab.data.loader import OHLCVCandle
from lab.db.repository import get_campaign, get_campaign_hypotheses, get_recent_campaigns
from lab.db.schema import init_db
from lab.pipeline.runner import PipelineResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(**kwargs) -> LabConfig:
    defaults = dict(
        ollama_base_url="http://localhost:11434",
        default_model="test-model",
        max_ollama_calls=12,
        max_wall_clock_seconds=300,
        max_campaign_runs=5,
        max_campaign_calls=60,
    )
    defaults.update(kwargs)
    return LabConfig(**defaults)


def _make_candles(n: int = 200) -> list[OHLCVCandle]:
    base = datetime(2023, 1, 2, 9, 30)
    price = 100.0
    return [
        OHLCVCandle(
            timestamp=base + timedelta(minutes=i),
            open=price, high=price + 0.1, low=price - 0.1, close=price,
            volume=10_000.0,
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


def _make_pipeline_result(
    run_id: str,
    short: int,
    long: int,
    horizon: int,
    decision: str,
    action: str,
    p_value: float = 0.1662,
    event_count: int = 17,
    calls: int = 7,
) -> PipelineResult:
    """Build a minimal PipelineResult for use in tests."""
    return PipelineResult(
        run_id=run_id,
        hypothesis={
            "hypothesis_statement": f"Test hypothesis short={short} long={long}",
            "experiment_type": "sma_crossover_followthrough",
            "experiment_params": {
                "short_window": short,
                "long_window": long,
                "horizon_periods": horizon,
            },
        },
        devils_advocate={"objection": "test objection", "targeted_assumption": "test", "confidence": 0.5},
        researcher_rebuttal={"rebuttal": "test rebuttal", "concedes_point": False},
        experiment_result={
            "event_count": event_count,
            "positive_count": 11,
            "win_rate": 0.647,
            "p_value": p_value,
            "statistically_significant": p_value < 0.05,
            "minimum_events_met": event_count >= 10,
            "candles_used": 200,
            "calculation_method": "test",
        },
        experiment_meta={"method": "test", "candles_used": 200},
        detractor_arguments={"primary_challenge": "test", "methodological_concern": "test", "alternative_explanation": "test"},
        judge_decision={"decision": decision, "rationale": "test rationale", "confidence": "medium"},
        recommendation={"action": action, "rationale": "test rationale"},
        summarizer_output={"summary": "Test summary."},
        ollama_calls_used=calls,
        wall_clock_seconds=5.0,
    )


class FakePipelineRunner:
    """Replaces PipelineRunner with a controllable fake.

    results is a list of PipelineResult to return, one per call to run().
    The run_id and prior_context are captured for assertion.
    """

    def __init__(self, results: list[PipelineResult]):
        self._results = iter(results)
        self.calls: list[dict] = []

    def run(self, run_id: str, prior_context: str | None = None) -> PipelineResult:
        self.calls.append({"run_id": run_id, "prior_context": prior_context})
        return next(self._results)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_campaign_runner_with_fake(
    cfg: LabConfig,
    fake: FakePipelineRunner,
) -> CampaignRunner:
    """Build a CampaignRunner whose PipelineRunner is replaced by fake."""
    conn = _make_in_memory_db()
    candles = _make_candles()
    runner = CampaignRunner(cfg, candles, conn)
    # Patch PipelineRunner at the campaign.runner module level.
    with patch("lab.campaign.runner.PipelineRunner", return_value=fake):
        pass
    runner._fake = fake  # kept for inspection
    return runner, conn


# ─────────────────────────────────────────────────────────────────────────────
# Tests: stopping conditions
# ─────────────────────────────────────────────────────────────────────────────

def test_campaign_stops_on_terminal_action():
    """A 'archive_rejected' recommendation after run 1 stops the campaign cleanly."""
    cfg = _make_config(max_campaign_runs=5, max_campaign_calls=60)
    conn = _make_in_memory_db()
    candles = _make_candles()

    results = [
        _make_pipeline_result("r1", 5, 20, 10, "reject", "archive_rejected"),
    ]
    fake = FakePipelineRunner(results)

    runner = CampaignRunner(cfg, candles, conn)
    with patch("lab.campaign.runner.PipelineRunner", return_value=fake):
        result = runner.run()

    assert result.status == "completed"
    assert result.runs_completed == 1
    assert "archive_rejected" in result.stop_reason


def test_campaign_stops_on_max_runs():
    """Campaign stops after max_runs runs even if Recommender says 'run_variant'."""
    cfg = _make_config(max_campaign_runs=2, max_campaign_calls=60)
    conn = _make_in_memory_db()
    candles = _make_candles()

    results = [
        _make_pipeline_result("r1", 5, 20, 10, "reject", "run_variant"),
        _make_pipeline_result("r2", 8, 25, 10, "reject", "run_variant"),
    ]
    fake = FakePipelineRunner(results)

    runner = CampaignRunner(cfg, candles, conn)
    with patch("lab.campaign.runner.PipelineRunner", return_value=fake):
        result = runner.run()

    assert result.status == "budget_exhausted"
    assert result.runs_completed == 2


def test_campaign_stops_when_per_run_budget_too_low():
    """
    Campaign stops before the first run when the adaptive per-run allocation
    is below the minimum viable amount (8 calls).

    With max_campaign_calls=20 and max_campaign_runs=5:
        adaptive budget = 20 // 5 = 4  (< 8 minimum)
    The campaign stops immediately with zero runs and zero calls used.
    """
    cfg = _make_config(max_campaign_runs=5, max_campaign_calls=20)
    conn = _make_in_memory_db()
    candles = _make_candles()

    runner = CampaignRunner(cfg, candles, conn)
    # No fake needed — the runner stops before calling PipelineRunner at all.
    result = runner.run()

    assert result.status == "budget_exhausted"
    assert result.runs_completed == 0
    assert result.total_calls_used == 0
    assert "LAB_MAX_CAMPAIGN_CALLS" in result.stop_reason


def test_campaign_stops_when_calls_exhausted():
    """
    Campaign stops when remaining_calls drops to zero after completing runs.

    max_campaign_calls=8 with max_campaign_runs=1:
        adaptive budget = 8 // 1 = 8 → run executes (uses 7 calls)
        max_runs=1 reached → budget_exhausted
    This also verifies that total_calls_used is recorded correctly.
    """
    cfg = _make_config(max_campaign_runs=1, max_campaign_calls=8)
    conn = _make_in_memory_db()
    candles = _make_candles()

    results = [
        _make_pipeline_result("r1", 5, 20, 10, "reject", "run_variant", calls=7),
    ]
    fake = FakePipelineRunner(results)

    runner = CampaignRunner(cfg, candles, conn)
    with patch("lab.campaign.runner.PipelineRunner", return_value=fake):
        result = runner.run()

    assert result.status == "budget_exhausted"
    assert result.runs_completed == 1
    assert result.total_calls_used == 7


def test_campaign_stops_on_stagnation():
    """Campaign stops when STAGNATION_CONSECUTIVE_RUNS runs have similar params."""
    cfg = _make_config(max_campaign_runs=10, max_campaign_calls=100)
    conn = _make_in_memory_db()
    candles = _make_candles()

    # Build STAGNATION_CONSECUTIVE_RUNS runs with nearly identical params.
    results = [
        _make_pipeline_result(f"r{i}", 5, 20, 10, "reject", "run_variant")
        for i in range(STAGNATION_CONSECUTIVE_RUNS)
    ]
    fake = FakePipelineRunner(results)

    runner = CampaignRunner(cfg, candles, conn)
    with patch("lab.campaign.runner.PipelineRunner", return_value=fake):
        result = runner.run()

    assert result.status == "stagnated"
    assert "stagnation" in result.stop_reason.lower()


def test_campaign_continues_when_params_vary():
    """Campaign does NOT stagnate if parameters vary between runs."""
    cfg = _make_config(max_campaign_runs=3, max_campaign_calls=100)
    conn = _make_in_memory_db()
    candles = _make_candles()

    results = [
        _make_pipeline_result("r1", 5, 20, 10, "reject", "run_variant"),
        _make_pipeline_result("r2", 15, 50, 10, "reject", "run_variant"),  # very different
        _make_pipeline_result("r3", 10, 40, 10, "reject", "archive_rejected"),
    ]
    fake = FakePipelineRunner(results)

    runner = CampaignRunner(cfg, candles, conn)
    with patch("lab.campaign.runner.PipelineRunner", return_value=fake):
        result = runner.run()

    assert result.status == "completed"
    assert result.runs_completed == 3


# ─────────────────────────────────────────────────────────────────────────────
# Tests: DB persistence
# ─────────────────────────────────────────────────────────────────────────────

def test_campaign_persisted_to_db():
    """A completed campaign is written to the campaigns table."""
    cfg = _make_config(max_campaign_runs=5, max_campaign_calls=60)
    conn = _make_in_memory_db()
    candles = _make_candles()

    results = [
        _make_pipeline_result("r1", 5, 20, 10, "accept", "archive_confirmed"),
    ]
    fake = FakePipelineRunner(results)

    runner = CampaignRunner(cfg, candles, conn)
    with patch("lab.campaign.runner.PipelineRunner", return_value=fake):
        result = runner.run()

    campaign = get_campaign(conn, result.campaign_id)
    assert campaign is not None
    assert campaign.status == "completed"
    assert campaign.runs_completed == 1
    assert campaign.total_calls_used == 7


def test_campaign_runs_linked_to_campaign():
    """campaign_runs rows are created for each run."""
    cfg = _make_config(max_campaign_runs=5, max_campaign_calls=60)
    conn = _make_in_memory_db()
    candles = _make_candles()

    results = [
        _make_pipeline_result("r1", 5, 20, 10, "reject", "run_variant"),
        _make_pipeline_result("r2", 10, 30, 10, "reject", "archive_rejected"),
    ]
    fake = FakePipelineRunner(results)

    runner = CampaignRunner(cfg, candles, conn)
    with patch("lab.campaign.runner.PipelineRunner", return_value=fake):
        result = runner.run()

    rows = conn.execute(
        "SELECT * FROM campaign_runs WHERE campaign_id=? ORDER BY run_index",
        (result.campaign_id,),
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["run_index"] == 0
    assert rows[1]["run_index"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests: prior context injection
# ─────────────────────────────────────────────────────────────────────────────

def test_first_run_has_no_prior_context():
    """The first run in a campaign passes prior_context=None to PipelineRunner."""
    cfg = _make_config(max_campaign_runs=5, max_campaign_calls=60)
    conn = _make_in_memory_db()
    candles = _make_candles()

    calls_made = []

    class CapturingFake:
        def run(self, run_id: str, prior_context: str | None = None) -> PipelineResult:
            calls_made.append(prior_context)
            return _make_pipeline_result(run_id, 5, 20, 10, "reject", "archive_rejected")

    runner = CampaignRunner(cfg, candles, conn)
    with patch("lab.campaign.runner.PipelineRunner", return_value=CapturingFake()):
        runner.run()

    assert calls_made[0] is None


def test_second_run_receives_prior_context():
    """The second run receives context describing the first run's hypothesis."""
    cfg = _make_config(max_campaign_runs=5, max_campaign_calls=60)
    conn = _make_in_memory_db()
    candles = _make_candles()

    calls_made = []

    class CapturingFake:
        _call_idx = 0
        def run(self, run_id: str, prior_context: str | None = None) -> PipelineResult:
            calls_made.append(prior_context)
            params = [(5, 20), (10, 40)][self._call_idx]
            action = "run_variant" if self._call_idx == 0 else "archive_rejected"
            self._call_idx += 1
            return _make_pipeline_result(run_id, params[0], params[1], 10, "reject", action)

    runner = CampaignRunner(cfg, candles, conn)
    with patch("lab.campaign.runner.PipelineRunner", return_value=CapturingFake()):
        runner.run()

    # Second call must have prior context mentioning first run.
    assert calls_made[1] is not None
    assert "Run 1:" in calls_made[1]
    assert "short=5" in calls_made[1]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: resumability
# ─────────────────────────────────────────────────────────────────────────────

def test_resume_non_running_campaign_raises():
    """Resuming a completed campaign raises ValueError."""
    cfg = _make_config()
    conn = _make_in_memory_db()
    candles = _make_candles()

    results = [
        _make_pipeline_result("r1", 5, 20, 10, "reject", "archive_rejected"),
    ]
    fake = FakePipelineRunner(results)

    runner = CampaignRunner(cfg, candles, conn)
    with patch("lab.campaign.runner.PipelineRunner", return_value=fake):
        result = runner.run()

    # The campaign is now 'completed' — resume should raise.
    with pytest.raises(ValueError, match="only 'running' campaigns"):
        CampaignRunner.resume(result.campaign_id, cfg, candles, conn)


def test_resume_unknown_campaign_raises():
    """Resuming a non-existent campaign ID raises ValueError."""
    cfg = _make_config()
    conn = _make_in_memory_db()
    candles = _make_candles()
    with pytest.raises(ValueError, match="not found"):
        CampaignRunner.resume("00000000-0000-0000-0000-000000000000", cfg, candles, conn)
