"""
Integration tests for the ReplayPipelineRunner (Milestone 4).

The ReplayPipelineRunner is mocked so tests run in milliseconds without Ollama.
The SQLite DB is an in-memory instance.

What is tested:
  - ReplayPipelineRunner produces a well-formed ReplayStepResult.
  - Budget enforcement: step stops when per-step budget is exhausted.
  - Multi-horizon result is passed through correctly to step result.
  - ReplayEngine × ReplayPipelineRunner produce the expected number of steps.
  - Replay run and step records are persisted to SQLite correctly.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lab.agents.schemas import (
    ReplayJudgeOutput,
    ReplayRecommenderOutput,
    ReplayResearcherOutput,
    ReplaySummarizerOutput,
    SMAParamsNoHorizon,
)
from lab.config import LabConfig
from lab.data.loader import OHLCVCandle
from lab.db.repository import (
    get_replay_run,
    get_replay_steps,
    save_replay_run,
    save_replay_step,
    update_replay_run,
    new_id,
    now_iso,
    ReplayRun as ReplayRunRecord,
    ReplayStep as ReplayStepRecord,
)
from lab.db.schema import init_db
from lab.pipeline.replay_runner import ReplayPipelineRunner, ReplayStepResult
from lab.pipeline.runner import BudgetExceededError
from lab.research.replay import ReplayEngine, ReplayStep


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(**kwargs) -> LabConfig:
    defaults = dict(
        ollama_base_url="http://localhost:11434",
        default_model="test-model",
        max_ollama_calls=12,
        max_wall_clock_seconds=300,
        replay_window_size=300,
        replay_step_size=60,
        replay_horizons=[5, 15, 30, 60],
        max_replay_calls=200,
    )
    defaults.update(kwargs)
    return LabConfig(**defaults)


def _make_candles(n: int = 600) -> list[OHLCVCandle]:
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


def _make_step(step_index: int = 0, candles: list[OHLCVCandle] | None = None) -> ReplayStep:
    if candles is None:
        candles = _make_candles(300)
    return ReplayStep(
        step_index=step_index,
        window_start=step_index * 60,
        window_end=step_index * 60 + 300,
        candles=candles,
        timestamp_start=candles[0].timestamp,
        timestamp_end=candles[-1].timestamp,
    )


def _make_researcher_output(short: int = 5, long: int = 20) -> ReplayResearcherOutput:
    return ReplayResearcherOutput(
        hypothesis_statement=f"Test hypothesis short={short} long={long}",
        experiment_type="sma_crossover_multihorizon",
        experiment_params=SMAParamsNoHorizon(short_window=short, long_window=long),
    )


def _make_judge_output(decision: str = "reject") -> ReplayJudgeOutput:
    return ReplayJudgeOutput(
        decision=decision,
        rationale="Test rationale with at least twenty characters here",
        confidence="medium",
        recommended_horizon=15,
    )


def _make_rec_output(action: str = "reject_signal") -> ReplayRecommenderOutput:
    return ReplayRecommenderOutput(
        action=action,
        rationale="Test recommendation rationale here",
    )


def _make_summary_output() -> ReplaySummarizerOutput:
    return ReplaySummarizerOutput(
        summary="Test summary with at least twenty characters.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: ReplayPipelineRunner
# ─────────────────────────────────────────────────────────────────────────────

def test_replay_step_result_has_correct_structure():
    """A mocked ReplayPipelineRunner.run() returns a well-formed ReplayStepResult."""
    cfg = _make_config()
    candles = _make_candles(300)
    step = _make_step(0, candles)

    researcher = _make_researcher_output()
    judge = _make_judge_output("accept")
    rec = _make_rec_output("accept_signal")
    summary = _make_summary_output()

    call_seq = iter([researcher, judge, rec, summary])

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = lambda **kw: next(call_seq)

    with patch("lab.pipeline.replay_runner.instructor") as mock_instructor, \
         patch("lab.pipeline.replay_runner.openai"):
        mock_instructor.from_openai.return_value = mock_client
        mock_instructor.Mode.JSON = "JSON"

        runner = ReplayPipelineRunner(cfg, step, budget=12)
        result = runner.run()

    assert isinstance(result, ReplayStepResult)
    assert result.step_index == 0
    assert result.hypothesis["experiment_type"] == "sma_crossover_multihorizon"
    assert result.judge_decision["decision"] == "accept"
    assert result.recommendation["action"] == "accept_signal"
    assert len(result.summarizer_output["summary"]) >= 20
    assert result.ollama_calls_used == 4  # 4 agent calls, no retries


def test_replay_step_experiment_uses_window_candles():
    """The experiment is run against the step's candle slice, not all candles."""
    cfg = _make_config()
    candles = _make_candles(300)
    step = _make_step(0, candles)

    researcher = _make_researcher_output()
    judge = _make_judge_output()
    rec = _make_rec_output()
    summary = _make_summary_output()
    call_seq = iter([researcher, judge, rec, summary])

    captured_candles = []

    def fake_run_experiment(candles, experiment_type, parameters):
        captured_candles.extend(candles)
        return {
            "result": {
                "experiment_type": "sma_crossover_multihorizon",
                "short_window": parameters["short_window"],
                "long_window": parameters["long_window"],
                "horizons": [],
                "candles_used": len(candles),
                "calculation_method": "test",
            },
            "candles_used": len(candles),
            "method": "test",
        }

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = lambda **kw: next(call_seq)

    with patch("lab.pipeline.replay_runner.instructor") as mock_instructor, \
         patch("lab.pipeline.replay_runner.openai"), \
         patch("lab.pipeline.replay_runner.run_experiment", side_effect=fake_run_experiment):
        mock_instructor.from_openai.return_value = mock_client
        mock_instructor.Mode.JSON = "JSON"

        runner = ReplayPipelineRunner(cfg, step, budget=12)
        runner.run()

    assert len(captured_candles) == 300


def test_replay_budget_is_enforced():
    """ReplayPipelineRunner raises BudgetExceededError when budget < agent calls."""
    cfg = _make_config()
    candles = _make_candles(300)
    step = _make_step(0, candles)

    call_count = 0

    mock_client = MagicMock()

    def _make_call(**kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_researcher_output()
        raise AssertionError("Should not reach agent 2")

    mock_client.chat.completions.create.side_effect = _make_call

    with patch("lab.pipeline.replay_runner.instructor") as mock_instructor, \
         patch("lab.pipeline.replay_runner.openai"):
        mock_instructor.from_openai.return_value = mock_client
        mock_instructor.Mode.JSON = "JSON"

        runner = ReplayPipelineRunner(cfg, step, budget=1)
        with pytest.raises(BudgetExceededError):
            runner.run()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: ReplayEngine integration
# ─────────────────────────────────────────────────────────────────────────────

def test_engine_generates_correct_number_of_steps():
    candles = _make_candles(600)
    engine = ReplayEngine(candles, window_size=300, step_size=60)
    steps = list(engine.steps())
    # (600 - 300) // 60 + 1 = 5 + 1 = 6 steps
    assert len(steps) == 6


def test_engine_windows_are_non_overlapping_when_step_equals_window():
    candles = _make_candles(600)
    engine = ReplayEngine(candles, window_size=300, step_size=300)
    steps = list(engine.steps())
    assert len(steps) == 2
    assert steps[0].window_end == steps[1].window_start


# ─────────────────────────────────────────────────────────────────────────────
# Tests: DB persistence
# ─────────────────────────────────────────────────────────────────────────────

def test_replay_run_persisted_to_db():
    conn = _make_in_memory_db()
    replay_id = new_id()
    ts = now_iso()

    with conn:
        save_replay_run(conn, ReplayRunRecord(
            id=replay_id,
            created_at=ts,
            status="running",
            window_size=300,
            step_size=60,
            horizons=[5, 15, 30, 60],
            steps_completed=0,
            total_calls_used=0,
            completed_at=None,
        ))

    run = get_replay_run(conn, replay_id)
    assert run is not None
    assert run.status == "running"
    assert run.horizons == [5, 15, 30, 60]
    assert run.window_size == 300


def test_replay_run_update():
    conn = _make_in_memory_db()
    replay_id = new_id()
    ts = now_iso()

    with conn:
        save_replay_run(conn, ReplayRunRecord(
            id=replay_id, created_at=ts, status="running",
            window_size=300, step_size=60, horizons=[5, 15],
            steps_completed=0, total_calls_used=0, completed_at=None,
        ))
        update_replay_run(conn, replay_id, "completed", 5, 42, ts)

    run = get_replay_run(conn, replay_id)
    assert run.status == "completed"
    assert run.steps_completed == 5
    assert run.total_calls_used == 42


def test_replay_step_persisted_to_db():
    conn = _make_in_memory_db()
    replay_id = new_id()
    ts = now_iso()

    with conn:
        save_replay_run(conn, ReplayRunRecord(
            id=replay_id, created_at=ts, status="running",
            window_size=300, step_size=60, horizons=[5, 15, 30, 60],
            steps_completed=0, total_calls_used=0, completed_at=None,
        ))
        save_replay_step(conn, ReplayStepRecord(
            id=new_id(),
            replay_id=replay_id,
            step_index=0,
            window_start=0,
            window_end=300,
            timestamp_start="2023-01-02T09:30:00",
            timestamp_end="2023-01-02T14:30:00",
            parameters={"short_window": 5, "long_window": 20},
            result={"experiment_type": "sma_crossover_multihorizon", "horizons": []},
            judge_decision={"decision": "reject", "rationale": "test", "confidence": "low", "recommended_horizon": 5},
            recommendation={"action": "reject_signal", "rationale": "test"},
            summary="Test summary for this replay step",
            ollama_calls=4,
            created_at=ts,
        ))

    steps = get_replay_steps(conn, replay_id)
    assert len(steps) == 1
    assert steps[0]["step_index"] == 0
    assert steps[0]["parameters"]["short_window"] == 5
    assert steps[0]["judge_decision"]["decision"] == "reject"
    assert steps[0]["ollama_calls"] == 4


def test_get_replay_run_returns_none_for_unknown_id():
    conn = _make_in_memory_db()
    assert get_replay_run(conn, "00000000-0000-0000-0000-000000000000") is None


def test_multiple_steps_ordered_by_index():
    conn = _make_in_memory_db()
    replay_id = new_id()
    ts = now_iso()

    with conn:
        save_replay_run(conn, ReplayRunRecord(
            id=replay_id, created_at=ts, status="running",
            window_size=300, step_size=60, horizons=[5, 15],
            steps_completed=0, total_calls_used=0, completed_at=None,
        ))
        for idx in [2, 0, 1]:  # deliberately out of order
            save_replay_step(conn, ReplayStepRecord(
                id=new_id(),
                replay_id=replay_id,
                step_index=idx,
                window_start=idx * 60,
                window_end=idx * 60 + 300,
                timestamp_start="2023-01-02T09:30:00",
                timestamp_end="2023-01-02T14:30:00",
                parameters={"short_window": 5, "long_window": 20},
                result={},
                judge_decision={"decision": "reject"},
                recommendation={"action": "reject_signal"},
                summary="test",
                ollama_calls=4,
                created_at=ts,
            ))

    steps = get_replay_steps(conn, replay_id)
    assert [s["step_index"] for s in steps] == [0, 1, 2]
