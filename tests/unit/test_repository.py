"""
Unit tests for the SQLite repository layer.

Uses an in-memory SQLite database (:memory:) so tests are fast and leave
no files on disk.  Each test gets a fresh connection via the fixture.

C# parallel: similar to using an InMemoryDatabase in EF Core for unit tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.db.repository import (
    AgentArgument,
    Experiment,
    Hypothesis,
    JudgeDecision,
    Recommendation,
    Summary,
    ResearchRun,
    get_recent_runs,
    get_run_details,
    new_id,
    now_iso,
    save_agent_argument,
    save_experiment,
    save_hypothesis,
    save_judge_decision,
    save_recommendation,
    save_summary,
    save_research_run,
    update_run_status,
)
from lab.db.schema import init_db


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def conn():
    """Open an in-memory SQLite database for one test, then close it."""
    # ":memory:" is a special SQLite path — the database exists only in RAM.
    import sqlite3
    from lab.db import schema
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    with c:
        for stmt in schema._CREATE_STATEMENTS:
            c.execute(stmt)
    yield c
    c.close()


def _sample_run(run_id: str | None = None) -> ResearchRun:
    return ResearchRun(
        id=run_id or new_id(),
        created_at=now_iso(),
        data_path="data/synthetic/spy_sample.csv",
        model="gemma4:latest",
        status="partial",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ResearchRun
# ─────────────────────────────────────────────────────────────────────────────

def test_save_and_retrieve_research_run(conn):
    run = _sample_run()
    with conn:
        save_research_run(conn, run)
    runs = get_recent_runs(conn, limit=10)
    assert len(runs) == 1
    assert runs[0]["id"] == run.id
    assert runs[0]["model"] == "gemma4:latest"
    assert runs[0]["status"] == "partial"


def test_update_run_status(conn):
    run = _sample_run()
    with conn:
        save_research_run(conn, run)
        update_run_status(conn, run.id, "completed")
    runs = get_recent_runs(conn)
    assert runs[0]["status"] == "completed"


def test_get_recent_runs_respects_limit(conn):
    with conn:
        for _ in range(5):
            save_research_run(conn, _sample_run())
    runs = get_recent_runs(conn, limit=3)
    assert len(runs) == 3


def test_get_recent_runs_returns_empty_list_when_no_runs(conn):
    runs = get_recent_runs(conn)
    assert runs == []


# ─────────────────────────────────────────────────────────────────────────────
# Full provenance chain
# ─────────────────────────────────────────────────────────────────────────────

def test_full_chain_is_retrievable(conn):
    """
    Save a complete research run (run → hypothesis → experiment → argument →
    decision) and verify that get_run_details returns all of it.
    """
    run_id = new_id()
    hyp_id = new_id()
    exp_id = new_id()
    ts = now_iso()

    with conn:
        save_research_run(conn, ResearchRun(
            id=run_id, created_at=ts,
            data_path="test.csv", model="gemma4:latest", status="completed",
        ))

        save_hypothesis(conn, Hypothesis(
            id=hyp_id, run_id=run_id,
            statement="Test hypothesis",
            experiment_type="sma_crossover_followthrough",
            parameters={"short_window": 5, "long_window": 20, "horizon_periods": 15},
            created_at=ts,
        ))

        save_experiment(conn, Experiment(
            id=exp_id, hypothesis_id=hyp_id,
            experiment_type="sma_crossover_followthrough",
            parameters={"short_window": 5, "long_window": 20, "horizon_periods": 15},
            result={"event_count": 42, "win_rate": 0.57, "p_value": 0.03,
                    "statistically_significant": True},
            calculation_method="scipy.stats.binomtest(k=24, n=42, p=0.5)",
            candles_used=1200,
            created_at=ts,
        ))

        save_agent_argument(conn, AgentArgument(
            id=new_id(), experiment_id=exp_id,
            agent_role="researcher",
            argument_text="Test hypothesis",
            raw_response='{"hypothesis_statement": "Test hypothesis"}',
            created_at=ts,
        ))

        save_judge_decision(conn, JudgeDecision(
            id=new_id(), experiment_id=exp_id,
            decision="accept",
            rationale="p=0.03 < 0.05, n=42 >= 10",
            confidence="high",
            created_at=ts,
        ))

    details = get_run_details(conn, run_id)

    assert details is not None
    assert details["run"]["id"] == run_id
    assert details["hypothesis"]["statement"] == "Test hypothesis"
    assert details["experiment"]["result"]["event_count"] == 42
    assert details["judge_decision"]["decision"] == "accept"
    assert len(details["agent_arguments"]) == 1


def test_get_run_details_returns_none_for_unknown_id(conn):
    result = get_run_details(conn, "nonexistent-uuid")
    assert result is None


def test_hypothesis_parameters_are_stored_as_json(conn):
    """Parameters dict is serialised to JSON and deserialised back correctly."""
    run_id = new_id()
    hyp_id = new_id()
    params = {"short_window": 7, "long_window": 21, "horizon_periods": 30}
    ts = now_iso()

    with conn:
        save_research_run(conn, _sample_run(run_id))
        save_hypothesis(conn, Hypothesis(
            id=hyp_id, run_id=run_id,
            statement="Test", experiment_type="sma_crossover_followthrough",
            parameters=params, created_at=ts,
        ))

    details = get_run_details(conn, run_id)
    assert details["hypothesis"]["parameters"] == params


# ─────────────────────────────────────────────────────────────────────────────
# Milestone 2 — Recommendation and Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def test_save_and_retrieve_recommendation(conn):
    """Recommendation is persisted and returned by get_run_details."""
    run_id = new_id()
    ts = now_iso()

    with conn:
        save_research_run(conn, _sample_run(run_id))
        save_recommendation(conn, Recommendation(
            id=new_id(),
            run_id=run_id,
            action="archive_confirmed",
            rationale="Strong evidence with p=0.02, n=45.",
            created_at=ts,
        ))

    details = get_run_details(conn, run_id)
    assert details["recommendation"] is not None
    assert details["recommendation"]["action"] == "archive_confirmed"
    assert "Strong evidence" in details["recommendation"]["rationale"]


def test_save_and_retrieve_summary(conn):
    """Summary text is persisted and returned by get_run_details."""
    run_id = new_id()
    ts = now_iso()

    with conn:
        save_research_run(conn, _sample_run(run_id))
        save_summary(conn, Summary(
            id=new_id(),
            run_id=run_id,
            summary_text="The team tested whether short-term averages predict upward moves.",
            created_at=ts,
        ))

    details = get_run_details(conn, run_id)
    assert details["summary"] is not None
    assert "short-term averages" in details["summary"]["summary_text"]


def test_get_run_details_returns_none_for_missing_recommendation(conn):
    """recommendation key is None when no recommendation was saved."""
    run_id = new_id()
    with conn:
        save_research_run(conn, _sample_run(run_id))
    details = get_run_details(conn, run_id)
    assert details["recommendation"] is None


def test_get_run_details_returns_none_for_missing_summary(conn):
    """summary key is None when no summary was saved."""
    run_id = new_id()
    with conn:
        save_research_run(conn, _sample_run(run_id))
    details = get_run_details(conn, run_id)
    assert details["summary"] is None


def test_recommendation_valid_actions(conn):
    """All four valid Recommender actions can be persisted."""
    actions = ["run_variant", "archive_confirmed", "archive_rejected", "needs_more_data"]
    for action in actions:
        run_id = new_id()
        with conn:
            save_research_run(conn, _sample_run(run_id))
            save_recommendation(conn, Recommendation(
                id=new_id(),
                run_id=run_id,
                action=action,
                rationale=f"Test rationale for {action}.",
                created_at=now_iso(),
            ))
        details = get_run_details(conn, run_id)
        assert details["recommendation"]["action"] == action
