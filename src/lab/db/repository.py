"""
Data access functions for the research memory database.

Each function accepts an open sqlite3.Connection and operates within
whatever transaction context the caller has established.

Design: thin functions over raw SQL rather than an ORM.  Each function
does one thing and is independently testable.

C# parallel: a repository class with explicit ADO.NET SqlCommand calls
instead of LINQ-to-Entities.  Parameter binding (the ? placeholders)
prevents SQL injection.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone


# ─────────────────────────────────────────────────────────────────────────────
# Domain dataclasses
#
# These are plain Python dataclasses used as data-transfer objects between the
# database layer and the rest of the application.
# C# parallel: record types or DTOs.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResearchRun:
    id: str
    created_at: str
    data_path: str
    model: str
    status: str  # 'completed' | 'failed' | 'partial'


@dataclass
class Hypothesis:
    id: str
    run_id: str
    statement: str
    experiment_type: str
    parameters: dict
    created_at: str


@dataclass
class Experiment:
    id: str
    hypothesis_id: str
    experiment_type: str
    parameters: dict
    result: dict
    calculation_method: str
    candles_used: int
    created_at: str


@dataclass
class AgentArgument:
    id: str
    experiment_id: str
    agent_role: str   # 'researcher' | 'detractor' | 'judge'
    argument_text: str
    raw_response: str
    created_at: str


@dataclass
class JudgeDecision:
    id: str
    experiment_id: str
    decision: str      # 'accept' | 'reject' | 'inconclusive'
    rationale: str
    confidence: str    # 'high' | 'medium' | 'low'
    created_at: str


@dataclass
class Recommendation:
    """The Recommender agent's suggested next action after full deliberation.

    action is one of:
      - 'run_variant'        — test a parameter variant (auto-triggered by CLI)
      - 'archive_confirmed'  — strong evidence; archive as confirmed finding
      - 'archive_rejected'   — hypothesis rejected; archive for record-keeping
      - 'needs_more_data'    — insufficient evidence to conclude either way

    FUTURE: add a suggested_params field (dict) to carry the Recommender's
    proposed parameter variant for the run_variant action.
    """
    id: str
    run_id: str
    action: str
    rationale: str
    created_at: str


@dataclass
class Summary:
    """Plain-English summary of a research session from the Summarizer agent."""
    id: str
    run_id: str
    summary_text: str
    created_at: str


# ─────────────────────────────────────────────────────────────────────────────
# Write operations
# ─────────────────────────────────────────────────────────────────────────────

def save_research_run(conn: sqlite3.Connection, run: ResearchRun) -> None:
    """Insert a new research run record."""
    conn.execute(
        "INSERT INTO research_runs (id, created_at, data_path, model, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (run.id, run.created_at, run.data_path, run.model, run.status),
    )


def update_run_status(conn: sqlite3.Connection, run_id: str, status: str) -> None:
    """Update the status of an existing research run."""
    conn.execute(
        "UPDATE research_runs SET status = ? WHERE id = ?",
        (status, run_id),
    )


def save_hypothesis(conn: sqlite3.Connection, h: Hypothesis) -> None:
    """Insert a new hypothesis record."""
    conn.execute(
        "INSERT INTO hypotheses "
        "(id, run_id, statement, experiment_type, parameters, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (h.id, h.run_id, h.statement, h.experiment_type,
         json.dumps(h.parameters), h.created_at),
    )


def save_experiment(conn: sqlite3.Connection, exp: Experiment) -> None:
    """Insert a new experiment record with its deterministic result."""
    conn.execute(
        "INSERT INTO experiments "
        "(id, hypothesis_id, experiment_type, parameters, result, "
        " calculation_method, candles_used, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (exp.id, exp.hypothesis_id, exp.experiment_type,
         json.dumps(exp.parameters), json.dumps(exp.result),
         exp.calculation_method, exp.candles_used, exp.created_at),
    )


def save_agent_argument(conn: sqlite3.Connection, arg: AgentArgument) -> None:
    """Insert one agent's argument or response."""
    conn.execute(
        "INSERT INTO agent_arguments "
        "(id, experiment_id, agent_role, argument_text, raw_response, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (arg.id, arg.experiment_id, arg.agent_role,
         arg.argument_text, arg.raw_response, arg.created_at),
    )


def save_judge_decision(conn: sqlite3.Connection, decision: JudgeDecision) -> None:
    """Insert the Judge's final decision."""
    conn.execute(
        "INSERT INTO judge_decisions "
        "(id, experiment_id, decision, rationale, confidence, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (decision.id, decision.experiment_id, decision.decision,
         decision.rationale, decision.confidence, decision.created_at),
    )


def save_recommendation(conn: sqlite3.Connection, rec: Recommendation) -> None:
    """Insert the Recommender agent's suggested next action."""
    conn.execute(
        "INSERT INTO recommendations (id, run_id, action, rationale, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (rec.id, rec.run_id, rec.action, rec.rationale, rec.created_at),
    )


def save_summary(conn: sqlite3.Connection, summary: Summary) -> None:
    """Insert the Summarizer agent's plain-English session summary."""
    conn.execute(
        "INSERT INTO summaries (id, run_id, summary_text, created_at) "
        "VALUES (?, ?, ?, ?)",
        (summary.id, summary.run_id, summary.summary_text, summary.created_at),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Read operations
# ─────────────────────────────────────────────────────────────────────────────

def get_recent_runs(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    """Return the most recent research runs, newest first."""
    cursor = conn.execute(
        "SELECT id, created_at, data_path, model, status "
        "FROM research_runs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    # sqlite3.Row supports dict-style access; dict() converts it fully.
    return [dict(row) for row in cursor.fetchall()]


def get_run_details(conn: sqlite3.Connection, run_id: str) -> dict | None:
    """
    Return a complete picture of one research run, including its hypothesis,
    experiment result, agent arguments, and judge decision.

    Returns None if the run_id is not found.
    """
    run_row = conn.execute(
        "SELECT * FROM research_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if run_row is None:
        return None

    hyp_row = conn.execute(
        "SELECT * FROM hypotheses WHERE run_id = ?", (run_id,)
    ).fetchone()

    # Query run-level tables before any early return so they are always present.
    rec_row = conn.execute(
        "SELECT * FROM recommendations WHERE run_id = ?", (run_id,)
    ).fetchone()

    summary_row = conn.execute(
        "SELECT * FROM summaries WHERE run_id = ?", (run_id,)
    ).fetchone()

    if hyp_row is None:
        return {
            "run": dict(run_row),
            "hypothesis": None,
            "experiment": None,
            "agent_arguments": [],
            "judge_decision": None,
            "recommendation": dict(rec_row) if rec_row else None,
            "summary": dict(summary_row) if summary_row else None,
        }

    exp_row = conn.execute(
        "SELECT * FROM experiments WHERE hypothesis_id = ?", (hyp_row["id"],)
    ).fetchone()

    args_rows = conn.execute(
        "SELECT * FROM agent_arguments WHERE experiment_id = ? ORDER BY created_at",
        (exp_row["id"] if exp_row else None,),
    ).fetchall() if exp_row else []

    decision_row = conn.execute(
        "SELECT * FROM judge_decisions WHERE experiment_id = ?",
        (exp_row["id"] if exp_row else None,),
    ).fetchone() if exp_row else None

    # Deserialise JSON blobs back to dicts
    hyp = dict(hyp_row)
    hyp["parameters"] = json.loads(hyp["parameters"])

    exp = None
    if exp_row:
        exp = dict(exp_row)
        exp["parameters"] = json.loads(exp["parameters"])
        exp["result"] = json.loads(exp["result"])

    return {
        "run": dict(run_row),
        "hypothesis": hyp,
        "experiment": exp,
        "agent_arguments": [dict(r) for r in args_rows],
        "judge_decision": dict(decision_row) if decision_row else None,
        "recommendation": dict(rec_row) if rec_row else None,
        "summary": dict(summary_row) if summary_row else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def new_id() -> str:
    """Generate a new UUID string.  All IDs in the schema are UUID TEXT."""
    return str(uuid.uuid4())


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
