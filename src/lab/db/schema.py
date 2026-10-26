"""
SQLite database schema for persistent research memory.

The schema stores the full provenance chain for every research run:
    ResearchRun → Hypothesis → Experiment → AgentArguments + JudgeDecision
                                           + Recommendations + Summaries (Milestone 2)

Design principles:
  • All IDs are UUIDs stored as TEXT (SQLite has no native UUID type).
  • All timestamps are stored as ISO-8601 TEXT (SQLite has no native DATETIME).
  • JSON blobs (parameters, results) are stored as TEXT using json.dumps().
  • Foreign keys are declared for documentation; SQLite requires explicit
    PRAGMA foreign_keys = ON to enforce them at runtime.

C# parallel: these CREATE TABLE statements are the equivalent of defining
EF Core entity classes with [Table], [Key], and [ForeignKey] attributes,
but without the ORM abstraction.  We use raw SQL for transparency and
simplicity.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


# Every CREATE TABLE uses IF NOT EXISTS so init_db() is idempotent —
# calling it multiple times never destroys existing data.
_CREATE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS research_runs (
        id          TEXT PRIMARY KEY,
        created_at  TEXT NOT NULL,
        data_path   TEXT NOT NULL,
        model       TEXT NOT NULL,
        status      TEXT NOT NULL    -- 'completed' | 'failed' | 'partial'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hypotheses (
        id           TEXT PRIMARY KEY,
        run_id       TEXT NOT NULL REFERENCES research_runs(id),
        statement    TEXT NOT NULL,
        experiment_type  TEXT NOT NULL,
        parameters   TEXT NOT NULL,  -- JSON blob
        created_at   TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS experiments (
        id              TEXT PRIMARY KEY,
        hypothesis_id   TEXT NOT NULL REFERENCES hypotheses(id),
        experiment_type TEXT NOT NULL,
        parameters      TEXT NOT NULL,   -- JSON blob
        result          TEXT NOT NULL,   -- JSON blob (deterministic output)
        calculation_method  TEXT NOT NULL,
        candles_used    INTEGER NOT NULL,
        created_at      TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_arguments (
        id              TEXT PRIMARY KEY,
        experiment_id   TEXT NOT NULL REFERENCES experiments(id),
        agent_role      TEXT NOT NULL,   -- 'researcher' | 'detractor' | 'judge'
        argument_text   TEXT NOT NULL,
        raw_response    TEXT NOT NULL,   -- full model output preserved for audit
        created_at      TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS judge_decisions (
        id              TEXT PRIMARY KEY,
        experiment_id   TEXT NOT NULL REFERENCES experiments(id),
        decision        TEXT NOT NULL,   -- 'accept' | 'reject' | 'inconclusive'
        rationale       TEXT NOT NULL,
        confidence      TEXT NOT NULL,   -- 'high' | 'medium' | 'low'
        created_at      TEXT NOT NULL
    )
    """,
    # ── Milestone 2 additions ────────────────────────────────────────────────
    #
    # recommendations: stores the Recommender agent's suggested next action
    # after the full deliberation chain completes.
    #
    # FUTURE: add a suggested_params JSON column here to support the
    # run_variant action (see design notes in milestone planning docs).
    """
    CREATE TABLE IF NOT EXISTS recommendations (
        id          TEXT PRIMARY KEY,
        run_id      TEXT NOT NULL REFERENCES research_runs(id),
        action      TEXT NOT NULL,   -- 'run_variant' | 'archive_confirmed' | 'archive_rejected' | 'needs_more_data'
        rationale   TEXT NOT NULL,
        created_at  TEXT NOT NULL
    )
    """,
    # summaries: plain-English summary produced by the Summarizer agent.
    # Stored separately so it can be retrieved quickly without joining
    # the full chain.
    """
    CREATE TABLE IF NOT EXISTS summaries (
        id           TEXT PRIMARY KEY,
        run_id       TEXT NOT NULL REFERENCES research_runs(id),
        summary_text TEXT NOT NULL,
        created_at   TEXT NOT NULL
    )
    """,

    # ── Milestone 3 additions ────────────────────────────────────────────────
    #
    # campaigns: a named sequence of research runs that share a budget and
    # a persistent memory of previously tested hypotheses.
    #
    # status values:
    #   'running'          — campaign is actively executing (or was interrupted)
    #   'completed'        — all runs finished and Recommender said to stop
    #   'stagnated'        — hypothesis diversity too low; campaign auto-stopped
    #   'budget_exhausted' — hit max_runs or max_total_calls limit
    #   'failed'           — unrecoverable error during a run
    """
    CREATE TABLE IF NOT EXISTS campaigns (
        id               TEXT PRIMARY KEY,
        created_at       TEXT NOT NULL,
        status           TEXT NOT NULL,
        max_runs         INTEGER NOT NULL,
        max_total_calls  INTEGER NOT NULL,
        runs_completed   INTEGER NOT NULL DEFAULT 0,
        total_calls_used INTEGER NOT NULL DEFAULT 0,
        stop_reason      TEXT,             -- human-readable reason campaign ended
        completed_at     TEXT
    )
    """,

    # campaign_runs: links each run to its parent campaign and records order.
    """
    CREATE TABLE IF NOT EXISTS campaign_runs (
        id          TEXT PRIMARY KEY,
        campaign_id TEXT NOT NULL REFERENCES campaigns(id),
        run_id      TEXT NOT NULL REFERENCES research_runs(id),
        run_index   INTEGER NOT NULL,   -- 0-based position in this campaign
        created_at  TEXT NOT NULL
    )
    """,
]


def init_db(db_path: Path) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and apply the schema.

    Returns an open connection.  The caller is responsible for closing it.
    Using a context manager (with conn:) commits or rolls back transactions.

    Args:
        db_path: File path for the SQLite database.  Created if absent.

    C# parallel: similar to DbContext.Database.EnsureCreated() in EF Core,
    but using raw DDL statements instead of migrations.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))

    # Return rows as dict-like objects so we can access columns by name:
    # row["id"] instead of row[0].  C# parallel: DataReader with column names.
    conn.row_factory = sqlite3.Row

    # SQLite foreign key enforcement is disabled by default for backwards
    # compatibility.  Enable it for every new connection.
    conn.execute("PRAGMA foreign_keys = ON")

    with conn:  # this context manager commits on success, rolls back on exception
        for statement in _CREATE_STATEMENTS:
            conn.execute(statement)

    return conn
