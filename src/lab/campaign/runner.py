"""
Autonomous multi-run research campaigns (Milestone 3).

A campaign runs the 8-agent pipeline repeatedly, accumulating a research
memory of previously tested hypotheses.  Each iteration's Researcher is given
that memory so it proposes novel parameters rather than repeating prior work.

The CampaignRunner is responsible for:
  - Campaign lifecycle (start, iterate, stop, persist state).
  - Routing based on the Recommender's action after each run.
  - Stagnation detection (consecutive runs with nearly identical parameters).
  - Total budget enforcement across all runs, including instructor retries.
  - Safe resumability: state is written to SQLite after every completed run.
  - Adaptive per-run budget: remaining campaign calls distributed across
    remaining allowed runs.

Architecture
────────────
  CampaignRunner
    └── PipelineRunner (one per run, reconfigured with adaptive budget)
          └── instructor + Ollama (LLM calls)
          └── run_experiment() (deterministic Python)

The CampaignRunner never calls Ollama directly.  All LLM interaction happens
inside PipelineRunner._call(), which counts every real inference attempt
(original + retries) against the campaign budget.

Stagnation detection
────────────────────
After each run, the new hypothesis's SMA windows are compared to the previous
run's.  If both short_window and long_window are within STAGNATION_WINDOW_TOLERANCE
bars of the previous run for STAGNATION_CONSECUTIVE_RUNS consecutive runs, the
campaign is flagged as stagnated and stops.  This prevents the model from
looping over the same parameter neighbourhood indefinitely.

Stopping conditions (in priority order)
────────────────────────────────────────
  1. Recommender action is 'archive_confirmed' or 'archive_rejected' → COMPLETED
  2. Stagnation detected → STAGNATED
  3. Total LLM calls would exceed max_campaign_calls → BUDGET_EXHAUSTED
  4. Runs completed == max_runs → BUDGET_EXHAUSTED

Safe resumability
─────────────────
A campaign with status='running' can be resumed:
  CampaignRunner.resume(campaign_id, conn, cfg, candles)
The campaign's runs_completed and total_calls_used are read from SQLite and
used to reconstruct state.  The prior_context (previously tested hypotheses)
is rebuilt from the campaign_runs + hypotheses tables.

C# parallel: a state-machine loop where each iteration is a transaction that
commits progress to the database before advancing.  An interrupted campaign
can be resumed by reading the last committed state.
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from typing import Any, Callable

from lab.config import LabConfig
from lab.data.loader import OHLCVCandle
from lab.db.repository import (
    Campaign,
    CampaignRun,
    ResearchRun,
    new_id,
    now_iso,
    get_campaign,
    get_campaign_hypotheses,
    save_campaign,
    save_campaign_run,
    save_research_run,
    update_campaign,
    update_run_status,
)
from lab.pipeline.runner import BudgetExceededError, PipelineResult, PipelineRunner


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Two consecutive runs are considered "similar" if both SMA windows are within
# this many bars of each other.
STAGNATION_WINDOW_TOLERANCE = 2

# How many consecutive similar-parameter runs trigger stagnation.
STAGNATION_CONSECUTIVE_RUNS = 3

# Terminal Recommender actions — campaign stops cleanly on these.
TERMINAL_ACTIONS = {"archive_confirmed", "archive_rejected"}

# Actions that indicate the campaign should continue.
CONTINUE_ACTIONS = {"run_variant", "needs_more_data"}


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CampaignResult:
    """
    The final state of a completed (or stopped) campaign.

    run_results holds the PipelineResult for each run, in order.
    stop_reason is a human-readable explanation of why the campaign ended.
    """
    campaign_id: str
    status: str          # 'completed' | 'stagnated' | 'budget_exhausted' | 'failed'
    stop_reason: str
    runs_completed: int
    total_calls_used: int
    run_results: list[PipelineResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Campaign runner
# ─────────────────────────────────────────────────────────────────────────────

class CampaignRunner:
    """
    Orchestrates an autonomous multi-run research campaign.

    Usage:
        runner = CampaignRunner(cfg, candles, conn)
        result = runner.run()

    Args:
        cfg:     Application configuration (model, budgets, etc.)
        candles: The OHLCV dataset to pass to each pipeline run.
        conn:    Open SQLite connection.  Must have been initialised with
                 init_db() so the campaign tables exist.
    """

    def __init__(
        self,
        cfg: LabConfig,
        candles: list[OHLCVCandle],
        conn: sqlite3.Connection,
        on_step: Callable[[str, Any], None] | None = None,
        researcher_system: str | None = None,
        researcher_prompt: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.candles = candles
        self.conn = conn
        self._on_step = on_step
        self._researcher_system = researcher_system
        self._researcher_prompt = researcher_prompt

        # Mutable campaign state — updated after every completed run.
        self._campaign_id: str = ""
        self._runs_completed: int = 0
        self._total_calls_used: int = 0
        self._run_results: list[PipelineResult] = []
        self._stagnation_streak: int = 0
        self._prev_params: dict[str, int] | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> CampaignResult:
        """Start a new campaign and run it to completion (or budget exhaustion)."""
        campaign_id = str(uuid.uuid4())
        self._campaign_id = campaign_id
        self._runs_completed = 0
        self._total_calls_used = 0
        self._run_results = []
        self._stagnation_streak = 0
        self._prev_params = None

        with self.conn:
            save_campaign(self.conn, Campaign(
                id=campaign_id,
                created_at=now_iso(),
                status="running",
                max_runs=self.cfg.max_campaign_runs,
                max_total_calls=self.cfg.max_campaign_calls,
                runs_completed=0,
                total_calls_used=0,
                stop_reason=None,
                completed_at=None,
            ))

        return self._campaign_loop(prior_hypotheses=[])

    @classmethod
    def resume(
        cls,
        campaign_id: str,
        cfg: LabConfig,
        candles: list[OHLCVCandle],
        conn: sqlite3.Connection,
    ) -> CampaignResult:
        """
        Resume an interrupted campaign by its ID.

        Reads the campaign's persisted state (runs_completed, total_calls_used)
        from SQLite and rebuilds the prior-hypothesis context from stored data.

        Raises:
            ValueError: if campaign_id is not found or status is not 'running'.
        """
        campaign = get_campaign(conn, campaign_id)
        if campaign is None:
            raise ValueError(f"Campaign not found: {campaign_id}")
        if campaign.status != "running":
            raise ValueError(
                f"Campaign {campaign_id[:8]}... has status '{campaign.status}' "
                "— only 'running' campaigns can be resumed."
            )

        runner = cls(cfg, candles, conn)
        runner._campaign_id = campaign_id
        runner._runs_completed = campaign.runs_completed
        runner._total_calls_used = campaign.total_calls_used

        prior_hypotheses = get_campaign_hypotheses(conn, campaign_id)

        # Reconstruct stagnation state from the last two persisted param sets.
        if len(prior_hypotheses) >= 2:
            last_params = prior_hypotheses[-1]["parameters"]
            prev_params = prior_hypotheses[-2]["parameters"]
            if _params_are_similar(last_params, prev_params):
                runner._stagnation_streak = 2
                runner._prev_params = last_params
            else:
                runner._stagnation_streak = 0
                runner._prev_params = last_params
        elif len(prior_hypotheses) == 1:
            runner._prev_params = prior_hypotheses[-1]["parameters"]

        return runner._campaign_loop(prior_hypotheses=prior_hypotheses)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _campaign_loop(
        self,
        prior_hypotheses: list[dict],
    ) -> CampaignResult:
        """
        Core loop: run → check action → check stagnation → check budget → repeat.

        prior_hypotheses is passed in (not read from DB each iteration) so that
        resume() can seed it from persisted data, and new campaigns start empty.
        Each completed run appends to it in-memory.
        """
        while True:
            # ── Budget pre-check ──────────────────────────────────────────────
            if self._runs_completed >= self.cfg.max_campaign_runs:
                return self._stop(
                    "budget_exhausted",
                    f"Campaign reached the maximum of {self.cfg.max_campaign_runs} runs "
                    "(LAB_MAX_CAMPAIGN_RUNS).",
                )

            remaining_calls = self.cfg.max_campaign_calls - self._total_calls_used
            if remaining_calls <= 0:
                return self._stop(
                    "budget_exhausted",
                    f"Total call budget of {self.cfg.max_campaign_calls} exhausted "
                    "(LAB_MAX_CAMPAIGN_CALLS).",
                )

            # ── Adaptive per-run budget ───────────────────────────────────────
            remaining_runs = self.cfg.max_campaign_runs - self._runs_completed
            # Give each remaining run an equal share, but never exceed the
            # configured per-run hard limit.
            per_run_calls = min(
                remaining_calls // max(remaining_runs, 1),
                self.cfg.max_ollama_calls,
            )
            if per_run_calls < 8:
                # Not enough budget for even one minimum-viable run (7 agents +
                # one retry headroom).
                return self._stop(
                    "budget_exhausted",
                    f"Remaining campaign budget ({remaining_calls} calls) is too low "
                    "for another run (need at least 8). "
                    "Increase LAB_MAX_CAMPAIGN_CALLS.",
                )

            # ── Build prior-context prompt for the Researcher ─────────────────
            prior_context = _build_prior_context(prior_hypotheses)

            # ── Execute one pipeline run ──────────────────────────────────────
            run_id = str(uuid.uuid4())
            run_index = self._runs_completed
            ts = now_iso()

            run_cfg = _with_per_run_budget(self.cfg, per_run_calls)

            with self.conn:
                save_research_run(self.conn, ResearchRun(
                    id=run_id,
                    created_at=ts,
                    data_path=str(run_cfg.data_path),
                    model=run_cfg.default_model,
                    status="partial",
                ))

            pipeline = PipelineRunner(
                run_cfg,
                self.candles,
                on_step=self._on_step,
                researcher_system=self._researcher_system,
                researcher_prompt=self._researcher_prompt,
            )
            try:
                result = pipeline.run(run_id, prior_context=prior_context)
            except BudgetExceededError as exc:
                with self.conn:
                    update_run_status(self.conn, run_id, "failed")
                # Per-run budget can be hit if the model retries heavily.
                # Treat as overall budget exhaustion.
                self._total_calls_used += pipeline._calls_used
                return self._stop(
                    "budget_exhausted",
                    f"Per-run budget exhausted on run {run_index + 1}: {exc}",
                )
            except Exception as exc:
                with self.conn:
                    update_run_status(self.conn, run_id, "failed")
                return self._stop(
                    "failed",
                    f"Run {run_index + 1} failed with an unexpected error: {exc}",
                )

            with self.conn:
                update_run_status(self.conn, run_id, "completed")

            # ── Commit progress to SQLite (resumability checkpoint) ───────────
            self._runs_completed += 1
            self._total_calls_used += result.ollama_calls_used
            self._run_results.append(result)

            with self.conn:
                save_campaign_run(self.conn, CampaignRun(
                    id=new_id(),
                    campaign_id=self._campaign_id,
                    run_id=run_id,
                    run_index=run_index,
                    created_at=ts,
                ))
                update_campaign(
                    self.conn,
                    campaign_id=self._campaign_id,
                    status="running",
                    runs_completed=self._runs_completed,
                    total_calls_used=self._total_calls_used,
                )

            # Append this run's hypothesis to the in-memory history.
            hyp = result.hypothesis
            prior_hypotheses.append({
                "statement": hyp.get("hypothesis_statement", ""),
                "experiment_type": hyp.get("experiment_type", ""),
                "parameters": hyp.get("experiment_params", {}),
                "judge_decision": result.judge_decision.get("decision", ""),
                "p_value": result.experiment_result.get("p_value"),
                "event_count": result.experiment_result.get("event_count"),
            })

            # ── Stagnation check ──────────────────────────────────────────────
            new_params = hyp.get("experiment_params", {})
            if self._prev_params and _params_are_similar(new_params, self._prev_params):
                self._stagnation_streak += 1
            else:
                self._stagnation_streak = 0
            self._prev_params = new_params

            if self._stagnation_streak >= STAGNATION_CONSECUTIVE_RUNS - 1:
                return self._stop(
                    "stagnated",
                    f"Stagnation detected after {self._runs_completed} runs: "
                    f"the last {STAGNATION_CONSECUTIVE_RUNS} hypotheses proposed "
                    "nearly identical SMA parameters. "
                    "Try a wider LAB_DEFAULT_MODEL or adjust the Researcher prompt.",
                )

            # ── Routing: check Recommender action ─────────────────────────────
            action = result.recommendation.get("action", "")
            if action in TERMINAL_ACTIONS:
                return self._stop(
                    "completed",
                    f"Recommender action '{action}' after run {self._runs_completed}: "
                    f"{result.recommendation.get('rationale', '')}",
                )
            # 'run_variant' or 'needs_more_data' → loop continues

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _stop(self, status: str, reason: str) -> CampaignResult:
        """Persist the final campaign state and return a CampaignResult."""
        with self.conn:
            update_campaign(
                self.conn,
                campaign_id=self._campaign_id,
                status=status,
                runs_completed=self._runs_completed,
                total_calls_used=self._total_calls_used,
                stop_reason=reason,
                completed_at=now_iso(),
            )
        return CampaignResult(
            campaign_id=self._campaign_id,
            status=status,
            stop_reason=reason,
            runs_completed=self._runs_completed,
            total_calls_used=self._total_calls_used,
            run_results=self._run_results,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _params_are_similar(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """
    Return True if two SMA parameter sets are too similar to be worth testing.

    Both short_window and long_window must be within STAGNATION_WINDOW_TOLERANCE
    bars of each other.  horizon_periods is not checked — small horizon changes
    don't constitute genuine hypothesis diversity.
    """
    return (
        abs(int(a.get("short_window", 0)) - int(b.get("short_window", 0)))
        <= STAGNATION_WINDOW_TOLERANCE
        and abs(int(a.get("long_window", 0)) - int(b.get("long_window", 0)))
        <= STAGNATION_WINDOW_TOLERANCE
    )


def _build_prior_context(hypotheses: list[dict]) -> str | None:
    """
    Build the prior-context text injected into the Researcher's user message.

    Returns None if there are no prior hypotheses (first run in a campaign).

    Format injected:
        Previously tested in this campaign (do not repeat these):
          Run 1: short=5, long=20, horizon=10 — REJECT (p=0.1662, n=17)
          Run 2: short=8, long=25, horizon=10 — ACCEPT (p=0.0301, n=22)
        Propose a genuinely different hypothesis with different parameter values.
    """
    if not hypotheses:
        return None

    lines = ["Previously tested in this campaign (do not repeat these):"]
    for i, h in enumerate(hypotheses, start=1):
        params = h.get("parameters", {})
        decision = (h.get("judge_decision") or "unknown").upper()
        p_val = h.get("p_value")
        n = h.get("event_count")
        p_str = f"p={p_val:.4f}" if p_val is not None else "p=?"
        n_str = f"n={n}" if n is not None else "n=?"
        lines.append(
            f"  Run {i}: short={params.get('short_window')}, "
            f"long={params.get('long_window')}, "
            f"horizon={params.get('horizon_periods')} "
            f"— {decision} ({p_str}, {n_str})"
        )
    lines.append(
        "Propose a genuinely different hypothesis with meaningfully different "
        "SMA windows or horizon."
    )
    return "\n".join(lines)


def _with_per_run_budget(cfg: LabConfig, per_run_calls: int) -> LabConfig:
    """
    Return a copy of cfg with max_ollama_calls set to per_run_calls.

    Uses model_copy (Pydantic v2) to create an isolated config instance so
    the original cfg is not mutated.
    """
    return cfg.model_copy(update={"max_ollama_calls": per_run_calls})
