"""
Comparative evaluation runner for Milestone 5.

Runs three approaches on the same dataset with the same LLM budget and
produces a side-by-side comparison:

  1. Baseline   — 0 LLM calls, random parameter grid search
  2. Single     — 1 LLM call per run (Researcher only, no deliberation)
  3. Multi      — ~7 LLM calls per run (full 8-agent CampaignRunner)

The comparison answers: given a fixed LLM budget, which approach finds
the most statistically significant patterns?

The key metric is calls-per-significant-finding.  A cheaper approach
that finds the same patterns is strictly better.

Persists to SQLite so results can be reviewed later.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from lab.campaign.runner import CampaignResult, CampaignRunner
from lab.config import LabConfig
from lab.data.loader import OHLCVCandle
from lab.db.repository import new_id, now_iso
from lab.evaluation.baseline import BaselineEvaluator, BaselineResult
from lab.evaluation.single_agent import SingleAgentResult, SingleAgentRunner


@dataclass
class EvaluationSummary:
    """Key metrics for one approach in the comparison."""
    approach: str           # 'baseline' | 'single_agent' | 'multi_agent'
    runs: int
    significant_count: int
    significant_rate: float
    best_p_value: float
    best_params: dict | None
    calls_used: int
    calls_per_significant: float   # inf if 0 significant findings


@dataclass
class ComparisonResult:
    """Full side-by-side comparison of all three approaches."""
    comparison_id: str
    created_at: str
    n_samples: int
    llm_budget: int

    baseline: BaselineResult
    single_agent: SingleAgentResult
    multi_agent: CampaignResult

    @property
    def summaries(self) -> list[EvaluationSummary]:
        """Return one EvaluationSummary per approach, ordered baseline → single → multi."""
        return [
            _summarise("baseline", self.baseline.n_samples,
                       self.baseline.significant_count, self.baseline.best_p_value,
                       self.baseline.best_params, 0),
            _summarise("single_agent", self.single_agent.n_runs,
                       self.single_agent.significant_count, self.single_agent.best_p_value,
                       self.single_agent.best_params, self.single_agent.total_calls_used),
            _summarise_campaign("multi_agent", self.multi_agent),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "created_at": self.created_at,
            "n_samples": self.n_samples,
            "llm_budget": self.llm_budget,
            "baseline": self.baseline.summary_dict(),
            "single_agent": self.single_agent.summary_dict(),
            "multi_agent": _campaign_summary_dict(self.multi_agent),
        }


def _summarise(
    approach: str,
    runs: int,
    significant_count: int,
    best_p: float,
    best_params: dict | None,
    calls: int,
) -> EvaluationSummary:
    rate = significant_count / runs if runs > 0 else 0.0
    cps = calls / significant_count if significant_count > 0 else float("inf")
    return EvaluationSummary(
        approach=approach,
        runs=runs,
        significant_count=significant_count,
        significant_rate=rate,
        best_p_value=best_p,
        best_params=best_params,
        calls_used=calls,
        calls_per_significant=cps,
    )


def _summarise_campaign(approach: str, result: CampaignResult) -> EvaluationSummary:
    sig_count = 0
    best_p = 1.0
    best_params = None

    for run in result.run_results:
        exp = run.experiment_result
        p = exp.get("p_value", 1.0)
        n = exp.get("event_count", 0)
        sig = exp.get("statistically_significant", False)
        met = exp.get("minimum_events_met", False)

        if sig and met:
            sig_count += 1
        if met and p < best_p:
            best_p = p
            best_params = run.hypothesis.get("experiment_params")

    runs = result.runs_completed
    calls = result.total_calls_used
    rate = sig_count / runs if runs > 0 else 0.0
    cps = calls / sig_count if sig_count > 0 else float("inf")

    return EvaluationSummary(
        approach=approach,
        runs=runs,
        significant_count=sig_count,
        significant_rate=rate,
        best_p_value=best_p,
        best_params=best_params,
        calls_used=calls,
        calls_per_significant=cps,
    )


def _campaign_summary_dict(result: CampaignResult) -> dict[str, Any]:
    s = _summarise_campaign("multi_agent", result)
    return {
        "n_runs": s.runs,
        "significant_count": s.significant_count,
        "significant_rate": round(s.significant_rate, 4),
        "best_p_value": round(s.best_p_value, 6),
        "best_params": s.best_params,
        "total_calls_used": s.calls_used,
        "status": result.status,
        "stop_reason": result.stop_reason,
    }


class ComparisonRunner:
    """
    Orchestrates the three-way comparative evaluation.

    Args:
        cfg:     Runtime configuration.  Uses:
                   - compare_n_samples (baseline + single-agent)
                   - compare_llm_budget (single-agent and multi-agent LLM calls)
                   - max_campaign_runs  (multi-agent via CampaignRunner)
        candles: Full OHLCV candle series.
        conn:    Open SQLite connection (schema already applied).
    """

    def __init__(
        self,
        cfg: LabConfig,
        candles: list[OHLCVCandle],
        conn: sqlite3.Connection,
    ) -> None:
        self._cfg = cfg
        self._candles = candles
        self._conn = conn

    def run(self) -> ComparisonResult:
        """
        Run all three approaches and return the comparison.

        Order of execution:
          1. Baseline (no LLM — fast)
          2. Single-agent (1 call/run)
          3. Multi-agent (CampaignRunner)
        """
        comparison_id = new_id()
        ts = now_iso()
        n_samples = self._cfg.compare_n_samples
        llm_budget = self._cfg.compare_llm_budget

        # ── 1. Baseline ───────────────────────────────────────────────────────
        baseline_result = BaselineEvaluator(self._candles).run(n_samples=n_samples)

        # ── 2. Single-agent ───────────────────────────────────────────────────
        single_runner = SingleAgentRunner(
            cfg=self._cfg,
            candles=self._candles,
            budget=llm_budget,
        )
        single_result = single_runner.run(n_runs=n_samples)

        # ── 3. Multi-agent (CampaignRunner) ──────────────────────────────────
        # Give the multi-agent runner the full LLM budget with a reasonable
        # per-run cap so it uses the same total call ceiling.
        multi_cfg = self._cfg.model_copy(update={
            "max_campaign_calls": llm_budget,
        })
        multi_runner = CampaignRunner(multi_cfg, self._candles, self._conn)
        multi_result = multi_runner.run()

        return ComparisonResult(
            comparison_id=comparison_id,
            created_at=ts,
            n_samples=n_samples,
            llm_budget=llm_budget,
            baseline=baseline_result,
            single_agent=single_result,
            multi_agent=multi_result,
        )
