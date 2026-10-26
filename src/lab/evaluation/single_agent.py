"""
Single-agent evaluator for Milestone 5 comparative evaluation.

Runs a minimal pipeline of just one LLM call (Researcher) followed by
the deterministic experiment.  No deliberation, no Devil's Advocate,
no Judge or Recommender.

This isolates the question: does the LLM researcher propose better
parameters than random chance alone?  Comparing it to the baseline
answers that question without the confounding cost of the full pipeline.

Budget accounting uses the same tenacity retry-counting approach as
PipelineRunner and ReplayPipelineRunner — every inference attempt
(including instructor validation retries) is counted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import instructor
import openai
from tenacity import RetryCallState, Retrying, stop_after_attempt, wait_none

from lab.agents.prompts import RESEARCHER_PROMPT, RESEARCHER_SYSTEM
from lab.agents.schemas import ResearcherOutput
from lab.config import LabConfig
from lab.data.loader import OHLCVCandle
from lab.pipeline.runner import BudgetExceededError
from lab.research.experiments import run_sma_crossover_followthrough


@dataclass
class SingleAgentRunResult:
    """Result of one Researcher call + deterministic experiment."""
    hypothesis_statement: str
    short_window: int
    long_window: int
    horizon_periods: int
    event_count: int
    positive_count: int
    win_rate: float
    p_value: float
    statistically_significant: bool
    minimum_events_met: bool
    calls_used: int     # >= 1; > 1 if instructor retried

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_statement": self.hypothesis_statement,
            "short_window": self.short_window,
            "long_window": self.long_window,
            "horizon_periods": self.horizon_periods,
            "event_count": self.event_count,
            "positive_count": self.positive_count,
            "win_rate": self.win_rate,
            "p_value": self.p_value,
            "statistically_significant": self.statistically_significant,
            "minimum_events_met": self.minimum_events_met,
            "calls_used": self.calls_used,
        }


@dataclass
class SingleAgentResult:
    """Aggregate result of the single-agent evaluation."""
    n_runs: int
    significant_count: int
    best_p_value: float
    best_params: dict | None
    all_results: list[SingleAgentRunResult] = field(default_factory=list)
    total_calls_used: int = 0

    @property
    def significant_rate(self) -> float:
        if self.n_runs == 0:
            return 0.0
        return self.significant_count / self.n_runs

    def summary_dict(self) -> dict[str, Any]:
        return {
            "n_runs": self.n_runs,
            "significant_count": self.significant_count,
            "significant_rate": round(self.significant_rate, 4),
            "best_p_value": round(self.best_p_value, 6),
            "best_params": self.best_params,
            "total_calls_used": self.total_calls_used,
        }


class SingleAgentRunner:
    """
    Minimal single-LLM-call research runner.

    Each run calls the Researcher once, receives a hypothesis with
    experiment parameters, runs the deterministic experiment, and
    records the result.  No deliberation pipeline.

    Args:
        cfg:     Runtime configuration.
        candles: Full OHLCV candle series.
        budget:  Maximum total LLM calls across all runs.
    """

    def __init__(
        self,
        cfg: LabConfig,
        candles: list[OHLCVCandle],
        budget: int,
    ) -> None:
        self._cfg = cfg
        self._candles = candles
        self._budget = budget
        self._calls_used = 0

        raw_client = openai.OpenAI(
            base_url=f"{cfg.ollama_base_url}/v1",
            api_key="ollama",
        )
        self._client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)  # type: ignore[attr-defined]

    @property
    def calls_used(self) -> int:
        return self._calls_used

    def run(self, n_runs: int) -> SingleAgentResult:
        """
        Execute up to `n_runs` single-agent evaluations, stopping early if
        the budget is exhausted.

        Returns:
            SingleAgentResult with per-run detail and aggregate statistics.
        """
        results: list[SingleAgentRunResult] = []

        for _ in range(n_runs):
            if self._calls_used >= self._budget:
                break
            try:
                run_result = self._run_one()
            except BudgetExceededError:
                break
            results.append(run_result)

        significant = [r for r in results if r.statistically_significant and r.minimum_events_met]
        significant_count = len(significant)

        if results:
            best = min(
                (r for r in results if r.minimum_events_met),
                key=lambda r: r.p_value,
                default=results[0],
            )
            best_p = best.p_value
            best_params = {
                "short_window": best.short_window,
                "long_window": best.long_window,
                "horizon_periods": best.horizon_periods,
            }
        else:
            best_p = 1.0
            best_params = None

        return SingleAgentResult(
            n_runs=len(results),
            significant_count=significant_count,
            best_p_value=best_p,
            best_params=best_params,
            all_results=results,
            total_calls_used=self._calls_used,
        )

    def _run_one(self) -> SingleAgentRunResult:
        """One Researcher call → deterministic experiment."""
        calls_before = self._calls_used

        researcher_out: ResearcherOutput = self._call(
            RESEARCHER_SYSTEM,
            RESEARCHER_PROMPT,
            ResearcherOutput,
        )

        params = researcher_out.experiment_params
        try:
            exp = run_sma_crossover_followthrough(
                candles=self._candles,
                short_window=params.short_window,
                long_window=params.long_window,
                horizon_periods=params.horizon_periods,
            )
        except ValueError as exc:
            # Invalid params despite schema validation — treat as failed run
            raise BudgetExceededError(f"Invalid params from Researcher: {exc}") from exc

        calls_this_run = self._calls_used - calls_before
        return SingleAgentRunResult(
            hypothesis_statement=researcher_out.hypothesis_statement,
            short_window=params.short_window,
            long_window=params.long_window,
            horizon_periods=params.horizon_periods,
            event_count=exp.event_count,
            positive_count=exp.positive_count,
            win_rate=exp.win_rate,
            p_value=exp.p_value,
            statistically_significant=exp.statistically_significant,
            minimum_events_met=exp.minimum_events_met,
            calls_used=calls_this_run,
        )

    def _check_budget(self) -> None:
        if self._calls_used >= self._budget:
            raise BudgetExceededError(
                f"Single-agent budget exhausted ({self._budget} calls)."
            )

    def _call(self, system: str, user: str, response_model: type):
        """Make one instructor call, counting every inference attempt."""
        self._check_budget()

        def _before_retry(retry_state: RetryCallState) -> None:
            if retry_state.attempt_number > 1:
                self._calls_used += 1
                self._check_budget()

        result = self._client.chat.completions.create(
            model=self._cfg.default_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_model=response_model,
            max_retries=Retrying(
                stop=stop_after_attempt(3),
                wait=wait_none(),
                before=_before_retry,
                reraise=True,
            ),
        )
        self._calls_used += 1
        return result
