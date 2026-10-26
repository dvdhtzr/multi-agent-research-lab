"""
4-agent replay pipeline runner (Milestone 4).

Runs a compact deliberation over each walk-forward window:
  [1] Researcher     — proposes short_window + long_window (no horizon)
  [2] Experiment     — deterministic multi-horizon SMA calculation
  [3] Judge          — evaluates results, picks best horizon
  [4] Recommender    — accept_signal | reject_signal | needs_more_data
  [5] Summarizer     — plain-English summary of this window

This is intentionally leaner than the full 8-agent pipeline (no DA/Rebuttal/
Detractor) because replay is about volume: many windows tested quickly.

The budget is enforced identically to PipelineRunner — instructor retries
are counted via tenacity callbacks, not just successful completions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import instructor
import openai
from tenacity import RetryCallState, Retrying, stop_after_attempt, wait_none

from lab.agents.prompts import (
    REPLAY_JUDGE_SYSTEM,
    REPLAY_RECOMMENDER_SYSTEM,
    REPLAY_RESEARCHER_PROMPT,
    REPLAY_RESEARCHER_SYSTEM,
    REPLAY_SUMMARIZER_SYSTEM,
)
from lab.agents.schemas import (
    ReplayJudgeOutput,
    ReplayRecommenderOutput,
    ReplayResearcherOutput,
    ReplaySummarizerOutput,
)
from lab.config import LabConfig
from lab.data.loader import OHLCVCandle
from lab.pipeline.runner import BudgetExceededError
from lab.research.experiments import run_experiment
from lab.research.replay import ReplayStep


@dataclass
class ReplayStepResult:
    """Full provenance for one walk-forward replay step."""
    step_index: int
    window_start: int
    window_end: int
    timestamp_start: str
    timestamp_end: str
    hypothesis: dict            # {experiment_type, experiment_params, hypothesis_statement}
    experiment_result: dict     # MultiHorizonResult.to_dict()["result"]
    experiment_meta: dict       # {method, candles_used}
    judge_decision: dict        # ReplayJudgeOutput fields
    recommendation: dict        # ReplayRecommenderOutput fields
    summarizer_output: dict     # ReplaySummarizerOutput fields
    ollama_calls_used: int
    wall_clock_seconds: float


@dataclass
class ReplayResult:
    """Aggregated result of a full walk-forward replay run."""
    replay_id: str
    status: str                 # 'completed' | 'budget_exhausted' | 'failed'
    steps_completed: int
    total_calls_used: int
    step_results: list[ReplayStepResult] = field(default_factory=list)


class ReplayPipelineRunner:
    """
    Drives the 4-agent replay pipeline for a single walk-forward step.

    Shares the instructor + tenacity retry-counting pattern with PipelineRunner:
    every inference attempt (including validation retries) increments the counter
    and is checked against the budget.

    Args:
        cfg:     Runtime configuration (model, budget limits, etc.)
        step:    The walk-forward window to process.
        budget:  Maximum Ollama calls allowed for this step (typically
                 max_ollama_calls from config, but may be tighter when
                 multiple steps share a replay budget).
    """

    def __init__(
        self,
        cfg: LabConfig,
        step: ReplayStep,
        budget: int | None = None,
        on_step: Callable[[str, Any], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._step = step
        self._calls_used = 0
        self._budget = budget if budget is not None else cfg.max_ollama_calls
        self._on_step = on_step

        raw_client = openai.OpenAI(
            base_url=f"{cfg.ollama_base_url}/v1",
            api_key="ollama",
        )
        self._client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)  # type: ignore[attr-defined]

    @property
    def calls_used(self) -> int:
        return self._calls_used

    def run(self) -> ReplayStepResult:
        """Execute the 4-agent pipeline for this step and return its result."""
        t0 = time.monotonic()
        step = self._step

        # ── [1] Researcher ────────────────────────────────────────────────────
        horizons_str = ", ".join(str(h) for h in self._cfg.replay_horizons)
        researcher_user = (
            REPLAY_RESEARCHER_PROMPT.format(
                window_size=step.window_size,
                step_index=step.step_index + 1,
                total_steps="?",
            )
            + f"\n\nFixed horizons (bars): {horizons_str}"
        )
        researcher_out: ReplayResearcherOutput = self._call(
            REPLAY_RESEARCHER_SYSTEM,
            researcher_user,
            ReplayResearcherOutput,
        )
        if self._on_step:
            self._on_step("researcher", researcher_out)
        params = researcher_out.experiment_params

        # ── [2] Deterministic experiment ──────────────────────────────────────
        exp_dispatch = run_experiment(
            candles=step.candles,
            experiment_type="sma_crossover_multihorizon",
            parameters={
                "short_window": params.short_window,
                "long_window": params.long_window,
                "horizon_list": self._cfg.replay_horizons,
            },
        )
        exp_result = exp_dispatch["result"]
        exp_meta = {
            "method": exp_dispatch["method"],
            "candles_used": exp_dispatch["candles_used"],
        }
        if self._on_step:
            self._on_step("experiment", exp_result)

        # ── [3] Judge ─────────────────────────────────────────────────────────
        horizons_evidence = "\n".join(
            f"  horizon={h['horizon_periods']}bars: "
            f"events={h['event_count']}, "
            f"win_rate={h['win_rate']:.3f}, "
            f"p={h['p_value']:.6f}, "
            f"significant={'yes' if h['statistically_significant'] else 'no'}"
            for h in exp_result.get("horizons", [])
        )
        judge_user = (
            f"SMA configuration: short={params.short_window}, long={params.long_window}\n"
            f"Window: candles {step.window_start}–{step.window_end} "
            f"({step.timestamp_start.isoformat()} → {step.timestamp_end.isoformat()})\n\n"
            f"Multi-horizon results:\n{horizons_evidence}"
        )
        judge_out: ReplayJudgeOutput = self._call(
            REPLAY_JUDGE_SYSTEM,
            judge_user,
            ReplayJudgeOutput,
        )
        if self._on_step:
            self._on_step("judge", judge_out)

        # ── [4] Recommender ───────────────────────────────────────────────────
        rec_user = (
            f"Judge decision: {judge_out.decision} "
            f"(confidence: {judge_out.confidence})\n"
            f"Rationale: {judge_out.rationale}\n"
            f"Recommended horizon: {judge_out.recommended_horizon} bars"
        )
        rec_out: ReplayRecommenderOutput = self._call(
            REPLAY_RECOMMENDER_SYSTEM,
            rec_user,
            ReplayRecommenderOutput,
        )
        if self._on_step:
            self._on_step("recommender", rec_out)

        # ── [5] Summarizer ────────────────────────────────────────────────────
        summary_user = (
            f"SMA configuration tested: short={params.short_window}, "
            f"long={params.long_window}\n"
            f"Multi-horizon results:\n{horizons_evidence}\n"
            f"Judge: {judge_out.decision} | Recommendation: {rec_out.action}"
        )
        summary_out: ReplaySummarizerOutput = self._call(
            REPLAY_SUMMARIZER_SYSTEM,
            summary_user,
            ReplaySummarizerOutput,
        )
        if self._on_step:
            self._on_step("summarizer", summary_out)

        return ReplayStepResult(
            step_index=step.step_index,
            window_start=step.window_start,
            window_end=step.window_end,
            timestamp_start=step.timestamp_start.isoformat(),
            timestamp_end=step.timestamp_end.isoformat(),
            hypothesis={
                "hypothesis_statement": researcher_out.hypothesis_statement,
                "experiment_type": researcher_out.experiment_type,
                "experiment_params": {
                    "short_window": params.short_window,
                    "long_window": params.long_window,
                },
            },
            experiment_result=exp_result,
            experiment_meta=exp_meta,
            judge_decision=judge_out.model_dump(),
            recommendation=rec_out.model_dump(),
            summarizer_output=summary_out.model_dump(),
            ollama_calls_used=self._calls_used,
            wall_clock_seconds=time.monotonic() - t0,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_budget(self) -> None:
        if self._calls_used >= self._budget:
            raise BudgetExceededError(
                f"Replay step budget exhausted ({self._budget} calls)."
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
