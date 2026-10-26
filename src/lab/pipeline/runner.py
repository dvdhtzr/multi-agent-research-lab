"""
8-agent linear research pipeline using instructor + Ollama.

Architecture overview
─────────────────────
Each of the 8 steps is either an LLM call (steps 1-3, 5-8) or a pure-Python
deterministic calculation (step 4).  For the LLM steps, instructor wraps the
Ollama client and enforces that the model's response matches the expected
Pydantic schema before the pipeline advances — invalid or incomplete responses
trigger an automatic retry, not a crash.

Pipeline
────────
  1. Researcher        — proposes a testable hypothesis
  2. Devil's Advocate  — challenges it before the experiment runs
  3. Researcher Rebuttal — responds to the DA's objection
  4. [Deterministic experiment — scipy.stats.binomtest, no LLM]
  5. Detractor         — challenges the results post-experiment
  6. Judge             — accept / reject / inconclusive
  7. Recommender       — suggests the next action
  8. Summarizer        — plain-English summary for the research record

Trust boundary
──────────────
The LLM agents receive the deterministic result as text in their context
message, but their OUTPUT schema contains no field for numerical values.
The Judge's schema is `{decision, rationale, confidence}` — there is nowhere
to put a different p-value or event count.  The trust boundary is enforced
by the type system, not by a policy prompt.

C# parallel: a typed workflow pipeline where each step has an explicit input
and output contract.  Here Pydantic + instructor play the role of C#'s type
system: if the model produces output that doesn't match the contract, the
step fails and retries rather than propagating bad data.

Cost-safety note
────────────────
This pipeline is designed for locally-hosted, non-metered Ollama inference.
The `max_ollama_calls` and `max_wall_clock_seconds` budgets are enforced
before each LLM call.  These limits must NEVER be overridden by agent
decisions, and this pipeline must NOT be adapted for paid API providers
without independently enforced spending ceilings.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import instructor
from openai import OpenAI
from tenacity import RetryCallState, Retrying, stop_after_attempt, wait_none

from lab.agents.prompts import (
    RESEARCHER_SYSTEM, RESEARCHER_PROMPT,
    DEVILS_ADVOCATE_SYSTEM, DEVILS_ADVOCATE_PROMPT,
    RESEARCHER_REBUTTAL_SYSTEM, RESEARCHER_REBUTTAL_PROMPT,
    DETRACTOR_SYSTEM,
    JUDGE_SYSTEM,
    RECOMMENDER_SYSTEM, RECOMMENDER_PROMPT,
    SUMMARIZER_SYSTEM, SUMMARIZER_PROMPT,
)
from lab.agents.schemas import (
    ResearcherOutput,
    DevilsAdvocateOutput,
    ResearcherRebuttalOutput,
    DetractorOutput,
    JudgeOutput,
    RecommenderOutput,
    SummarizerOutput,
)
from lab.config import LabConfig
from lab.data.loader import OHLCVCandle
from lab.research.experiments import run_experiment
from lab.research.validator import validate_experiment_spec


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class BudgetExceededError(RuntimeError):
    """Raised when the pipeline would exceed a hard resource limit."""


class ExperimentValidationError(ValueError):
    """Raised when the Researcher's proposed experiment spec is invalid."""


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    The complete output of one research pipeline run.

    All fields are plain dicts so callers (CLI, tests) can work with them
    without importing the agent schema classes.

    C# parallel: an immutable record type holding the combined output of
    each pipeline stage.
    """
    run_id: str
    hypothesis: dict[str, Any]
    devils_advocate: dict[str, Any]
    researcher_rebuttal: dict[str, Any]
    experiment_result: dict[str, Any]
    experiment_meta: dict[str, Any]
    detractor_arguments: dict[str, Any]
    judge_decision: dict[str, Any]
    recommendation: dict[str, Any]
    summarizer_output: dict[str, Any]
    # Hardcoded False: the judge's schema has no field for numerical results,
    # so it structurally cannot alter them.
    result_integrity_check: dict[str, Any] = field(
        default_factory=lambda: {"judge_altered_result": False}
    )
    ollama_calls_used: int = 0
    wall_clock_seconds: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRunner:
    """
    Executes the 8-agent linear research pipeline.

    Instantiate once per research run; call `run(run_id)` to execute.

    Args:
        cfg:     Application configuration (Ollama URL, model, budgets).
        candles: The loaded OHLCV dataset.  Passed directly to the
                 deterministic experiment — no network call needed.
    """

    def __init__(
        self,
        cfg: LabConfig,
        candles: list[OHLCVCandle],
        on_step: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.candles = candles
        self._on_step = on_step
        # instructor wraps the Ollama OpenAI-compatible client and handles
        # structured-output enforcement and retries.
        #
        # C# parallel: a typed HTTP client (Refit) with a retry policy (Polly)
        # that re-tries on deserialization failure.
        self._client = instructor.from_openai(
            OpenAI(
                base_url=f"{cfg.ollama_base_url}/v1",
                api_key="ollama",  # Ollama's API ignores the key value
            ),
            mode=instructor.Mode.JSON,
        )
        self._calls_used: int = 0
        self._start_time: float = 0.0

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, run_id: str, prior_context: str | None = None) -> PipelineResult:
        """
        Execute the full 8-agent pipeline and return a PipelineResult.

        Args:
            run_id:        Unique identifier for this run (stored in DB).
            prior_context: Optional text injected into the Researcher's user
                           message, listing previously tested hypotheses so the
                           model proposes something novel.  Provided by the
                           CampaignRunner when running multiple iterations.

        Raises:
            BudgetExceededError: if the call count or wall-clock limit is hit.
            ExperimentValidationError: if the Researcher's spec is invalid.
            openai.APIConnectionError: if Ollama is not reachable.
        """
        self._calls_used = 0
        self._start_time = time.monotonic()

        researcher_prompt = RESEARCHER_PROMPT
        if prior_context:
            researcher_prompt = f"{RESEARCHER_PROMPT}\n\n{prior_context}"

        # ── Step 1: Researcher ────────────────────────────────────────────────
        researcher_out: ResearcherOutput = self._call(
            RESEARCHER_SYSTEM,
            researcher_prompt,
            ResearcherOutput,
        )
        if self._on_step:
            self._on_step("researcher", researcher_out)

        # ── Step 2: Devil's Advocate ──────────────────────────────────────────
        da_out: DevilsAdvocateOutput = self._call(
            DEVILS_ADVOCATE_SYSTEM,
            _da_user(researcher_out),
            DevilsAdvocateOutput,
        )
        if self._on_step:
            self._on_step("devils_advocate", da_out)

        # ── Step 3: Researcher Rebuttal ───────────────────────────────────────
        rebuttal_out: ResearcherRebuttalOutput = self._call(
            RESEARCHER_REBUTTAL_SYSTEM,
            _rebuttal_user(researcher_out, da_out),
            ResearcherRebuttalOutput,
        )
        if self._on_step:
            self._on_step("researcher_rebuttal", rebuttal_out)

        # ── Step 4: Deterministic experiment (no LLM) ────────────────────────
        params = researcher_out.experiment_params
        validation = validate_experiment_spec(
            researcher_out.experiment_type,
            params.model_dump(),
        )
        if not validation.is_valid:
            raise ExperimentValidationError(
                f"Researcher proposed an invalid experiment spec: {validation.errors}"
            )

        exp_outcome = run_experiment(
            self.candles,
            researcher_out.experiment_type,
            params.model_dump(),
        )
        exp_result: dict[str, Any] = exp_outcome["result"]
        exp_meta: dict[str, Any] = {
            "method": exp_outcome["method"],
            "candles_used": exp_outcome["candles_used"],
        }
        if self._on_step:
            self._on_step("experiment", exp_result)

        # ── Step 5: Detractor ─────────────────────────────────────────────────
        detractor_out: DetractorOutput = self._call(
            DETRACTOR_SYSTEM,
            _detractor_user(researcher_out, exp_result),
            DetractorOutput,
        )
        if self._on_step:
            self._on_step("detractor", detractor_out)

        # ── Step 6: Judge ─────────────────────────────────────────────────────
        judge_out: JudgeOutput = self._call(
            JUDGE_SYSTEM,
            _judge_user(researcher_out, da_out, rebuttal_out, exp_result, detractor_out),
            JudgeOutput,
        )
        if self._on_step:
            self._on_step("judge", judge_out)

        # ── Step 7: Recommender ───────────────────────────────────────────────
        recommender_out: RecommenderOutput = self._call(
            RECOMMENDER_SYSTEM,
            _recommender_user(researcher_out, da_out, rebuttal_out, exp_result,
                              detractor_out, judge_out),
            RecommenderOutput,
        )
        if self._on_step:
            self._on_step("recommender", recommender_out)

        # ── Step 8: Summarizer ────────────────────────────────────────────────
        summarizer_out: SummarizerOutput = self._call(
            SUMMARIZER_SYSTEM,
            _summarizer_user(researcher_out, exp_result, judge_out, recommender_out),
            SummarizerOutput,
        )
        if self._on_step:
            self._on_step("summarizer", summarizer_out)

        elapsed = time.monotonic() - self._start_time

        return PipelineResult(
            run_id=run_id,
            hypothesis=researcher_out.model_dump(),
            devils_advocate=da_out.model_dump(),
            researcher_rebuttal=rebuttal_out.model_dump(),
            experiment_result=exp_result,
            experiment_meta=exp_meta,
            detractor_arguments=detractor_out.model_dump(),
            judge_decision=judge_out.model_dump(),
            recommendation=recommender_out.model_dump(),
            summarizer_output=summarizer_out.model_dump(),
            ollama_calls_used=self._calls_used,
            wall_clock_seconds=round(elapsed, 1),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call(self, system: str, user: str, response_model: type) -> Any:
        """
        Make one LLM call and return the validated typed response.

        Checks budget limits before the call and before each validation retry.
        instructor retries on Pydantic validation failures; every such retry
        is a real model invocation that counts against the budget.

        Budget accounting: the `before` hook on the tenacity Retrying instance
        is called before each attempt.  attempt_number == 1 is the original call
        (counted after the call returns); attempt_number > 1 is a retry (counted
        before that attempt fires, and the budget is re-checked).
        """
        self._check_budget()

        def _before_retry(retry_state: RetryCallState) -> None:
            if retry_state.attempt_number > 1:
                # A validation retry — counts as an additional inference call.
                self._calls_used += 1
                self._check_budget()

        result = self._client.chat.completions.create(
            model=self.cfg.default_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_model=response_model,
            max_retries=Retrying(
                stop=stop_after_attempt(3),   # original + up to 2 retries
                wait=wait_none(),
                before=_before_retry,
                reraise=True,
            ),
        )
        self._calls_used += 1  # count the original successful (or last) attempt
        return result

    def _check_budget(self) -> None:
        """Raise BudgetExceededError if either hard limit has been reached."""
        if self._calls_used >= self.cfg.max_ollama_calls:
            raise BudgetExceededError(
                f"Ollama call budget exhausted ({self.cfg.max_ollama_calls} calls). "
                "Increase LAB_MAX_OLLAMA_CALLS in .env to allow more calls per run."
            )
        elapsed = time.monotonic() - self._start_time
        if elapsed >= self.cfg.max_wall_clock_seconds:
            raise BudgetExceededError(
                f"Wall-clock budget exhausted ({self.cfg.max_wall_clock_seconds}s). "
                "Increase LAB_MAX_WALL_CLOCK_SECONDS in .env or use a faster model."
            )


# ─────────────────────────────────────────────────────────────────────────────
# User message builders
#
# Each function constructs the "user" turn for one agent.  The context from
# previous steps is embedded directly in the message text.
#
# Keeping these as module-level functions (not methods) makes them easy to
# test independently and keeps the PipelineRunner class focused on orchestration.
# ─────────────────────────────────────────────────────────────────────────────

def _da_user(researcher: ResearcherOutput) -> str:
    p = researcher.experiment_params
    return (
        f"{DEVILS_ADVOCATE_PROMPT}\n\n"
        f"Hypothesis: {researcher.hypothesis_statement}\n"
        f"Experiment: {researcher.experiment_type}\n"
        f"Parameters:\n"
        f"  short_window     = {p.short_window}\n"
        f"  long_window      = {p.long_window}\n"
        f"  horizon_periods  = {p.horizon_periods} bars"
    )


def _rebuttal_user(
    researcher: ResearcherOutput,
    da: DevilsAdvocateOutput,
) -> str:
    p = researcher.experiment_params
    return (
        f"{RESEARCHER_REBUTTAL_PROMPT}\n\n"
        f"Your hypothesis: {researcher.hypothesis_statement}\n"
        f"Parameters: short={p.short_window}, long={p.long_window}, "
        f"horizon={p.horizon_periods}\n\n"
        f"Devil's Advocate raised:\n"
        f"  Objection:           {da.objection}\n"
        f"  Targeted assumption: {da.targeted_assumption}\n"
        f"  Confidence:          {da.confidence:.0%}"
    )


def _detractor_user(
    researcher: ResearcherOutput,
    exp_result: dict[str, Any],
) -> str:
    p = researcher.experiment_params
    return (
        f"A hypothesis has been tested. Challenge the interpretation and methodology.\n\n"
        f"Hypothesis: {researcher.hypothesis_statement}\n"
        f"Parameters: short={p.short_window}, long={p.long_window}, "
        f"horizon={p.horizon_periods}\n\n"
        f"--- DETERMINISTIC RESULTS (do not dispute these numbers) ---\n"
        f"Events found:                        {exp_result.get('event_count')}\n"
        f"Win rate:                            {exp_result.get('win_rate', 0):.1%}\n"
        f"p-value:                             {exp_result.get('p_value', 1):.4f}\n"
        f"Statistically significant (p<0.05):  {exp_result.get('statistically_significant')}\n"
        f"Minimum events met (n≥10):           {exp_result.get('minimum_events_met')}\n"
        f"Candles analysed:                    {exp_result.get('candles_used')}"
    )


def _judge_user(
    researcher: ResearcherOutput,
    da: DevilsAdvocateOutput,
    rebuttal: ResearcherRebuttalOutput,
    exp_result: dict[str, Any],
    detractor: DetractorOutput,
) -> str:
    p = researcher.experiment_params
    concedes = "conceded" if rebuttal.concedes_point else "stood firm"
    return (
        f"Make the final decision based on all evidence.\n\n"
        f"Hypothesis: {researcher.hypothesis_statement}\n"
        f"Parameters: short={p.short_window}, long={p.long_window}, "
        f"horizon={p.horizon_periods}\n\n"
        f"Pre-experiment deliberation:\n"
        f"  DA objection:    {da.objection}\n"
        f"  Researcher ({concedes}): {rebuttal.rebuttal}\n\n"
        f"--- DETERMINISTIC RESULTS (do not alter these numbers) ---\n"
        f"Events:       {exp_result.get('event_count')}\n"
        f"Win rate:     {exp_result.get('win_rate', 0):.1%}\n"
        f"p-value:      {exp_result.get('p_value', 1):.4f}\n"
        f"Significant:  {exp_result.get('statistically_significant')}\n"
        f"Min sample:   {exp_result.get('minimum_events_met')}\n\n"
        f"Post-experiment:\n"
        f"  Detractor primary challenge:    {detractor.primary_challenge}\n"
        f"  Methodological concern:         {detractor.methodological_concern}\n"
        f"  Alternative explanation:        {detractor.alternative_explanation}"
    )


def _recommender_user(
    researcher: ResearcherOutput,
    da: DevilsAdvocateOutput,
    rebuttal: ResearcherRebuttalOutput,
    exp_result: dict[str, Any],
    detractor: DetractorOutput,
    judge: JudgeOutput,
) -> str:
    p = researcher.experiment_params
    concedes = "conceded point" if rebuttal.concedes_point else "stood firm"
    return (
        f"{RECOMMENDER_PROMPT}\n\n"
        f"Hypothesis: {researcher.hypothesis_statement}\n"
        f"Parameters: short={p.short_window}, long={p.long_window}, "
        f"horizon={p.horizon_periods}\n\n"
        f"Pre-experiment:\n"
        f"  DA: {da.objection}\n"
        f"  Researcher ({concedes}): {rebuttal.rebuttal}\n\n"
        f"Results:\n"
        f"  Events: {exp_result.get('event_count')}, "
        f"win rate: {exp_result.get('win_rate', 0):.1%}, "
        f"p={exp_result.get('p_value', 1):.4f}\n"
        f"  Significant: {exp_result.get('statistically_significant')} | "
        f"Min sample: {exp_result.get('minimum_events_met')}\n\n"
        f"Post-experiment:\n"
        f"  Detractor: {detractor.primary_challenge}\n\n"
        f"Judge decision: {judge.decision} (confidence: {judge.confidence})\n"
        f"Judge rationale: {judge.rationale}"
    )


def _summarizer_user(
    researcher: ResearcherOutput,
    exp_result: dict[str, Any],
    judge: JudgeOutput,
    recommender: RecommenderOutput,
) -> str:
    p = researcher.experiment_params
    return (
        f"{SUMMARIZER_PROMPT}\n\n"
        f"Hypothesis: {researcher.hypothesis_statement}\n"
        f"Parameters: short={p.short_window}, long={p.long_window}, "
        f"horizon={p.horizon_periods}\n\n"
        f"Results: {exp_result.get('event_count')} events, "
        f"{exp_result.get('win_rate', 0):.1%} win rate, "
        f"p={exp_result.get('p_value', 1):.4f}\n"
        f"Statistically significant: {exp_result.get('statistically_significant')}\n\n"
        f"Judge decision: {judge.decision}\n"
        f"Recommendation: {recommender.action}"
    )
