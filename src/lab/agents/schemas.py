"""
Pydantic models for typed agent input/output.

Each model is the CONTRACT between the pipeline runner and one LLM agent.
instructor enforces that every Ollama response matches the schema before
the pipeline continues — invalid JSON or missing fields cause an automatic
retry, not a crash.

C# parallel: strongly typed DTOs with [Required] and [Range] attributes,
validated at deserialization time rather than discovered at runtime.

Usage:
    from lab.agents.schemas import ResearcherOutput
    result: ResearcherOutput = client.chat.completions.create(
        model=..., messages=[...], response_model=ResearcherOutput
    )
    # result.hypothesis_statement is a validated str — guaranteed non-empty.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Shared parameter model
# ─────────────────────────────────────────────────────────────────────────────

class SMAParams(BaseModel):
    """Parameters for the sma_crossover_followthrough experiment."""
    short_window: int = Field(ge=2, le=30)
    long_window: int = Field(ge=5, le=100)
    horizon_periods: int = Field(ge=1, le=120)

    @model_validator(mode="after")
    def long_must_exceed_short(self) -> "SMAParams":
        if self.long_window <= self.short_window:
            raise ValueError(
                f"long_window ({self.long_window}) must be > short_window ({self.short_window})"
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Milestone 1 agent outputs
# ─────────────────────────────────────────────────────────────────────────────

class ResearcherOutput(BaseModel):
    """The Researcher's structured hypothesis proposal."""
    hypothesis_statement: str = Field(min_length=10)
    experiment_type: Literal["sma_crossover_followthrough"]
    experiment_params: SMAParams


class DetractorOutput(BaseModel):
    """The Detractor's structured challenge to the experiment results."""
    primary_challenge: str = Field(min_length=10)
    methodological_concern: str = Field(min_length=10)
    alternative_explanation: str = Field(min_length=10)


class JudgeOutput(BaseModel):
    """The Judge's structured final decision.

    The schema intentionally has no field for numerical results — the Judge
    cannot produce, modify, or restate the p-value or event count.  That
    architectural constraint is what makes the trust boundary hold.
    """
    decision: Literal["accept", "reject", "inconclusive"]
    rationale: str = Field(min_length=20)
    confidence: Literal["high", "medium", "low"]


# ─────────────────────────────────────────────────────────────────────────────
# Milestone 2 agent outputs
# ─────────────────────────────────────────────────────────────────────────────

class DevilsAdvocateOutput(BaseModel):
    """The Devil's Advocate's structured pre-experiment challenge.

    The DA challenges the hypothesis *before* the experiment runs, forcing the
    Researcher to justify the design choice.  Structurally different from the
    Detractor, which challenges the *results* after the experiment.
    """
    objection: str = Field(min_length=10)
    targeted_assumption: str = Field(
        min_length=5,
        description="The specific assumption the DA is challenging.",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How strongly the DA believes the objection undermines the hypothesis.",
    )


class ResearcherRebuttalOutput(BaseModel):
    """The Researcher's one-round response to the Devil's Advocate.

    FUTURE: allow the Researcher to revise experiment parameters here
    if the DA's objection reveals a design flaw (currently locked to keep
    the pipeline simple).
    """
    rebuttal: str = Field(min_length=10)
    concedes_point: bool = Field(
        description="True if the Researcher acknowledges the DA raised a valid point.",
    )


class RecommenderOutput(BaseModel):
    """The Recommender's suggested next action after the full deliberation.

    FUTURE: add suggested_params (SMAParams | None) for the run_variant action
    once auto-triggering is implemented.

    WARNING (COST-SAFETY): if run_variant is ever made to auto-trigger a new
    Ollama run, that feature must never override hard spending limits and must
    be disabled by default for paid API providers.
    """
    action: Literal[
        "run_variant",
        "archive_confirmed",
        "archive_rejected",
        "needs_more_data",
    ]
    rationale: str = Field(min_length=10)


class SummarizerOutput(BaseModel):
    """A plain-English summary of the entire research session."""
    summary: str = Field(
        min_length=20,
        description="3-5 sentences suitable for a non-technical reader.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Milestone 4 — Replay pipeline schemas
# ─────────────────────────────────────────────────────────────────────────────

class SMAParamsNoHorizon(BaseModel):
    """SMA window parameters without a horizon (replay — horizons are fixed by config)."""
    short_window: int = Field(ge=2, le=30)
    long_window: int = Field(ge=5, le=100)

    @model_validator(mode="after")
    def long_must_exceed_short(self) -> "SMAParamsNoHorizon":
        if self.long_window <= self.short_window:
            raise ValueError(
                f"long_window ({self.long_window}) must be > short_window ({self.short_window})"
            )
        return self


class ReplayResearcherOutput(BaseModel):
    """Researcher output for the replay pipeline.

    Uses sma_crossover_multihorizon — horizons are supplied by the engine,
    not proposed by the agent.
    """
    hypothesis_statement: str = Field(min_length=10)
    experiment_type: Literal["sma_crossover_multihorizon"]
    experiment_params: SMAParamsNoHorizon


class ReplayJudgeOutput(BaseModel):
    """Judge output for a multi-horizon replay step.

    The schema has no numerical fields — the Judge interprets the evidence
    but cannot alter the deterministic p-values or event counts.
    """
    decision: Literal["accept", "reject", "inconclusive"]
    rationale: str = Field(min_length=20)
    confidence: Literal["high", "medium", "low"]
    # Interpretive choice among the config-defined horizons; not a
    # free-form number — the Judge must pick from the horizons shown.
    recommended_horizon: int = Field(
        description="Which horizon (bars) the Judge considers most promising, "
                    "chosen from the horizons listed in the evidence.",
        ge=1,
    )


class ReplayRecommenderOutput(BaseModel):
    """Recommender output for a replay step."""
    action: Literal["accept_signal", "reject_signal", "needs_more_data"]
    rationale: str = Field(min_length=10)


class ReplaySummarizerOutput(BaseModel):
    """Plain-English summary of a single replay step."""
    summary: str = Field(
        min_length=20,
        description="2-3 sentences describing this window's findings.",
    )
