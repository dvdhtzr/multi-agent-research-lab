"""
Unit tests for agent output schemas.

These tests verify that:
  1. Valid model outputs are accepted by the Pydantic models.
  2. Invalid outputs are rejected with clear error messages.

Why we test schemas separately:
  The schemas are the contract between the pipeline and Ollama.  If a
  field is renamed or a constraint is loosened, the whole pipeline breaks.
  These tests catch that without needing Ollama to be running.

C# parallel: unit tests for model classes with data-annotation validation —
verifying that [Required] and [Range] fire correctly.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from lab.agents.schemas import (
    SMAParams,
    ResearcherOutput,
    DetractorOutput,
    JudgeOutput,
    DevilsAdvocateOutput,
    ResearcherRebuttalOutput,
    RecommenderOutput,
    SummarizerOutput,
)


# ─────────────────────────────────────────────────────────────────────────────
# SMAParams
# ─────────────────────────────────────────────────────────────────────────────

def test_sma_params_valid():
    p = SMAParams(short_window=5, long_window=20, horizon_periods=10)
    assert p.short_window == 5


def test_sma_params_long_must_exceed_short():
    with pytest.raises(ValidationError):
        SMAParams(short_window=20, long_window=10, horizon_periods=5)


def test_sma_params_equal_windows_rejected():
    with pytest.raises(ValidationError):
        SMAParams(short_window=10, long_window=10, horizon_periods=5)


# ─────────────────────────────────────────────────────────────────────────────
# ResearcherOutput
# ─────────────────────────────────────────────────────────────────────────────

def test_researcher_output_valid():
    r = ResearcherOutput(
        hypothesis_statement="The 5-bar SMA crossing above the 20-bar SMA predicts a price rise.",
        experiment_type="sma_crossover_followthrough",
        experiment_params=SMAParams(short_window=5, long_window=20, horizon_periods=10),
    )
    assert r.experiment_type == "sma_crossover_followthrough"


def test_researcher_output_rejects_unknown_experiment_type():
    with pytest.raises(ValidationError):
        ResearcherOutput(
            hypothesis_statement="Some hypothesis.",
            experiment_type="invented_by_model",
            experiment_params=SMAParams(short_window=5, long_window=20, horizon_periods=10),
        )


# ─────────────────────────────────────────────────────────────────────────────
# DevilsAdvocateOutput
# ─────────────────────────────────────────────────────────────────────────────

def test_devils_advocate_valid():
    da = DevilsAdvocateOutput(
        objection="The horizon of 5 bars is too short to measure meaningful follow-through.",
        targeted_assumption="5 minutes is a meaningful holding period.",
        confidence=0.7,
    )
    assert da.confidence == 0.7
    assert "horizon" in da.objection


def test_devils_advocate_confidence_out_of_range():
    with pytest.raises(ValidationError, match="confidence"):
        DevilsAdvocateOutput(
            objection="Valid objection text here.",
            targeted_assumption="Some assumption.",
            confidence=1.5,
        )


def test_devils_advocate_objection_too_short():
    with pytest.raises(ValidationError):
        DevilsAdvocateOutput(
            objection="Too short",
            targeted_assumption="Some assumption.",
            confidence=0.5,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ResearcherRebuttalOutput
# ─────────────────────────────────────────────────────────────────────────────

def test_researcher_rebuttal_stands_firm():
    reb = ResearcherRebuttalOutput(
        rebuttal="The 5-bar horizon reflects intraday momentum windows commonly cited in academic literature.",
        concedes_point=False,
    )
    assert reb.concedes_point is False


def test_researcher_rebuttal_concedes():
    reb = ResearcherRebuttalOutput(
        rebuttal="Fair point — the short horizon may conflate noise with signal.",
        concedes_point=True,
    )
    assert reb.concedes_point is True


# ─────────────────────────────────────────────────────────────────────────────
# RecommenderOutput
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("action", [
    "run_variant",
    "archive_confirmed",
    "archive_rejected",
    "needs_more_data",
])
def test_recommender_all_valid_actions(action: str):
    rec = RecommenderOutput(
        action=action,
        rationale="Based on Judge decision and evidence, this is the right next step.",
    )
    assert rec.action == action


def test_recommender_invalid_action():
    with pytest.raises(ValidationError, match="action"):
        RecommenderOutput(
            action="do_nothing",
            rationale="Some rationale here.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# SummarizerOutput
# ─────────────────────────────────────────────────────────────────────────────

def test_summarizer_valid():
    s = SummarizerOutput(
        summary=(
            "The research team tested whether a crossover of the 5-bar average above the 20-bar average "
            "predicted a price gain over the next 15 minutes. "
            "Out of 42 crossovers in the dataset, prices rose afterward 57% of the time. "
            "The probability this result is random chance is about 3%, which is below the team's threshold."
        )
    )
    assert len(s.summary) > 20


def test_summarizer_too_short():
    with pytest.raises(ValidationError):
        SummarizerOutput(summary="Too brief.")


# ─────────────────────────────────────────────────────────────────────────────
# JudgeOutput — trust boundary
# ─────────────────────────────────────────────────────────────────────────────

def test_judge_output_has_no_numerical_field():
    """The Judge's schema must not contain a field for p-value or event count.

    This is the trust-boundary test: the Judge cannot structurally alter
    the deterministic results because there is no field in which to do so.
    """
    model_fields = set(JudgeOutput.model_fields.keys())
    forbidden = {"p_value", "event_count", "win_rate", "positive_count"}
    overlap = model_fields & forbidden
    assert not overlap, (
        f"JudgeOutput contains numerical result field(s): {overlap}. "
        "Remove them — the Judge must not be able to alter the deterministic results."
    )


def test_judge_valid_decisions():
    for decision in ("accept", "reject", "inconclusive"):
        j = JudgeOutput(
            decision=decision,
            rationale="p=0.03 < 0.05, n=42 >= 10, evidence supports the hypothesis.",
            confidence="high",
        )
        assert j.decision == decision
