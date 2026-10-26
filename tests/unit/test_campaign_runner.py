"""
Unit tests for campaign-level logic in lab/campaign/runner.py.

These tests cover the pure functions and deterministic logic only —
no DB, no Ollama, no instructor.

What is tested:
  - _params_are_similar: tolerance boundary for stagnation detection
  - _build_prior_context: correct text injected into Researcher prompt
  - _with_per_run_budget: correct config mutation for adaptive budgets

C# parallel: unit tests for a service's pure helper methods, isolated from
I/O and external dependencies.
"""
from __future__ import annotations

import pytest

from lab.campaign.runner import (
    STAGNATION_WINDOW_TOLERANCE,
    _build_prior_context,
    _params_are_similar,
    _with_per_run_budget,
)
from lab.config import LabConfig


# ─────────────────────────────────────────────────────────────────────────────
# _params_are_similar
# ─────────────────────────────────────────────────────────────────────────────

def _params(short: int, long: int, horizon: int = 10) -> dict:
    return {"short_window": short, "long_window": long, "horizon_periods": horizon}


def test_identical_params_are_similar():
    assert _params_are_similar(_params(5, 20), _params(5, 20)) is True


def test_params_within_tolerance_are_similar():
    # Both windows within STAGNATION_WINDOW_TOLERANCE bars.
    a = _params(5, 20)
    b = _params(5 + STAGNATION_WINDOW_TOLERANCE, 20 + STAGNATION_WINDOW_TOLERANCE)
    assert _params_are_similar(a, b) is True


def test_params_just_outside_tolerance_not_similar():
    a = _params(5, 20)
    b = _params(5 + STAGNATION_WINDOW_TOLERANCE + 1, 20)
    assert _params_are_similar(a, b) is False


def test_long_window_difference_alone_breaks_similarity():
    a = _params(5, 20)
    b = _params(5, 20 + STAGNATION_WINDOW_TOLERANCE + 1)
    assert _params_are_similar(a, b) is False


def test_horizon_difference_ignored_for_stagnation():
    # horizon_periods does not affect stagnation detection.
    a = _params(5, 20, horizon=5)
    b = _params(5, 20, horizon=50)
    assert _params_are_similar(a, b) is True


# ─────────────────────────────────────────────────────────────────────────────
# _build_prior_context
# ─────────────────────────────────────────────────────────────────────────────

def test_no_prior_context_returns_none():
    assert _build_prior_context([]) is None


def test_single_prior_hypothesis_included():
    hyps = [{
        "statement": "Test hypothesis",
        "experiment_type": "sma_crossover_followthrough",
        "parameters": {"short_window": 5, "long_window": 20, "horizon_periods": 10},
        "judge_decision": "reject",
        "p_value": 0.1662,
        "event_count": 17,
    }]
    ctx = _build_prior_context(hyps)
    assert ctx is not None
    assert "Run 1:" in ctx
    assert "short=5" in ctx
    assert "long=20" in ctx
    assert "REJECT" in ctx
    assert "p=0.1662" in ctx
    assert "n=17" in ctx


def test_multiple_runs_all_included():
    hyps = [
        {
            "parameters": {"short_window": 5, "long_window": 20, "horizon_periods": 10},
            "judge_decision": "reject",
            "p_value": 0.1662,
            "event_count": 17,
        },
        {
            "parameters": {"short_window": 10, "long_window": 40, "horizon_periods": 15},
            "judge_decision": "accept",
            "p_value": 0.0301,
            "event_count": 22,
        },
    ]
    ctx = _build_prior_context(hyps)
    assert "Run 1:" in ctx
    assert "Run 2:" in ctx
    assert "short=10" in ctx
    assert "ACCEPT" in ctx


def test_prior_context_ends_with_instruction():
    hyps = [{
        "parameters": {"short_window": 5, "long_window": 20, "horizon_periods": 10},
        "judge_decision": "reject",
        "p_value": 0.1662,
        "event_count": 17,
    }]
    ctx = _build_prior_context(hyps)
    assert "Propose a genuinely different hypothesis" in ctx


def test_prior_context_handles_missing_p_value():
    hyps = [{
        "parameters": {"short_window": 5, "long_window": 20, "horizon_periods": 10},
        "judge_decision": "inconclusive",
        "p_value": None,
        "event_count": None,
    }]
    ctx = _build_prior_context(hyps)
    assert ctx is not None
    assert "p=?" in ctx
    assert "n=?" in ctx


# ─────────────────────────────────────────────────────────────────────────────
# _with_per_run_budget
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(**kwargs) -> LabConfig:
    defaults = dict(
        ollama_base_url="http://localhost:11434",
        default_model="test-model",
        max_ollama_calls=12,
        max_wall_clock_seconds=300,
        max_campaign_runs=5,
        max_campaign_calls=60,
    )
    defaults.update(kwargs)
    return LabConfig(**defaults)


def test_with_per_run_budget_sets_max_ollama_calls():
    cfg = _make_config(max_ollama_calls=12)
    new_cfg = _with_per_run_budget(cfg, 8)
    assert new_cfg.max_ollama_calls == 8


def test_with_per_run_budget_does_not_mutate_original():
    cfg = _make_config(max_ollama_calls=12)
    _with_per_run_budget(cfg, 8)
    assert cfg.max_ollama_calls == 12  # original unchanged


def test_with_per_run_budget_preserves_other_fields():
    cfg = _make_config(max_ollama_calls=12, default_model="mymodel")
    new_cfg = _with_per_run_budget(cfg, 8)
    assert new_cfg.default_model == "mymodel"
    assert new_cfg.max_campaign_runs == 5
