"""
Unit tests for the experiment specification validator.

These tests verify that the trust boundary is correctly enforced:
agents can only request experiments that are explicitly registered, with
parameters inside declared ranges.
"""
from __future__ import annotations

import pytest

from lab.research.validator import (
    get_allowed_experiment_names,
    validate_experiment_spec,
)


# ─────────────────────────────────────────────────────────────────────────────
# Valid specs
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_sma_spec_passes():
    result = validate_experiment_spec(
        "sma_crossover_followthrough",
        {"short_window": 5, "long_window": 20, "horizon_periods": 15},
    )
    assert result.is_valid
    assert result.errors == []


def test_float_params_are_coerced_to_int():
    """Models sometimes output 5.0 instead of 5 — should still be accepted."""
    result = validate_experiment_spec(
        "sma_crossover_followthrough",
        {"short_window": 5.0, "long_window": 20.0, "horizon_periods": 15.0},
    )
    assert result.is_valid


# ─────────────────────────────────────────────────────────────────────────────
# Invalid experiment types
# ─────────────────────────────────────────────────────────────────────────────

def test_unknown_experiment_type_fails():
    result = validate_experiment_spec("made_up_by_agent", {"foo": 1})
    assert not result.is_valid
    assert any("Unknown experiment type" in e for e in result.errors)


def test_allowed_experiments_list_is_non_empty():
    names = get_allowed_experiment_names()
    assert len(names) > 0
    assert "sma_crossover_followthrough" in names


# ─────────────────────────────────────────────────────────────────────────────
# Missing parameters
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_short_window_fails():
    result = validate_experiment_spec(
        "sma_crossover_followthrough",
        {"long_window": 20, "horizon_periods": 15},
    )
    assert not result.is_valid
    assert any("short_window" in e for e in result.errors)


def test_missing_all_params_reports_all_missing():
    result = validate_experiment_spec("sma_crossover_followthrough", {})
    assert not result.is_valid
    assert len(result.errors) == 3  # all three parameters missing


# ─────────────────────────────────────────────────────────────────────────────
# Out-of-range parameters
# ─────────────────────────────────────────────────────────────────────────────

def test_short_window_below_minimum_fails():
    result = validate_experiment_spec(
        "sma_crossover_followthrough",
        {"short_window": 1, "long_window": 20, "horizon_periods": 15},
    )
    assert not result.is_valid
    assert any("short_window" in e for e in result.errors)


def test_long_window_above_maximum_fails():
    result = validate_experiment_spec(
        "sma_crossover_followthrough",
        {"short_window": 5, "long_window": 999, "horizon_periods": 15},
    )
    assert not result.is_valid
    assert any("long_window" in e for e in result.errors)


def test_horizon_above_maximum_fails():
    result = validate_experiment_spec(
        "sma_crossover_followthrough",
        {"short_window": 5, "long_window": 20, "horizon_periods": 500},
    )
    assert not result.is_valid
    assert any("horizon_periods" in e for e in result.errors)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-parameter rules
# ─────────────────────────────────────────────────────────────────────────────

def test_long_window_equal_to_short_window_fails():
    result = validate_experiment_spec(
        "sma_crossover_followthrough",
        {"short_window": 10, "long_window": 10, "horizon_periods": 5},
    )
    assert not result.is_valid
    assert any("long_window" in e for e in result.errors)


def test_long_window_less_than_short_window_fails():
    result = validate_experiment_spec(
        "sma_crossover_followthrough",
        {"short_window": 15, "long_window": 10, "horizon_periods": 5},
    )
    assert not result.is_valid


# ─────────────────────────────────────────────────────────────────────────────
# Non-integer parameters
# ─────────────────────────────────────────────────────────────────────────────

def test_string_param_fails_with_clear_error():
    result = validate_experiment_spec(
        "sma_crossover_followthrough",
        {"short_window": "five", "long_window": 20, "horizon_periods": 15},
    )
    assert not result.is_valid
    assert any("must be an integer" in e for e in result.errors)
