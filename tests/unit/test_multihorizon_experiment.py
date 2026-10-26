"""
Unit tests for the sma_crossover_multihorizon experiment (Milestone 4).

These tests exercise the deterministic engine only — no LLM calls.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from lab.data.loader import OHLCVCandle
from lab.research.experiments import (
    MultiHorizonResult,
    HorizonResult,
    run_sma_crossover_multihorizon,
    run_experiment,
)
from lab.research.validator import validate_experiment_spec


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flat_candles(n: int, price: float = 100.0) -> list[OHLCVCandle]:
    base = datetime(2023, 1, 2, 9, 30)
    return [
        OHLCVCandle(
            timestamp=base + timedelta(minutes=i),
            open=price, high=price + 0.1, low=price - 0.1, close=price,
            volume=10_000.0,
        )
        for i in range(n)
    ]


def _alternating_candles(n: int) -> list[OHLCVCandle]:
    """Candles whose price alternates: 100, 101, 100, 101, ..."""
    base = datetime(2023, 1, 2, 9, 30)
    return [
        OHLCVCandle(
            timestamp=base + timedelta(minutes=i),
            open=100.0, high=101.0, low=99.0,
            close=100.0 + (i % 2),
            volume=10_000.0,
        )
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Structural tests
# ─────────────────────────────────────────────────────────────────────────────

def test_returns_one_horizon_per_requested():
    candles = _flat_candles(500)
    result = run_sma_crossover_multihorizon(
        candles, short_window=5, long_window=20, horizon_list=[5, 15, 30, 60]
    )
    assert isinstance(result, MultiHorizonResult)
    assert len(result.horizons) == 4
    assert [h.horizon_periods for h in result.horizons] == [5, 15, 30, 60]


def test_horizons_are_sorted():
    candles = _flat_candles(500)
    result = run_sma_crossover_multihorizon(
        candles, short_window=5, long_window=20, horizon_list=[60, 5, 30, 15]
    )
    periods = [h.horizon_periods for h in result.horizons]
    assert periods == sorted(periods)


def test_experiment_type_is_correct():
    candles = _flat_candles(500)
    result = run_sma_crossover_multihorizon(
        candles, short_window=5, long_window=20, horizon_list=[5]
    )
    assert result.experiment_type == "sma_crossover_multihorizon"


def test_to_dict_roundtrip():
    candles = _flat_candles(500)
    result = run_sma_crossover_multihorizon(
        candles, short_window=5, long_window=20, horizon_list=[5, 30]
    )
    d = result.to_dict()
    assert d["experiment_type"] == "sma_crossover_multihorizon"
    assert len(d["horizons"]) == 2
    assert "p_value" in d["horizons"][0]
    assert "win_rate" in d["horizons"][0]


# ─────────────────────────────────────────────────────────────────────────────
# Parameter validation
# ─────────────────────────────────────────────────────────────────────────────

def test_raises_on_empty_horizon_list():
    candles = _flat_candles(500)
    with pytest.raises(ValueError, match="horizon_list"):
        run_sma_crossover_multihorizon(
            candles, short_window=5, long_window=20, horizon_list=[]
        )


def test_raises_when_long_window_not_greater_than_short():
    candles = _flat_candles(500)
    with pytest.raises(ValueError):
        run_sma_crossover_multihorizon(
            candles, short_window=20, long_window=10, horizon_list=[5]
        )


def test_raises_when_insufficient_candles():
    candles = _flat_candles(10)
    with pytest.raises(ValueError):
        run_sma_crossover_multihorizon(
            candles, short_window=5, long_window=20, horizon_list=[5]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_flat_candles_produce_no_crossovers():
    """Flat price → no golden crosses → zero events for all horizons."""
    candles = _flat_candles(500)
    result = run_sma_crossover_multihorizon(
        candles, short_window=5, long_window=20, horizon_list=[5, 30]
    )
    for h in result.horizons:
        assert h.event_count == 0
        assert h.win_rate == 0.0


def test_each_horizon_is_independent():
    """
    Longer horizons may exclude events near the end of the series (no future candles),
    so event counts can decrease monotonically with horizon length.
    The key property is that all horizons are evaluated on the same candle series.
    """
    candles = _alternating_candles(500)
    result = run_sma_crossover_multihorizon(
        candles, short_window=3, long_window=10, horizon_list=[1, 5, 20]
    )
    counts = [h.event_count for h in result.horizons]
    # Shorter horizons have >= events than longer ones (events near end are discarded)
    assert counts[0] >= counts[1] >= counts[2]
    # All horizons have a non-negative count
    assert all(c >= 0 for c in counts)


def test_best_horizon_returns_lowest_pvalue():
    candles = _flat_candles(500)
    result = run_sma_crossover_multihorizon(
        candles, short_window=5, long_window=20, horizon_list=[5, 30, 60]
    )
    best = result.best_horizon()
    if best is not None:
        for h in result.horizons:
            if h.minimum_events_met:
                assert best.p_value <= h.p_value


def test_best_horizon_returns_none_when_no_events():
    candles = _flat_candles(500)
    result = run_sma_crossover_multihorizon(
        candles, short_window=5, long_window=20, horizon_list=[5]
    )
    assert result.best_horizon() is None


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def test_run_experiment_dispatcher_multihorizon():
    candles = _flat_candles(500)
    result = run_experiment(
        candles,
        experiment_type="sma_crossover_multihorizon",
        parameters={
            "short_window": 5,
            "long_window": 20,
            "horizon_list": [5, 15],
        },
    )
    assert "result" in result
    assert result["result"]["experiment_type"] == "sma_crossover_multihorizon"
    assert len(result["result"]["horizons"]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Validator integration
# ─────────────────────────────────────────────────────────────────────────────

def test_validator_accepts_multihorizon_params():
    vr = validate_experiment_spec(
        "sma_crossover_multihorizon",
        {"short_window": 5, "long_window": 20},
    )
    assert vr.is_valid


def test_validator_rejects_missing_long_window():
    vr = validate_experiment_spec(
        "sma_crossover_multihorizon",
        {"short_window": 5},
    )
    assert not vr.is_valid
    assert any("long_window" in e for e in vr.errors)


def test_validator_rejects_out_of_range_short_window():
    vr = validate_experiment_spec(
        "sma_crossover_multihorizon",
        {"short_window": 50, "long_window": 80},
    )
    assert not vr.is_valid
