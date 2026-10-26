"""
Unit tests for the BaselineEvaluator (evaluation/baseline.py).

These are pure unit tests — no LLM, no I/O.  The baseline evaluator is
entirely deterministic given a seed and candle series.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from lab.data.loader import OHLCVCandle
from lab.evaluation.baseline import BaselineEvaluator, BaselineResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flat_candles(n: int = 500, price: float = 100.0) -> list[OHLCVCandle]:
    base = datetime(2023, 1, 2, 9, 30)
    return [
        OHLCVCandle(
            timestamp=base + timedelta(minutes=i),
            open=price, high=price + 0.1, low=price - 0.1, close=price,
            volume=10_000.0,
        )
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: structural
# ─────────────────────────────────────────────────────────────────────────────

def test_baseline_returns_correct_type():
    candles = _flat_candles()
    evaluator = BaselineEvaluator(candles, seed=42)
    result = evaluator.run(n_samples=10)
    assert isinstance(result, BaselineResult)


def test_baseline_runs_requested_samples():
    candles = _flat_candles()
    result = BaselineEvaluator(candles, seed=42).run(n_samples=20)
    assert result.n_samples == 20


def test_baseline_calls_used_is_zero():
    candles = _flat_candles()
    result = BaselineEvaluator(candles, seed=42).run(n_samples=10)
    assert result.calls_used == 0


def test_baseline_all_results_count_matches_n_samples():
    candles = _flat_candles()
    result = BaselineEvaluator(candles, seed=42).run(n_samples=15)
    assert len(result.all_results) == result.n_samples


# ─────────────────────────────────────────────────────────────────────────────
# Tests: reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def test_same_seed_produces_same_results():
    candles = _flat_candles()
    r1 = BaselineEvaluator(candles, seed=7).run(n_samples=20)
    r2 = BaselineEvaluator(candles, seed=7).run(n_samples=20)
    assert r1.best_p_value == r2.best_p_value
    assert r1.significant_count == r2.significant_count
    pairs = list(zip(r1.all_results, r2.all_results))
    assert all(a.short_window == b.short_window and a.long_window == b.long_window
               for a, b in pairs)


def test_different_seeds_produce_different_params():
    candles = _flat_candles()
    r1 = BaselineEvaluator(candles, seed=1).run(n_samples=20)
    r2 = BaselineEvaluator(candles, seed=99).run(n_samples=20)
    params1 = [(r.short_window, r.long_window) for r in r1.all_results]
    params2 = [(r.short_window, r.long_window) for r in r2.all_results]
    assert params1 != params2


# ─────────────────────────────────────────────────────────────────────────────
# Tests: parameter validity
# ─────────────────────────────────────────────────────────────────────────────

def test_all_samples_have_valid_windows():
    candles = _flat_candles(1000)
    result = BaselineEvaluator(candles, seed=42).run(n_samples=50)
    for r in result.all_results:
        assert r.short_window >= 2
        assert r.short_window <= 30
        assert r.long_window >= 5
        assert r.long_window <= 100
        assert r.long_window > r.short_window
        assert r.horizon_periods >= 1
        assert r.horizon_periods <= 120


def test_significant_count_does_not_exceed_n_samples():
    candles = _flat_candles()
    result = BaselineEvaluator(candles, seed=42).run(n_samples=30)
    assert result.significant_count <= result.n_samples


def test_best_p_value_is_1_on_flat_candles():
    """Flat price → no crossovers → p_value=1.0 for all samples → best=1.0."""
    candles = _flat_candles()
    result = BaselineEvaluator(candles, seed=42).run(n_samples=30)
    assert result.best_p_value == 1.0
    assert result.significant_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: summary_dict
# ─────────────────────────────────────────────────────────────────────────────

def test_summary_dict_has_expected_keys():
    candles = _flat_candles()
    result = BaselineEvaluator(candles, seed=42).run(n_samples=10)
    d = result.summary_dict()
    assert "n_samples" in d
    assert "significant_count" in d
    assert "significant_rate" in d
    assert "best_p_value" in d
    assert "calls_used" in d


def test_significant_rate_matches_counts():
    candles = _flat_candles()
    result = BaselineEvaluator(candles, seed=42).run(n_samples=20)
    expected_rate = result.significant_count / result.n_samples
    assert abs(result.significant_rate - expected_rate) < 1e-9
