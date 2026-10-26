"""
Unit tests for the walk-forward ReplayEngine (research/replay.py).

These are pure unit tests — no I/O, no Ollama, no DB.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from lab.data.loader import OHLCVCandle
from lab.research.replay import ReplayEngine, ReplayStep


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_candles(n: int) -> list[OHLCVCandle]:
    base = datetime(2023, 1, 2, 9, 30)
    price = 100.0
    return [
        OHLCVCandle(
            timestamp=base + timedelta(minutes=i),
            open=price, high=price + 0.1, low=price - 0.1, close=price,
            volume=10_000.0,
        )
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Construction validation
# ─────────────────────────────────────────────────────────────────────────────

def test_engine_raises_on_zero_window_size():
    candles = _make_candles(100)
    with pytest.raises(ValueError, match="window_size"):
        ReplayEngine(candles, window_size=0, step_size=10)


def test_engine_raises_on_zero_step_size():
    candles = _make_candles(100)
    with pytest.raises(ValueError, match="step_size"):
        ReplayEngine(candles, window_size=50, step_size=0)


def test_engine_raises_when_candles_fewer_than_window():
    candles = _make_candles(30)
    with pytest.raises(ValueError, match="Not enough candles"):
        ReplayEngine(candles, window_size=50, step_size=10)


# ─────────────────────────────────────────────────────────────────────────────
# Step generation
# ─────────────────────────────────────────────────────────────────────────────

def test_single_window_when_candles_equals_window_size():
    candles = _make_candles(100)
    engine = ReplayEngine(candles, window_size=100, step_size=50)
    steps = list(engine.steps())
    assert len(steps) == 1
    assert steps[0].step_index == 0
    assert steps[0].window_start == 0
    assert steps[0].window_end == 100
    assert len(steps[0].candles) == 100


def test_multiple_windows_non_overlapping():
    candles = _make_candles(300)
    engine = ReplayEngine(candles, window_size=100, step_size=100)
    steps = list(engine.steps())
    assert len(steps) == 3
    for i, step in enumerate(steps):
        assert step.step_index == i
        assert step.window_start == i * 100
        assert step.window_end == (i + 1) * 100
        assert len(step.candles) == 100


def test_overlapping_windows():
    candles = _make_candles(200)
    engine = ReplayEngine(candles, window_size=100, step_size=50)
    steps = list(engine.steps())
    assert len(steps) == 3   # 0:100, 50:150, 100:200
    assert steps[0].window_start == 0
    assert steps[1].window_start == 50
    assert steps[2].window_start == 100


def test_partial_last_window_is_skipped():
    candles = _make_candles(250)
    engine = ReplayEngine(candles, window_size=100, step_size=100)
    steps = list(engine.steps())
    # 0:100, 100:200 fit; 200:300 would be 200:250 (< 100) so skipped
    assert len(steps) == 2


def test_total_steps_matches_generated_count():
    candles = _make_candles(500)
    engine = ReplayEngine(candles, window_size=100, step_size=60)
    assert engine.total_steps() == len(list(engine.steps()))


# ─────────────────────────────────────────────────────────────────────────────
# Point-in-time isolation
# ─────────────────────────────────────────────────────────────────────────────

def test_candles_are_point_in_time_slice():
    candles = _make_candles(300)
    engine = ReplayEngine(candles, window_size=100, step_size=100)
    steps = list(engine.steps())
    # Window 1 must contain exactly candles[100:200]
    for idx, candle in enumerate(steps[1].candles):
        assert candle is candles[100 + idx]


def test_step_timestamps_match_window():
    candles = _make_candles(200)
    engine = ReplayEngine(candles, window_size=100, step_size=100)
    steps = list(engine.steps())
    for step in steps:
        assert step.timestamp_start == step.candles[0].timestamp
        assert step.timestamp_end == step.candles[-1].timestamp


# ─────────────────────────────────────────────────────────────────────────────
# window_size property
# ─────────────────────────────────────────────────────────────────────────────

def test_window_size_property():
    candles = _make_candles(200)
    engine = ReplayEngine(candles, window_size=100, step_size=50)
    for step in engine.steps():
        assert step.window_size == 100
