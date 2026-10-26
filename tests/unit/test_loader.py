"""
Unit tests for the OHLCV CSV loader.

These tests exercise the loader in complete isolation — no database, no network,
no Ollama.  They use small in-memory CSV strings rather than real files.

pytest note: fixtures decorated with @pytest.fixture are dependency-injected
by pytest into test functions that name them as parameters.  C# parallel:
similar to [SetUp] methods or constructor injection in a test class, but
without the class requirement.
"""
from __future__ import annotations

import io
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from lab.data.loader import OHLCVCandle, load_csv, validate_partition_size


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _write_temp_csv(content: str) -> Path:
    """Write a CSV string to a temporary file and return its Path."""
    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    tf.write(content)
    tf.flush()
    tf.close()
    return Path(tf.name)


VALID_CSV = """\
timestamp,open,high,low,close,volume
2023-01-02 09:30:00,450.10,450.50,449.90,450.23,45000
2023-01-02 09:31:00,450.23,450.80,450.00,450.55,32000
2023-01-02 09:32:00,450.55,451.00,450.40,450.88,28000
"""


# ─────────────────────────────────────────────────────────────────────────────
# Happy-path tests
# ─────────────────────────────────────────────────────────────────────────────

def test_load_valid_csv_returns_correct_candle_count():
    path = _write_temp_csv(VALID_CSV)
    result = load_csv(path)
    assert len(result.candles) == 3
    assert result.skipped_rows == 0
    assert result.errors == []


def test_loaded_candle_fields_match_csv_values():
    path = _write_temp_csv(VALID_CSV)
    result = load_csv(path)
    first = result.candles[0]

    assert first.timestamp == datetime(2023, 1, 2, 9, 30, 0)
    assert first.open == 450.10
    assert first.high == 450.50
    assert first.low == 449.90
    assert first.close == 450.23
    assert first.volume == 45000.0


def test_candles_are_in_chronological_order():
    path = _write_temp_csv(VALID_CSV)
    result = load_csv(path)
    timestamps = [c.timestamp for c in result.candles]
    assert timestamps == sorted(timestamps)


# ─────────────────────────────────────────────────────────────────────────────
# Error-handling tests
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_csv(Path("nonexistent_file_xyz.csv"))


def test_missing_column_raises_value_error():
    bad_csv = "timestamp,open,high,low,close\n2023-01-02 09:30:00,1,2,0,1\n"
    path = _write_temp_csv(bad_csv)
    with pytest.raises(ValueError, match="missing required columns"):
        load_csv(path)


def test_non_numeric_price_is_skipped_by_default():
    bad_csv = (
        "timestamp,open,high,low,close,volume\n"
        "2023-01-02 09:30:00,NOT_A_NUMBER,450.50,449.90,450.23,45000\n"
        "2023-01-02 09:31:00,450.23,450.80,450.00,450.55,32000\n"
    )
    path = _write_temp_csv(bad_csv)
    result = load_csv(path)
    assert result.skipped_rows == 1
    assert len(result.candles) == 1


def test_non_numeric_price_raises_in_strict_mode():
    bad_csv = (
        "timestamp,open,high,low,close,volume\n"
        "2023-01-02 09:30:00,NOT_A_NUMBER,450.50,449.90,450.23,45000\n"
    )
    path = _write_temp_csv(bad_csv)
    with pytest.raises(ValueError):
        load_csv(path, strict=True)


def test_bad_timestamp_is_skipped():
    bad_csv = (
        "timestamp,open,high,low,close,volume\n"
        "NOT-A-DATE,450.10,450.50,449.90,450.23,45000\n"
        "2023-01-02 09:31:00,450.23,450.80,450.00,450.55,32000\n"
    )
    path = _write_temp_csv(bad_csv)
    result = load_csv(path)
    assert result.skipped_rows == 1
    assert len(result.candles) == 1


def test_high_less_than_low_is_skipped():
    # high=449.00 < low=450.50 — invalid OHLCV
    bad_csv = (
        "timestamp,open,high,low,close,volume\n"
        "2023-01-02 09:30:00,450.10,449.00,450.50,450.23,45000\n"
        "2023-01-02 09:31:00,450.23,450.80,450.00,450.55,32000\n"
    )
    path = _write_temp_csv(bad_csv)
    result = load_csv(path)
    assert result.skipped_rows == 1
    assert len(result.candles) == 1


def test_out_of_order_timestamps_raise_value_error():
    reversed_csv = (
        "timestamp,open,high,low,close,volume\n"
        "2023-01-02 09:32:00,450.55,451.00,450.40,450.88,28000\n"
        "2023-01-02 09:30:00,450.10,450.50,449.90,450.23,45000\n"
    )
    path = _write_temp_csv(reversed_csv)
    with pytest.raises(ValueError, match="not strictly increasing"):
        load_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# Partition validation
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_partition_size_passes_with_enough_candles():
    path = _write_temp_csv(VALID_CSV)
    result = load_csv(path)
    # Should not raise for min_candles=3
    validate_partition_size(result.candles, min_candles=3)


def test_validate_partition_size_raises_when_too_few():
    path = _write_temp_csv(VALID_CSV)
    result = load_csv(path)
    with pytest.raises(ValueError, match="at least"):
        validate_partition_size(result.candles, min_candles=100)
