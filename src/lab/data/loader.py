"""
OHLCV CSV loader and data validation.

The CSV format expected is:
    timestamp,open,high,low,close,volume
    2023-01-02 09:30:00,450.12,450.45,449.98,450.23,45231

Timestamps must be parseable by datetime.fromisoformat().
Rows with missing or malformed fields are rejected with a descriptive error.

C# parallel: similar to a CsvHelper reader with a validation step, but using
only the standard library (csv module + dataclasses).  No third-party CSV
library is needed for this format.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OHLCVCandle:
    """
    One OHLCV bar (one minute of market data).

    frozen=True makes instances immutable and hashable — equivalent to a
    C# record struct.  We never modify candles after loading.
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        """Validate field relationships that the type system cannot express."""
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low}) at {self.timestamp}")
        if self.high < self.open or self.high < self.close:
            raise ValueError(f"high is not the period maximum at {self.timestamp}")
        if self.low > self.open or self.low > self.close:
            raise ValueError(f"low is not the period minimum at {self.timestamp}")
        if self.volume < 0:
            raise ValueError(f"volume cannot be negative at {self.timestamp}")


class LoadResult(NamedTuple):
    """
    The outcome of loading a CSV file.

    Using NamedTuple here (instead of a dataclass) because this is a simple
    value returned from a function — think of it as a C# ValueTuple with names.
    """
    candles: list[OHLCVCandle]
    skipped_rows: int
    errors: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def load_csv(path: Path, strict: bool = False) -> LoadResult:
    """
    Load a CSV file of 1-minute OHLCV candles.

    Args:
        path:   Path to the CSV file.
        strict: If True, raise ValueError on the first bad row instead of
                skipping it.  Useful for validating production data.

    Returns:
        LoadResult with the parsed candles, a count of skipped rows, and
        a list of error strings describing any problems found.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError:        if the header row is missing required columns,
                           or if strict=True and any row is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    candles: list[OHLCVCandle] = []
    errors: list[str] = []
    skipped = 0

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate that all required columns are present in the header.
        # csv.DictReader exposes fieldnames after the first read attempt.
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty: {path}")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV missing required columns: {sorted(missing)}. "
                f"Found: {sorted(reader.fieldnames)}"
            )

        for line_number, row in enumerate(reader, start=2):  # +2: 1 header + 1-indexed
            try:
                candle = _parse_row(row, line_number)
                candles.append(candle)
            except (ValueError, KeyError) as exc:
                msg = f"Line {line_number}: {exc}"
                if strict:
                    raise ValueError(msg) from exc
                errors.append(msg)
                skipped += 1

    # Enforce chronological order — out-of-order data causes look-ahead leakage.
    if candles:
        for i in range(1, len(candles)):
            if candles[i].timestamp <= candles[i - 1].timestamp:
                raise ValueError(
                    f"Timestamps are not strictly increasing at index {i}: "
                    f"{candles[i - 1].timestamp} -> {candles[i].timestamp}"
                )

    return LoadResult(candles=candles, skipped_rows=skipped, errors=errors)


def validate_partition_size(candles: list[OHLCVCandle], min_candles: int = 50) -> None:
    """
    Raise ValueError if the dataset is too small to run experiments reliably.

    The minimum is intentionally low for the synthetic sample; production data
    will be orders of magnitude larger.
    """
    if len(candles) < min_candles:
        raise ValueError(
            f"Dataset has only {len(candles)} candles; "
            f"at least {min_candles} are required for reliable experiments."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_row(row: dict[str, str], line_number: int) -> OHLCVCandle:
    """Parse one CSV row into an OHLCVCandle, raising ValueError on bad data."""
    try:
        timestamp = datetime.fromisoformat(row["timestamp"].strip())
    except ValueError as exc:
        raise ValueError(f"unparseable timestamp '{row['timestamp']}': {exc}") from exc

    try:
        open_ = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        volume = float(row["volume"])
    except ValueError as exc:
        raise ValueError(f"non-numeric price/volume field: {exc}") from exc

    # OHLCVCandle.__post_init__ validates the field relationships.
    return OHLCVCandle(
        timestamp=timestamp,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
