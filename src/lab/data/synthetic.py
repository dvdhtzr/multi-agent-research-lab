"""
Synthetic OHLCV data generator.

Produces reproducible, SPY-like 1-minute candle data suitable for testing
the research engine without requiring private market data.

The generator uses a seeded random walk with mild mean reversion so that
SMA crossovers occur at a realistic frequency.  The data is not intended to
replicate real market microstructure — it exists solely to make automated
tests deterministic and to provide a working example for new contributors.

Run as a script to regenerate the checked-in sample:
    python -m lab.data.synthetic
"""
from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

from lab.data.loader import OHLCVCandle


def generate_synthetic_ohlcv(
    n_candles: int = 1200,
    start_price: float = 450.0,
    seed: int = 42,
    start_datetime: datetime | None = None,
) -> list[OHLCVCandle]:
    """
    Generate a reproducible list of synthetic 1-minute OHLCV candles.

    Args:
        n_candles:      Number of candles to generate.
        start_price:    Opening price of the first candle.
        seed:           Random seed — same seed always produces the same output.
        start_datetime: Timestamp of the first candle.  Defaults to a fixed
                        Monday market open.

    Returns:
        A list of OHLCVCandle in chronological order.

    Note on realism: timestamps advance by exactly one minute per candle and
    do not skip weekends or overnight gaps.  This is intentional — the
    synthetic data is only for testing the calculation logic, not for
    realistic backtesting.
    """
    rng = random.Random(seed)
    if start_datetime is None:
        start_datetime = datetime(2023, 1, 2, 9, 30, 0)  # Monday market open

    candles: list[OHLCVCandle] = []
    price = start_price

    for i in range(n_candles):
        timestamp = start_datetime + timedelta(minutes=i)

        # Random walk: 0.08% per-minute standard deviation, mild mean reversion
        # toward start_price prevents the series from drifting too far.
        drift = (start_price - price) * 0.0005          # gentle pull back to start
        shock = rng.gauss(0, price * 0.0008)             # random component
        price = max(price + drift + shock, 1.0)          # floor at $1

        # Intra-candle spread: open is near the previous close, high/low are
        # symmetric around the OHLC midpoint.
        spread = price * rng.uniform(0.00005, 0.00030)
        open_ = price + rng.gauss(0, spread * 0.5)
        open_ = max(open_, 0.01)

        # High and low must bracket both open and close.
        candle_high = max(open_, price) + abs(rng.gauss(0, spread))
        candle_low  = min(open_, price) - abs(rng.gauss(0, spread))
        candle_low  = max(candle_low, 0.01)

        volume = int(rng.uniform(5_000, 120_000))

        candles.append(OHLCVCandle(
            timestamp=timestamp,
            open=round(open_, 2),
            high=round(candle_high, 2),
            low=round(candle_low, 2),
            close=round(price, 2),
            volume=float(volume),
        ))

    return candles


def write_synthetic_csv(candles: list[OHLCVCandle], path: Path) -> None:
    """Write a list of OHLCVCandle objects to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            writer.writerow([
                c.timestamp.isoformat(sep=" "),
                c.open, c.high, c.low, c.close, int(c.volume),
            ])


# ─────────────────────────────────────────────────────────────────────────────
# Script entry point: python -m lab.data.synthetic
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    output_path = Path("data/synthetic/spy_sample.csv")
    candles = generate_synthetic_ohlcv()
    write_synthetic_csv(candles, output_path)
    print(f"Generated {len(candles)} candles → {output_path}")
