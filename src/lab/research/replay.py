"""
Walk-forward replay engine for historical multi-horizon research (Milestone 4).

Slides a fixed-size window of candles through the full dataset, yielding one
ReplayStep per position.  Each step contains a point-in-time slice of the
candle series — only candles available up to that moment in history.

This is the deterministic part of the replay pipeline.  No LLM calls happen
here.  The replay_runner.py uses these steps to drive the 4-agent pipeline.

C# parallel: a generator/yield-based pipeline stage, similar to IEnumerable<T>
with lazy materialisation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from lab.data.loader import OHLCVCandle


@dataclass(frozen=True)
class ReplayStep:
    """One window in the walk-forward replay.

    window_start and window_end are 0-based indices into the full candle list.
    candles contains only the slice [window_start, window_end) — agents and
    experiments see nothing outside this window (point-in-time isolation).
    """
    step_index: int
    window_start: int       # inclusive
    window_end: int         # exclusive
    candles: list[OHLCVCandle]
    timestamp_start: datetime
    timestamp_end: datetime

    @property
    def window_size(self) -> int:
        return self.window_end - self.window_start


class ReplayEngine:
    """
    Generates walk-forward windows over a candle series.

    Each call to steps() yields successive ReplayStep objects.  The last
    window may be smaller than window_size if the candle count is not an
    exact multiple of step_size — it is skipped if it would be too small
    to support the experiment (fewer than window_size candles).

    Args:
        candles:     Full chronologically-ordered OHLCV candle series.
        window_size: Number of candles in each window.
        step_size:   Candles to advance between consecutive windows.
    """

    def __init__(
        self,
        candles: list[OHLCVCandle],
        window_size: int,
        step_size: int,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")
        if len(candles) < window_size:
            raise ValueError(
                f"Not enough candles for even one window: "
                f"have {len(candles)}, need {window_size}."
            )
        self._candles = candles
        self._window_size = window_size
        self._step_size = step_size

    def total_steps(self) -> int:
        """Return the number of full windows that fit in the candle series."""
        n = len(self._candles)
        count = 0
        start = 0
        while start + self._window_size <= n:
            count += 1
            start += self._step_size
        return count

    def steps(self):
        """
        Yield ReplayStep objects for each walk-forward window.

        Yields:
            ReplayStep with point-in-time candle slice and metadata.
        """
        n = len(self._candles)
        start = 0
        step_idx = 0
        while start + self._window_size <= n:
            end = start + self._window_size
            window = self._candles[start:end]
            yield ReplayStep(
                step_index=step_idx,
                window_start=start,
                window_end=end,
                candles=window,
                timestamp_start=window[0].timestamp,
                timestamp_end=window[-1].timestamp,
            )
            start += self._step_size
            step_idx += 1
