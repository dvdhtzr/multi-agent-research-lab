"""
Baseline evaluator for Milestone 5 comparative evaluation.

Performs a random parameter-space search with no LLM involvement.
Samples (short_window, long_window, horizon_periods) combinations
uniformly at random within the valid parameter ranges and runs the
deterministic sma_crossover_followthrough experiment for each.

This is the zero-cost baseline against which the LLM-assisted pipelines
are measured.  Because no inference is involved, it is perfectly reproducible
given the same seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from lab.data.loader import OHLCVCandle
from lab.research.experiments import run_sma_crossover_followthrough


# Parameter ranges — must match validator.ALLOWED_EXPERIMENTS["sma_crossover_followthrough"]
_SHORT_MIN, _SHORT_MAX = 2, 30
_LONG_MIN, _LONG_MAX = 5, 100
_HORIZON_MIN, _HORIZON_MAX = 1, 120


@dataclass
class BaselineRunResult:
    """Result of one parameter draw."""
    short_window: int
    long_window: int
    horizon_periods: int
    event_count: int
    positive_count: int
    win_rate: float
    p_value: float
    statistically_significant: bool
    minimum_events_met: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "horizon_periods": self.horizon_periods,
            "event_count": self.event_count,
            "positive_count": self.positive_count,
            "win_rate": self.win_rate,
            "p_value": self.p_value,
            "statistically_significant": self.statistically_significant,
            "minimum_events_met": self.minimum_events_met,
        }


@dataclass
class BaselineResult:
    """Aggregate result of the baseline parameter search."""
    n_samples: int
    significant_count: int
    best_p_value: float      # 1.0 if nothing significant was found
    best_params: dict | None # params of the run with best (lowest) p-value
    all_results: list[BaselineRunResult] = field(default_factory=list)
    calls_used: int = 0      # always 0 — no LLM

    @property
    def significant_rate(self) -> float:
        if self.n_samples == 0:
            return 0.0
        return self.significant_count / self.n_samples

    def summary_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "significant_count": self.significant_count,
            "significant_rate": round(self.significant_rate, 4),
            "best_p_value": round(self.best_p_value, 6),
            "best_params": self.best_params,
            "calls_used": self.calls_used,
        }


class BaselineEvaluator:
    """
    Random parameter-space baseline.

    Draws `n_samples` parameter combinations from the valid range,
    runs the deterministic experiment for each, and returns the aggregate.

    Args:
        candles:    Full OHLCV candle series.
        seed:       Random seed for reproducibility.
    """

    def __init__(self, candles: list[OHLCVCandle], seed: int = 42) -> None:
        self._candles = candles
        self._rng = random.Random(seed)

    def run(self, n_samples: int = 50) -> BaselineResult:
        """
        Sample `n_samples` random (short, long, horizon) combinations
        and run the deterministic experiment for each.

        Returns:
            BaselineResult with per-run detail and aggregate statistics.
        """
        results: list[BaselineRunResult] = []

        for _ in range(n_samples):
            short = self._rng.randint(_SHORT_MIN, _SHORT_MAX)
            # long must be strictly > short
            long_min = max(_LONG_MIN, short + 1)
            if long_min > _LONG_MAX:
                continue
            long = self._rng.randint(long_min, _LONG_MAX)
            horizon = self._rng.randint(_HORIZON_MIN, _HORIZON_MAX)

            try:
                exp = run_sma_crossover_followthrough(
                    candles=self._candles,
                    short_window=short,
                    long_window=long,
                    horizon_periods=horizon,
                )
            except ValueError:
                continue

            results.append(BaselineRunResult(
                short_window=short,
                long_window=long,
                horizon_periods=horizon,
                event_count=exp.event_count,
                positive_count=exp.positive_count,
                win_rate=exp.win_rate,
                p_value=exp.p_value,
                statistically_significant=exp.statistically_significant,
                minimum_events_met=exp.minimum_events_met,
            ))

        significant = [r for r in results if r.statistically_significant and r.minimum_events_met]
        significant_count = len(significant)

        if results:
            best = min(
                (r for r in results if r.minimum_events_met),
                key=lambda r: r.p_value,
                default=results[0],
            )
            best_p = best.p_value
            best_params = {
                "short_window": best.short_window,
                "long_window": best.long_window,
                "horizon_periods": best.horizon_periods,
            }
        else:
            best_p = 1.0
            best_params = None

        return BaselineResult(
            n_samples=len(results),
            significant_count=significant_count,
            best_p_value=best_p,
            best_params=best_params,
            all_results=results,
            calls_used=0,
        )
