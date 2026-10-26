"""
Deterministic research experiments.

ARCHITECTURAL RULE:  These functions are the sole source of numerical truth.
Agents may propose experiment parameters and interpret results, but they may
never produce, modify, or override the numbers calculated here.

Every function:
  • Takes only the OHLCV candles that existed at or before the research
    timestamp (point-in-time safety).
  • Returns a typed result that includes its own calculation provenance.
  • Is pure: same inputs always produce the same outputs.

Milestone 1 implements one experiment type: SMA Crossover Follow-Through Rate.

Future experiment types would be added as new functions in this module and
registered in validator.py's ALLOWED_EXPERIMENTS set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from scipy.stats import binomtest  # type: ignore[import-untyped]

from lab.data.loader import OHLCVCandle


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SMAFollowthroughResult:
    """
    The deterministic output of the SMA crossover follow-through experiment.

    All numeric fields are calculated by this module; no agent may change them.
    The calculation_method field makes the result self-documenting — anyone
    reading the database row can reproduce it.
    """
    experiment_type: str
    short_window: int
    long_window: int
    horizon_periods: int

    # Core findings
    event_count: int           # number of crossover events found
    positive_count: int        # events followed by a positive return
    win_rate: float            # positive_count / event_count (0.0 if event_count == 0)
    p_value: float             # binomial test p-value against null hypothesis p=0.5
    statistically_significant: bool  # True if p_value < 0.05

    # Data sufficiency flags
    minimum_events_met: bool   # True if event_count >= MIN_EVENTS_REQUIRED
    candles_used: int

    # Provenance — what formula produced this result
    calculation_method: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for JSON serialisation."""
        return {
            "experiment_type": self.experiment_type,
            "short_window": self.short_window,
            "long_window": self.long_window,
            "horizon_periods": self.horizon_periods,
            "event_count": self.event_count,
            "positive_count": self.positive_count,
            "win_rate": round(self.win_rate, 4),
            "p_value": round(self.p_value, 6),
            "statistically_significant": self.statistically_significant,
            "minimum_events_met": self.minimum_events_met,
            "candles_used": self.candles_used,
            "calculation_method": self.calculation_method,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MIN_EVENTS_REQUIRED = 10   # below this, the binomial test is unreliable
SIGNIFICANCE_LEVEL = 0.05  # standard α threshold


# ─────────────────────────────────────────────────────────────────────────────
# SMA Crossover Follow-Through Rate
# ─────────────────────────────────────────────────────────────────────────────

def run_sma_crossover_followthrough(
    candles: list[OHLCVCandle],
    short_window: int,
    long_window: int,
    horizon_periods: int,
) -> SMAFollowthroughResult:
    """
    Calculate the historical follow-through rate after SMA golden crosses.

    A "golden cross" occurs when the short-period SMA crosses *above* the
    long-period SMA (the short was at-or-below the long on the previous bar).

    For each such event, we check whether the closing price after
    `horizon_periods` bars is higher than the closing price at the crossover.

    We then test whether the observed win rate is significantly greater than
    50% (the null hypothesis of random chance) using a one-sided binomial test.

    Args:
        candles:         Chronologically ordered OHLCV candles.
        short_window:    Period for the fast/short SMA (e.g. 5).
        long_window:     Period for the slow/long SMA (e.g. 20).
        horizon_periods: Bars to look forward after a crossover event.

    Returns:
        SMAFollowthroughResult with all findings and provenance metadata.

    Raises:
        ValueError: if parameters are out of range or candles are insufficient.

    Note on SMA calculation: we use a simple (unweighted) rolling mean over
    closing prices.  No pandas dependency — the loop is O(n × max_window) but
    is fine for the dataset sizes in this project.
    """
    _validate_params(candles, short_window, long_window, horizon_periods)

    closes = [c.close for c in candles]
    n = len(closes)

    # --- Compute SMAs ---
    # short_smas[i] and long_smas[i] are None until enough bars have been seen.
    short_smas: list[float | None] = [None] * n
    long_smas: list[float | None] = [None] * n

    # Running sums make this O(n) instead of O(n × window).
    short_sum = 0.0
    long_sum = 0.0
    for i in range(n):
        short_sum += closes[i]
        long_sum += closes[i]
        if i >= short_window:
            short_sum -= closes[i - short_window]
        if i >= long_window:
            long_sum -= closes[i - long_window]

        if i >= short_window - 1:
            short_smas[i] = short_sum / short_window
        if i >= long_window - 1:
            long_smas[i] = long_sum / long_window

    # --- Detect golden crosses ---
    # A cross at index i: short was at-or-below long at i-1, now above at i.
    crossover_indices: list[int] = []
    for i in range(1, n):
        prev_short = short_smas[i - 1]
        prev_long = long_smas[i - 1]
        curr_short = short_smas[i]
        curr_long = long_smas[i]

        if None in (prev_short, prev_long, curr_short, curr_long):
            continue

        was_below_or_equal = prev_short <= prev_long  # type: ignore[operator]
        is_now_above = curr_short > curr_long          # type: ignore[operator]
        if was_below_or_equal and is_now_above:
            crossover_indices.append(i)

    # --- Evaluate follow-through ---
    # Only include events where we have a complete horizon (no lookahead leakage
    # beyond the end of the dataset; those events are simply excluded).
    positive_count = 0
    valid_events = 0
    for idx in crossover_indices:
        future_idx = idx + horizon_periods
        if future_idx >= n:
            continue  # no complete horizon — exclude rather than fabricate
        valid_events += 1
        if closes[future_idx] > closes[idx]:
            positive_count += 1

    win_rate = positive_count / valid_events if valid_events > 0 else 0.0

    # --- Binomial significance test ---
    # Null hypothesis: P(positive return) = 0.5 (random coin flip).
    # Alternative: P > 0.5 (the crossover has predictive power).
    # scipy.stats.binomtest(k, n, p, alternative) returns a BinomTestResult.
    if valid_events >= MIN_EVENTS_REQUIRED:
        binom_result = binomtest(positive_count, n=valid_events, p=0.5, alternative="greater")
        p_value = float(binom_result.pvalue)
    else:
        # Not enough data for a meaningful test.
        p_value = 1.0  # conservative — treat as non-significant

    statistically_significant = p_value < SIGNIFICANCE_LEVEL
    minimum_events_met = valid_events >= MIN_EVENTS_REQUIRED

    method_str = (
        f"scipy.stats.binomtest(k={positive_count}, n={valid_events}, "
        f"p=0.5, alternative='greater') | "
        f"alpha={SIGNIFICANCE_LEVEL} | "
        f"min_events={MIN_EVENTS_REQUIRED}"
    )

    return SMAFollowthroughResult(
        experiment_type="sma_crossover_followthrough",
        short_window=short_window,
        long_window=long_window,
        horizon_periods=horizon_periods,
        event_count=valid_events,
        positive_count=positive_count,
        win_rate=win_rate,
        p_value=p_value,
        statistically_significant=statistically_significant,
        minimum_events_met=minimum_events_met,
        candles_used=n,
        calculation_method=method_str,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher — called by the tool server
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    candles: list[OHLCVCandle],
    experiment_type: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """
    Dispatch an experiment request to the correct deterministic function.

    This is the single entry point called by the HTTP tool server.  It returns
    the result as a plain dict for JSON serialisation.

    Args:
        candles:         The OHLCV candles available at research time.
        experiment_type: Must match an entry in validator.ALLOWED_EXPERIMENTS.
        parameters:      Experiment-specific parameters (already validated).

    Returns:
        A dict containing both the experiment result and metadata.

    Raises:
        ValueError: if the experiment_type is not recognised.  The validator
                    should prevent this, but we check defensively here too.
    """
    if experiment_type == "sma_crossover_followthrough":
        result = run_sma_crossover_followthrough(
            candles=candles,
            short_window=int(parameters["short_window"]),
            long_window=int(parameters["long_window"]),
            horizon_periods=int(parameters["horizon_periods"]),
        )
        return {
            "result": result.to_dict(),
            "candles_used": result.candles_used,
            "method": result.calculation_method,
        }

    raise ValueError(
        f"Unknown experiment type: '{experiment_type}'. "
        f"This type has not been implemented."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_params(
    candles: list[OHLCVCandle],
    short_window: int,
    long_window: int,
    horizon_periods: int,
) -> None:
    if short_window < 2:
        raise ValueError(f"short_window must be >= 2, got {short_window}")
    if long_window <= short_window:
        raise ValueError(
            f"long_window ({long_window}) must be > short_window ({short_window})"
        )
    if horizon_periods < 1:
        raise ValueError(f"horizon_periods must be >= 1, got {horizon_periods}")
    min_needed = long_window + horizon_periods + 1
    if len(candles) < min_needed:
        raise ValueError(
            f"Need at least {min_needed} candles for these parameters "
            f"(long_window={long_window} + horizon={horizon_periods} + 1); "
            f"only {len(candles)} available."
        )
