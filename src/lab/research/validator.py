"""
Experiment specification validator.

Validates agent-proposed experiment specs before they reach the deterministic
research engine.  This is the enforcement point for the trust boundary:

  Agents *request* experiments using structured specs.
  The validator decides whether a request is permissible.
  The research engine executes *only* approved requests.

An agent cannot invent a new experiment type at runtime — it can only use
the types and parameter ranges explicitly declared in ALLOWED_EXPERIMENTS.

C# parallel: similar to a FluentValidation validator or a guard-clause class
that sits between a controller and a domain service.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Allowed experiment registry
#
# Each entry describes one permitted experiment type and the constraints on its
# parameters.  Adding a new experiment type requires a human developer to edit
# this file — it cannot happen at runtime.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParameterSpec:
    """Describes the valid range for one integer experiment parameter."""
    name: str
    min_value: int
    max_value: int
    description: str


@dataclass(frozen=True)
class ExperimentSpec:
    """Describes a permitted experiment type and its parameter constraints."""
    name: str
    description: str
    parameters: list[ParameterSpec]


ALLOWED_EXPERIMENTS: dict[str, ExperimentSpec] = {
    "sma_crossover_followthrough": ExperimentSpec(
        name="sma_crossover_followthrough",
        description=(
            "Measures the proportion of SMA golden-cross events (short SMA "
            "crossing above long SMA) that are followed by positive returns "
            "after a specified number of bars.  Tests whether this "
            "rate is significantly greater than random chance (p=0.5)."
        ),
        parameters=[
            ParameterSpec("short_window",   min_value=2,  max_value=30,  description="Period of the fast SMA"),
            ParameterSpec("long_window",    min_value=5,  max_value=100, description="Period of the slow SMA"),
            ParameterSpec("horizon_periods",min_value=1,  max_value=120, description="Bars to look forward for outcome"),
        ],
    ),
    "sma_crossover_multihorizon": ExperimentSpec(
        name="sma_crossover_multihorizon",
        description=(
            "Tests the SMA golden-cross follow-through pattern at multiple "
            "prediction horizons simultaneously.  The set of horizons is fixed "
            "by configuration — agents propose only short_window and long_window.  "
            "Returns one statistical result per horizon, allowing comparison of "
            "how the signal's predictive strength varies with the look-ahead window."
        ),
        parameters=[
            ParameterSpec("short_window", min_value=2,  max_value=30,  description="Period of the fast SMA"),
            ParameterSpec("long_window",  min_value=5,  max_value=100, description="Period of the slow SMA"),
        ],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Validation logic
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of validating an experiment specification."""
    is_valid: bool
    errors: list[str]


def validate_experiment_spec(
    experiment_type: str,
    parameters: dict[str, Any],
) -> ValidationResult:
    """
    Validate an agent-proposed experiment specification.

    Args:
        experiment_type: The experiment type string (e.g. 'sma_crossover_followthrough').
        parameters:      A dict of parameter name → value from the agent.

    Returns:
        ValidationResult.  Check .is_valid before calling the research engine.
        .errors contains human-readable explanations for any failures.
    """
    errors: list[str] = []

    # --- Check experiment type is allowed ---
    if experiment_type not in ALLOWED_EXPERIMENTS:
        known = ", ".join(sorted(ALLOWED_EXPERIMENTS.keys()))
        errors.append(
            f"Unknown experiment type: '{experiment_type}'. "
            f"Allowed types: {known}."
        )
        return ValidationResult(is_valid=False, errors=errors)

    spec = ALLOWED_EXPERIMENTS[experiment_type]

    # --- Check each declared parameter ---
    for param_spec in spec.parameters:
        if param_spec.name not in parameters:
            errors.append(f"Missing required parameter: '{param_spec.name}'.")
            continue

        raw_value = parameters[param_spec.name]

        # Coerce to int (models sometimes emit floats like 5.0)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            errors.append(
                f"Parameter '{param_spec.name}' must be an integer, "
                f"got {type(raw_value).__name__!r}: {raw_value!r}."
            )
            continue

        if not (param_spec.min_value <= value <= param_spec.max_value):
            errors.append(
                f"Parameter '{param_spec.name}' = {value} is outside "
                f"the allowed range [{param_spec.min_value}, {param_spec.max_value}]."
            )

    # --- Cross-parameter rule for sma_crossover_followthrough ---
    if experiment_type == "sma_crossover_followthrough" and not errors:
        short = int(parameters.get("short_window", 0))
        long_ = int(parameters.get("long_window", 0))
        if long_ <= short:
            errors.append(
                f"long_window ({long_}) must be strictly greater than "
                f"short_window ({short})."
            )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def get_allowed_experiment_names() -> list[str]:
    """Return the list of experiment type strings agents may request."""
    return sorted(ALLOWED_EXPERIMENTS.keys())
