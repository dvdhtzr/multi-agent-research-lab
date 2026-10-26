"""
Application configuration using Pydantic BaseSettings.

Pydantic BaseSettings reads field values from environment variables, in
priority order:
  1. Environment variable (e.g. LAB_DEFAULT_MODEL=qwen3:30b)
  2. .env file in the working directory
  3. The default value declared on the field

C# parallel: a combination of IOptions<T> (typed configuration binding),
IConfiguration (source chaining), and appsettings.json defaults — all in
one class, with automatic type coercion and validation.

Usage:
    cfg = load_config()           # reads .env + env vars
    cfg = load_config(".env.dev") # reads a specific env file

Values can be overridden at the shell without touching any file:
    LAB_DEFAULT_MODEL=qwen3:30b python -m lab research
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LabConfig(BaseSettings):
    """
    All runtime configuration for the lab.

    Each field has a sensible default so the project works out-of-the-box
    after a fresh clone with no .env file at all.
    """

    model_config = SettingsConfigDict(
        env_prefix="LAB_",          # LAB_DEFAULT_MODEL maps to default_model
        env_file=".env",            # read .env if it exists
        env_file_encoding="utf-8",
        extra="ignore",             # silently ignore unknown variables in .env
    )

    # --- Ollama ---
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the locally running Ollama instance",
    )
    default_model: str = Field(
        default="gemma4:latest",
        description="Ollama model used by all agents unless overridden",
    )

    # --- Data and persistence ---
    data_path: Path = Field(default=Path("data/synthetic/spy_sample.csv"))
    db_path: Path = Field(default=Path("research.db"))

    # --- Hard limits (per single run) ---
    # WARNING: These limits are designed for locally-hosted, non-metered Ollama.
    # Never use progress-based budget extensions with paid API providers without
    # independently enforced spending ceilings that no agent decision can override.
    #
    # Milestone 2 uses 8 Ollama calls per run (7 agent calls + headroom for retries).
    # Default of 12 leaves headroom for slow models that may retry internally.
    max_ollama_calls: int = Field(
        default=12,
        description="Maximum Ollama model invocations per single research run (including retries)",
    )
    max_wall_clock_seconds: int = Field(
        default=300,
        description="Maximum wall-clock seconds allowed for a single research run",
    )

    # --- Campaign limits (Milestone 3) ---
    # A campaign runs multiple pipeline iterations autonomously.
    # max_campaign_runs caps how many runs the campaign may execute.
    # max_campaign_calls caps total LLM calls across ALL runs, including retries.
    max_campaign_runs: int = Field(
        default=5,
        description="Maximum pipeline runs in one autonomous campaign",
    )
    max_campaign_calls: int = Field(
        default=60,
        description="Maximum total LLM calls (including retries) across an entire campaign",
    )

    # --- Replay settings (Milestone 4) ---
    # Walk-forward replay slides a window of REPLAY_WINDOW_SIZE candles through
    # the full dataset, stepping forward REPLAY_STEP_SIZE candles each time.
    # At each step a compact 4-agent pipeline tests REPLAY_HORIZONS simultaneously.
    replay_window_size: int = Field(
        default=300,
        description="Number of candles per walk-forward replay window",
    )
    replay_step_size: int = Field(
        default=60,
        description="Candles to advance between consecutive replay windows",
    )
    replay_horizons: list[int] = Field(
        default=[5, 15, 30, 60],
        description="Prediction horizons (bars) tested simultaneously in each replay window",
    )
    max_replay_calls: int = Field(
        default=200,
        description="Maximum total LLM calls (including retries) across an entire replay run",
    )

    # --- Strategy ---
    # A plain-English description of the market pattern or hypothesis to investigate.
    # When set, it is prepended to the Researcher's prompt on every run so the
    # agent proposes SMA parameters that are grounded in the stated strategy.
    # Set via LAB_STRATEGY or LAB_STRATEGY_FILE; CLI --strategy / --strategy-file
    # take precedence.
    strategy: Optional[str] = Field(
        default=None,
        description="Strategy description to guide the Researcher's parameter choices",
    )
    strategy_file: Optional[Path] = Field(
        default=None,
        description="Path to a text file containing the strategy description",
    )

    # --- Custom researcher prompts ---
    # Point these at plain-text files to override the built-in researcher
    # instructions without touching the source code.  Leave unset to use
    # the defaults in lab/agents/prompts.py.
    #
    # The main pipeline and campaign use researcher_system_file /
    # researcher_prompt_file.  The replay pipeline uses the replay_* variants.
    researcher_system_file: Optional[Path] = Field(
        default=None,
        description="Path to a text file containing a custom RESEARCHER_SYSTEM prompt",
    )
    researcher_prompt_file: Optional[Path] = Field(
        default=None,
        description="Path to a text file containing a custom RESEARCHER_PROMPT",
    )
    replay_researcher_system_file: Optional[Path] = Field(
        default=None,
        description="Path to a text file containing a custom REPLAY_RESEARCHER_SYSTEM prompt",
    )
    replay_researcher_prompt_file: Optional[Path] = Field(
        default=None,
        description="Path to a text file containing a custom REPLAY_RESEARCHER_PROMPT",
    )

    # --- Comparison settings (Milestone 5) ---
    # compare_n_samples: number of parameter draws for baseline + single-agent.
    # compare_llm_budget: total LLM calls for the single-agent and multi-agent arms.
    compare_n_samples: int = Field(
        default=50,
        description="Parameter samples for baseline and single-agent comparison arms",
    )
    compare_llm_budget: int = Field(
        default=60,
        description="Total LLM call budget for single-agent and multi-agent comparison arms",
    )


def load_config(env_file: Optional[str] = None) -> LabConfig:
    """
    Load and return application configuration.

    Args:
        env_file: Optional path to an alternative .env file.
                  Defaults to .env in the working directory.
    """
    if env_file is not None:
        return LabConfig(_env_file=env_file)
    return LabConfig()
