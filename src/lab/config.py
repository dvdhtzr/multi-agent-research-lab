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

    # --- Hard limits ---
    # WARNING: These limits are designed for locally-hosted, non-metered Ollama.
    # Never use progress-based budget extensions with paid API providers without
    # independently enforced spending ceilings that no agent decision can override.
    #
    # Milestone 2 uses 8 Ollama calls per run.
    # Default of 12 leaves headroom for slow models that may retry internally.
    max_ollama_calls: int = Field(
        default=12,
        description="Maximum Ollama model invocations per research run",
    )
    max_wall_clock_seconds: int = Field(
        default=300,
        description="Maximum wall-clock seconds allowed for a research run",
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
