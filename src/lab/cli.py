"""
Console entry point for the SPY Research Lab.

Commands:
  research  — start a bounded autonomous research run
  history   — display stored research runs and their outcomes

The `research` command runs the full 8-agent pipeline directly in-process:
  1. Load and validate the OHLCV data.
  2. Initialise the SQLite database.
  3. Run the pipeline:
       Researcher → Devil's Advocate → Researcher Rebuttal
       → [deterministic experiment] → Detractor → Judge
       → Recommender → Summarizer
  4. Persist the complete result to SQLite.
  5. Print a human-readable summary.

No separate services are required.  Ollama must be running locally.

C# parallel: Click is the Python equivalent of System.CommandLine or Cocona.
`@click.group()` is the root command and each `@group.command()` is a sub-command.
"""
from __future__ import annotations

import json
import sys
import textwrap
import uuid
from pathlib import Path

import click

from lab.campaign.runner import CampaignResult, CampaignRunner
from lab.config import LabConfig, load_config
from lab.data.loader import load_csv, validate_partition_size
from lab.db.repository import (
    AgentArgument,
    Experiment,
    Hypothesis,
    JudgeDecision,
    Recommendation,
    Summary,
    ResearchRun,
    ReplayRun as ReplayRunRecord,
    ReplayStep as ReplayStepRecord,
    new_id,
    now_iso,
    save_agent_argument,
    save_experiment,
    save_hypothesis,
    save_judge_decision,
    save_recommendation,
    save_summary,
    save_research_run,
    update_run_status,
    get_recent_runs,
    get_recent_campaigns,
    get_recent_replay_runs,
    get_replay_steps,
    get_run_details,
    save_replay_run,
    save_replay_step,
    update_replay_run,
)
from lab.db.schema import init_db
from lab.pipeline.runner import PipelineResult, PipelineRunner, BudgetExceededError
from lab.pipeline.replay_runner import ReplayPipelineRunner, ReplayResult, ReplayStepResult
from lab.research.replay import ReplayEngine
from lab.evaluation.runner import ComparisonResult, ComparisonRunner
from lab.db.repository import (
    Comparison,
    save_comparison,
    get_recent_comparisons,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI root group
# ─────────────────────────────────────────────────────────────────────────────

@click.group()
def main() -> None:
    """SPY Research Lab — local multi-agent historical research."""


# ─────────────────────────────────────────────────────────────────────────────
# research command
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
@click.option(
    "--env-file",
    default=None,
    help="Path to a .env configuration file.  Defaults to .env in the working directory.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Print each agent's deliberation as it completes.",
)
def research(env_file: str | None, verbose: bool) -> None:
    """
    Start a bounded autonomous research run.

    Loads the configured OHLCV data, runs the 8-agent pipeline against
    Ollama, persists the result to SQLite, and prints a summary.
    """
    cfg = load_config(env_file)

    click.echo("═" * 60)
    click.echo("  SPY Research Lab — Research Run")
    click.echo("═" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    click.echo(f"\n[1/4] Loading data from {cfg.data_path} ...")
    try:
        load_result = load_csv(cfg.data_path)
    except FileNotFoundError as exc:
        _error(str(exc))
        sys.exit(1)

    if load_result.errors:
        click.echo(f"  ⚠  Skipped {load_result.skipped_rows} rows: {load_result.errors[:3]}")

    try:
        validate_partition_size(load_result.candles)
    except ValueError as exc:
        _error(str(exc))
        sys.exit(1)

    click.echo(f"  ✓  {len(load_result.candles)} candles loaded.")

    # ── 2. Initialise database ────────────────────────────────────────────────
    conn = init_db(cfg.db_path)
    run_id = str(uuid.uuid4())
    ts = now_iso()

    with conn:
        save_research_run(conn, ResearchRun(
            id=run_id,
            created_at=ts,
            data_path=str(cfg.data_path),
            model=cfg.default_model,
            status="partial",
        ))

    # ── 3. Run the pipeline ───────────────────────────────────────────────────
    click.echo(f"\n[2/4] Running 8-agent pipeline ...")
    click.echo(f"  Model: {cfg.default_model} | Run ID: {run_id[:8]}...")
    click.echo(f"  Budget: {cfg.max_ollama_calls} Ollama calls, {cfg.max_wall_clock_seconds}s wall clock")
    click.echo(f"  Ollama: {cfg.ollama_base_url}")
    click.echo("  (This may take 1–5 minutes while the models run.)\n")

    on_step = _make_verbose_callback() if verbose else None
    runner = PipelineRunner(cfg, load_result.candles, on_step=on_step)
    try:
        result = runner.run(run_id)
    except BudgetExceededError as exc:
        _error(str(exc))
        with conn:
            update_run_status(conn, run_id, "failed")
        sys.exit(1)
    except Exception as exc:
        # Covers openai.APIConnectionError (Ollama not running) and others.
        _error(
            f"Pipeline failed: {exc}\n\n"
            "  Is Ollama running?  Start it with: ollama serve\n"
            f"  Is the model installed?  Run: ollama pull {cfg.default_model}"
        )
        with conn:
            update_run_status(conn, run_id, "failed")
        sys.exit(1)

    # ── 4. Persist and display ────────────────────────────────────────────────
    click.echo("\n[3/4] Persisting results to database ...")
    try:
        _persist_result(conn, run_id, result)
    except Exception as exc:
        _error(f"Failed to persist result: {exc}")

    with conn:
        update_run_status(conn, run_id, "completed")

    click.echo(
        f"[4/4] Done. "
        f"({result.ollama_calls_used} Ollama calls, {result.wall_clock_seconds}s)\n"
    )
    _print_summary(result)


# ─────────────────────────────────────────────────────────────────────────────
# campaign command
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
@click.option(
    "--max-runs",
    default=None,
    type=int,
    help=(
        "Maximum pipeline runs in this campaign.  "
        "Defaults to LAB_MAX_CAMPAIGN_RUNS (currently 5)."
    ),
)
@click.option(
    "--max-calls",
    default=None,
    type=int,
    help=(
        "Total LLM call budget across all runs (including retries).  "
        "Defaults to LAB_MAX_CAMPAIGN_CALLS (currently 60)."
    ),
)
@click.option(
    "--resume",
    default=None,
    metavar="CAMPAIGN_ID",
    help="Resume a previously interrupted campaign by its ID or 8-character prefix.",
)
@click.option(
    "--env-file",
    default=None,
    help="Path to a .env configuration file.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Print each agent's deliberation as it completes.",
)
def campaign(
    max_runs: int | None,
    max_calls: int | None,
    resume: str | None,
    env_file: str | None,
    verbose: bool,
) -> None:
    """
    Run an autonomous multi-run research campaign.

    The campaign loops the 8-agent pipeline, feeding each run's results back
    as context for the next Researcher.  It stops when the Recommender decides
    to archive a finding, when stagnation is detected, or when the budget runs
    out.

    Use --resume CAMPAIGN_ID to continue an interrupted campaign.
    """
    cfg = load_config(env_file)

    # Apply CLI overrides on top of config defaults.
    if max_runs is not None:
        cfg = cfg.model_copy(update={"max_campaign_runs": max_runs})
    if max_calls is not None:
        cfg = cfg.model_copy(update={"max_campaign_calls": max_calls})

    click.echo("═" * 60)
    click.echo("  SPY Research Lab — Autonomous Campaign")
    click.echo("═" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    click.echo(f"\n[1/3] Loading data from {cfg.data_path} ...")
    try:
        load_result = load_csv(cfg.data_path)
    except FileNotFoundError as exc:
        _error(str(exc))
        import sys; sys.exit(1)

    if load_result.errors:
        click.echo(f"  ⚠  Skipped {load_result.skipped_rows} rows: {load_result.errors[:3]}")

    try:
        validate_partition_size(load_result.candles)
    except ValueError as exc:
        _error(str(exc))
        import sys; sys.exit(1)

    click.echo(f"  ✓  {len(load_result.candles)} candles loaded.")

    # ── Initialise DB ─────────────────────────────────────────────────────────
    conn = init_db(cfg.db_path)

    # ── Run or resume campaign ────────────────────────────────────────────────
    click.echo(f"\n[2/3] Starting campaign ...")
    click.echo(f"  Model:      {cfg.default_model}")
    click.echo(f"  Budget:     {cfg.max_campaign_runs} runs | {cfg.max_campaign_calls} total calls")
    click.echo(f"  Per-run:    up to {cfg.max_ollama_calls} calls, {cfg.max_wall_clock_seconds}s")
    click.echo(f"  Ollama:     {cfg.ollama_base_url}")

    on_step = _make_verbose_callback() if verbose else None
    runner = CampaignRunner(cfg, load_result.candles, conn, on_step=on_step)

    try:
        if resume:
            # Allow short prefix (first 8 chars) as well as full UUID.
            campaign_id = _resolve_campaign_id(conn, resume)
            if campaign_id is None:
                _error(f"No running campaign found matching '{resume}'.")
                import sys; sys.exit(1)
            click.echo(f"  Resuming:   {campaign_id[:8]}...\n")
            result = CampaignRunner.resume(campaign_id, cfg, load_result.candles, conn)
        else:
            click.echo()
            result = runner.run()
    except ValueError as exc:
        _error(str(exc))
        import sys; sys.exit(1)
    except Exception as exc:
        _error(
            f"Campaign failed unexpectedly: {exc}\n\n"
            "  Is Ollama running?  Start it with: ollama serve\n"
            f"  Is the model installed?  Run: ollama pull {cfg.default_model}"
        )
        import sys; sys.exit(1)

    # ── Display results ───────────────────────────────────────────────────────
    click.echo(f"\n[3/3] Campaign complete.\n")
    _print_campaign_summary(result)


def _resolve_campaign_id(conn, prefix: str) -> str | None:
    """Find a 'running' campaign whose ID starts with prefix (or is the full ID)."""
    from lab.db.repository import get_recent_campaigns
    campaigns = get_recent_campaigns(conn, limit=20)
    for c in campaigns:
        if c["status"] == "running" and c["id"].startswith(prefix):
            return c["id"]
    return None


def _print_campaign_summary(result: CampaignResult) -> None:
    """Print a table-style summary of all runs in the campaign."""
    status_colour = {
        "completed": "green",
        "stagnated": "yellow",
        "budget_exhausted": "yellow",
        "failed": "red",
    }.get(result.status, "white")

    click.echo("═" * 60)
    click.echo("  CAMPAIGN SUMMARY")
    click.echo("═" * 60)
    click.echo(
        f"  Campaign:    {result.campaign_id[:8]}..."
    )
    click.echo(
        f"  Status:      "
        + click.style(result.status.upper().replace("_", " "), fg=status_colour, bold=True)
    )
    click.echo(f"  Runs:        {result.runs_completed}")
    click.echo(f"  Total calls: {result.total_calls_used}")
    click.echo(f"  Stop reason: {result.stop_reason}")

    if result.run_results:
        click.echo("\n  Run-by-run decisions:")
        click.echo(f"  {'#':3}  {'Decision':12}  {'p-value':8}  {'Events':7}  {'Params'}")
        click.echo("  " + "─" * 54)
        for i, run in enumerate(result.run_results, start=1):
            hyp = run.hypothesis
            exp = run.experiment_result
            judge = run.judge_decision
            params = hyp.get("experiment_params", {})
            decision = judge.get("decision", "?").upper()
            decision_colour = {"ACCEPT": "green", "REJECT": "red"}.get(decision, "yellow")
            p_str = f"{exp.get('p_value', 0):.4f}"
            n_str = str(exp.get("event_count", "?"))
            param_str = (
                f"short={params.get('short_window')}, "
                f"long={params.get('long_window')}, "
                f"horizon={params.get('horizon_periods')}"
            )
            click.echo(
                f"  {i:3}  "
                + click.style(f"{decision:12}", fg=decision_colour)
                + f"  {p_str:8}  {n_str:7}  {param_str}"
            )

    click.echo("\n" + "═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# replay command (Milestone 4)
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
@click.option(
    "--window-size",
    default=None,
    type=int,
    help="Number of candles per walk-forward window.  Defaults to LAB_REPLAY_WINDOW_SIZE (300).",
)
@click.option(
    "--step-size",
    default=None,
    type=int,
    help="Candles to advance between windows.  Defaults to LAB_REPLAY_STEP_SIZE (60).",
)
@click.option(
    "--horizons",
    default=None,
    type=str,
    metavar="LIST",
    help="Comma-separated prediction horizons, e.g. '5,15,30,60'.  Defaults to LAB_REPLAY_HORIZONS.",
)
@click.option(
    "--max-calls",
    default=None,
    type=int,
    help="Total LLM call budget across all windows.  Defaults to LAB_MAX_REPLAY_CALLS (200).",
)
@click.option("--env-file", default=None, help="Path to a .env configuration file.")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Print each agent's deliberation as it completes.",
)
def replay(
    window_size: int | None,
    step_size: int | None,
    horizons: str | None,
    max_calls: int | None,
    env_file: str | None,
    verbose: bool,
) -> None:
    """
    Walk-forward historical replay with multi-horizon experiment.

    Slides a candle window through the full dataset.  At each step a 4-agent
    pipeline (Researcher → Experiment → Judge → Recommender → Summarizer)
    tests the sma_crossover_multihorizon experiment on that window.

    Results are persisted to SQLite.  Use 'python -m lab replay --list' to
    see past replay runs.
    """
    cfg = load_config(env_file)

    if window_size is not None:
        cfg = cfg.model_copy(update={"replay_window_size": window_size})
    if step_size is not None:
        cfg = cfg.model_copy(update={"replay_step_size": step_size})
    if horizons is not None:
        parsed = [int(h.strip()) for h in horizons.split(",") if h.strip()]
        cfg = cfg.model_copy(update={"replay_horizons": parsed})
    if max_calls is not None:
        cfg = cfg.model_copy(update={"max_replay_calls": max_calls})

    click.echo("═" * 60)
    click.echo("  SPY Research Lab — Walk-Forward Replay")
    click.echo("═" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    click.echo(f"\n[1/4] Loading data from {cfg.data_path} ...")
    try:
        load_result = load_csv(cfg.data_path)
    except FileNotFoundError as exc:
        _error(str(exc))
        sys.exit(1)

    if load_result.errors:
        click.echo(f"  ⚠  Skipped {load_result.skipped_rows} rows: {load_result.errors[:3]}")

    try:
        validate_partition_size(load_result.candles)
    except ValueError as exc:
        _error(str(exc))
        sys.exit(1)

    click.echo(f"  ✓  {len(load_result.candles)} candles loaded.")

    # ── Build engine and compute step count ───────────────────────────────────
    try:
        engine = ReplayEngine(
            load_result.candles,
            window_size=cfg.replay_window_size,
            step_size=cfg.replay_step_size,
        )
    except ValueError as exc:
        _error(str(exc))
        sys.exit(1)

    total_steps = engine.total_steps()
    horizons_str = ", ".join(str(h) for h in cfg.replay_horizons)

    click.echo(f"\n[2/4] Replay plan:")
    click.echo(f"  Windows:    {total_steps} × {cfg.replay_window_size} candles")
    click.echo(f"  Step size:  {cfg.replay_step_size} candles")
    click.echo(f"  Horizons:   {horizons_str} bars")
    click.echo(f"  Budget:     {cfg.max_replay_calls} total LLM calls")
    click.echo(f"  Per-window: up to {cfg.max_ollama_calls} calls")
    click.echo(f"  Model:      {cfg.default_model}")
    click.echo(f"  Ollama:     {cfg.ollama_base_url}")

    # ── Initialise DB and replay run ──────────────────────────────────────────
    conn = init_db(cfg.db_path)
    replay_id = str(uuid.uuid4())
    ts = now_iso()

    with conn:
        save_replay_run(conn, ReplayRunRecord(
            id=replay_id,
            created_at=ts,
            status="running",
            window_size=cfg.replay_window_size,
            step_size=cfg.replay_step_size,
            horizons=cfg.replay_horizons,
            steps_completed=0,
            total_calls_used=0,
            completed_at=None,
        ))

    click.echo(f"\n  Replay ID: {replay_id[:8]}...\n")

    # ── Run each window ───────────────────────────────────────────────────────
    click.echo(f"[3/4] Running {total_steps} windows ...")
    step_results: list[ReplayStepResult] = []
    total_calls_used = 0
    final_status = "completed"

    for step in engine.steps():
        remaining_calls = cfg.max_replay_calls - total_calls_used
        per_step_budget = min(remaining_calls, cfg.max_ollama_calls)

        if per_step_budget < 5:
            click.echo(
                f"\n  Budget exhausted after {step.step_index} windows "
                f"({total_calls_used} calls used)."
            )
            final_status = "budget_exhausted"
            break

        replay_on_step = (
            _make_verbose_callback(_REPLAY_STEP_LABELS) if verbose else None
        )
        if verbose:
            click.echo(
                f"\n{'─' * 60}\n"
                f"  Window {step.step_index + 1}/{total_steps}  "
                f"candles {step.window_start}–{step.window_end}  "
                f"({step.timestamp_start.strftime('%Y-%m-%d %H:%M')} → "
                f"{step.timestamp_end.strftime('%Y-%m-%d %H:%M')})"
            )
        runner = ReplayPipelineRunner(cfg, step, budget=per_step_budget, on_step=replay_on_step)
        try:
            result = runner.run()
        except BudgetExceededError as exc:
            click.echo(f"\n  ⚠  Window {step.step_index} hit budget: {exc}")
            final_status = "budget_exhausted"
            break
        except Exception as exc:
            _error(
                f"Window {step.step_index} failed: {exc}\n"
                "  Is Ollama running?  Start it with: ollama serve"
            )
            final_status = "failed"
            with conn:
                update_replay_run(conn, replay_id, "failed",
                                  len(step_results), total_calls_used, now_iso())
            sys.exit(1)

        step_results.append(result)
        total_calls_used += result.ollama_calls_used

        decision = result.judge_decision.get("decision", "?").upper()
        decision_colour = {"ACCEPT": "green", "REJECT": "red"}.get(decision, "yellow")
        click.echo(
            f"  [{step.step_index + 1:3}/{total_steps}] "
            + click.style(f"{decision:12}", fg=decision_colour)
            + f"  calls={result.ollama_calls_used:3}  "
            f"short={result.hypothesis['experiment_params']['short_window']}, "
            f"long={result.hypothesis['experiment_params']['long_window']}"
        )

        # Persist this step
        with conn:
            save_replay_step(conn, ReplayStepRecord(
                id=new_id(),
                replay_id=replay_id,
                step_index=result.step_index,
                window_start=result.window_start,
                window_end=result.window_end,
                timestamp_start=result.timestamp_start,
                timestamp_end=result.timestamp_end,
                parameters=result.hypothesis["experiment_params"],
                result=result.experiment_result,
                judge_decision=result.judge_decision,
                recommendation=result.recommendation,
                summary=result.summarizer_output.get("summary", ""),
                ollama_calls=result.ollama_calls_used,
                created_at=now_iso(),
            ))
            update_replay_run(conn, replay_id, "running",
                              len(step_results), total_calls_used)

    # ── Finalise ──────────────────────────────────────────────────────────────
    with conn:
        update_replay_run(conn, replay_id, final_status,
                          len(step_results), total_calls_used, now_iso())

    click.echo(f"\n[4/4] Replay complete.")
    _print_replay_summary(replay_id, final_status, step_results, total_calls_used)


def _print_comparison_summary(result: ComparisonResult) -> None:
    """Print a formatted side-by-side comparison table."""
    click.echo("\n" + "═" * 72)
    click.echo("  COMPARISON SUMMARY")
    click.echo("═" * 72)
    click.echo(f"  ID:        {result.comparison_id[:8]}...")
    click.echo(f"  Samples:   {result.n_samples}  |  LLM budget: {result.llm_budget} calls")
    click.echo()

    header = f"  {'Approach':18}  {'Runs':>6}  {'Sig.':>5}  {'Rate':>7}  {'Best p':>8}  {'Calls':>6}  {'Calls/Sig':>9}"
    click.echo(header)
    click.echo("  " + "─" * 68)

    for s in result.summaries:
        approach_label = s.approach.replace("_", " ").title()
        sig_colour = "green" if s.significant_count > 0 else "red"
        cps = f"{s.calls_per_significant:.1f}" if s.calls_per_significant != float("inf") else "∞"
        click.echo(
            f"  {approach_label:18}  {s.runs:6}  "
            + click.style(f"{s.significant_count:5}", fg=sig_colour)
            + f"  {s.significant_rate:6.1%}  {s.best_p_value:8.4f}  {s.calls_used:6}  {cps:>9}"
        )

    click.echo()
    click.echo("  Columns: Runs=experiments executed, Sig=significant findings (p<0.05 ∧ n≥10),")
    click.echo("           Rate=sig/runs, Best p=lowest p-value found, Calls/Sig=LLM efficiency")
    click.echo("\n" + "═" * 72)


def _print_replay_summary(
    replay_id: str,
    status: str,
    step_results: list[ReplayStepResult],
    total_calls: int,
) -> None:
    """Print a concise replay summary."""
    status_colour = {
        "completed": "green",
        "budget_exhausted": "yellow",
        "failed": "red",
    }.get(status, "white")

    click.echo("\n" + "═" * 60)
    click.echo("  REPLAY SUMMARY")
    click.echo("═" * 60)
    click.echo(f"  Replay ID:   {replay_id[:8]}...")
    click.echo(
        f"  Status:      "
        + click.style(status.upper().replace("_", " "), fg=status_colour, bold=True)
    )
    click.echo(f"  Windows run: {len(step_results)}")
    click.echo(f"  Total calls: {total_calls}")

    if step_results:
        accepts = sum(
            1 for r in step_results
            if r.judge_decision.get("decision") == "accept"
        )
        click.echo(f"  Accept rate: {accepts}/{len(step_results)} windows")

        click.echo("\n  Sample decisions (first 10 windows):")
        click.echo(f"  {'#':>4}  {'Decision':12}  {'Action':18}  {'SMA params'}")
        click.echo("  " + "─" * 54)
        for r in step_results[:10]:
            decision = r.judge_decision.get("decision", "?").upper()
            action = r.recommendation.get("action", "?")
            decision_colour = {"ACCEPT": "green", "REJECT": "red"}.get(decision, "yellow")
            params = r.hypothesis["experiment_params"]
            click.echo(
                f"  {r.step_index + 1:4}  "
                + click.style(f"{decision:12}", fg=decision_colour)
                + f"  {action:18}  "
                f"short={params['short_window']}, long={params['long_window']}"
            )
        if len(step_results) > 10:
            click.echo(f"  ... and {len(step_results) - 10} more windows")

    click.echo("\n" + "═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# compare command (Milestone 5)
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
@click.option(
    "--n-samples",
    default=None,
    type=int,
    help="Parameter draws for baseline + single-agent arms.  Defaults to LAB_COMPARE_N_SAMPLES (50).",
)
@click.option(
    "--llm-budget",
    default=None,
    type=int,
    help="Total LLM call budget for single-agent and multi-agent arms.  Defaults to LAB_COMPARE_LLM_BUDGET (60).",
)
@click.option("--env-file", default=None, help="Path to a .env configuration file.")
def compare(
    n_samples: int | None,
    llm_budget: int | None,
    env_file: str | None,
) -> None:
    """
    Compare baseline vs single-agent vs multi-agent research approaches.

    Runs three evaluations on the same dataset with the same LLM budget:

    \b
      1. Baseline     — random parameter grid search, 0 LLM calls
      2. Single-agent — Researcher only (no deliberation), 1 call/run
      3. Multi-agent  — full 8-agent CampaignRunner

    Prints a side-by-side table of significant findings, best p-value,
    and LLM efficiency (calls per significant finding).
    """
    cfg = load_config(env_file)

    if n_samples is not None:
        cfg = cfg.model_copy(update={"compare_n_samples": n_samples})
    if llm_budget is not None:
        cfg = cfg.model_copy(update={"compare_llm_budget": llm_budget})

    click.echo("═" * 60)
    click.echo("  SPY Research Lab — Comparative Evaluation")
    click.echo("═" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    click.echo(f"\n[1/5] Loading data from {cfg.data_path} ...")
    try:
        load_result = load_csv(cfg.data_path)
    except FileNotFoundError as exc:
        _error(str(exc))
        sys.exit(1)

    if load_result.errors:
        click.echo(f"  ⚠  Skipped {load_result.skipped_rows} rows: {load_result.errors[:3]}")

    try:
        validate_partition_size(load_result.candles)
    except ValueError as exc:
        _error(str(exc))
        sys.exit(1)

    click.echo(f"  ✓  {len(load_result.candles)} candles loaded.")

    conn = init_db(cfg.db_path)

    click.echo(f"\n  Samples (baseline + single-agent): {cfg.compare_n_samples}")
    click.echo(f"  LLM budget (single + multi):       {cfg.compare_llm_budget} calls")
    click.echo(f"  Model:                             {cfg.default_model}")

    # ── Run comparison ─────────────────────────────────────────────────────────
    click.echo(f"\n[2/5] Running baseline (random grid search) ...")
    runner = ComparisonRunner(cfg, load_result.candles, conn)

    # Run baseline first for a quick progress indication
    from lab.evaluation.baseline import BaselineEvaluator
    baseline_result = BaselineEvaluator(load_result.candles).run(n_samples=cfg.compare_n_samples)
    click.echo(
        f"  ✓  {baseline_result.n_samples} samples, "
        f"{baseline_result.significant_count} significant findings, "
        f"best p={baseline_result.best_p_value:.4f}"
    )

    click.echo(f"\n[3/5] Running single-agent ({cfg.compare_n_samples} runs, budget={cfg.compare_llm_budget}) ...")
    from lab.evaluation.single_agent import SingleAgentRunner
    single_runner = SingleAgentRunner(cfg, load_result.candles, budget=cfg.compare_llm_budget)
    try:
        single_result = single_runner.run(n_runs=cfg.compare_n_samples)
    except Exception as exc:
        _error(
            f"Single-agent evaluation failed: {exc}\n"
            "  Is Ollama running?  Start it with: ollama serve"
        )
        sys.exit(1)

    click.echo(
        f"  ✓  {single_result.n_runs} runs, "
        f"{single_result.significant_count} significant findings, "
        f"{single_result.total_calls_used} calls"
    )

    click.echo(f"\n[4/5] Running multi-agent CampaignRunner (budget={cfg.compare_llm_budget}) ...")
    from lab.campaign.runner import CampaignRunner as _CR
    from lab.config import LabConfig as _LC
    multi_cfg = cfg.model_copy(update={"max_campaign_calls": cfg.compare_llm_budget})
    multi_runner = _CR(multi_cfg, load_result.candles, conn)
    try:
        multi_result = multi_runner.run()
    except Exception as exc:
        _error(
            f"Multi-agent evaluation failed: {exc}\n"
            "  Is Ollama running?  Start it with: ollama serve"
        )
        sys.exit(1)

    click.echo(
        f"  ✓  {multi_result.runs_completed} runs, "
        f"{multi_result.total_calls_used} calls, "
        f"status: {multi_result.status}"
    )

    # ── Build and persist result ───────────────────────────────────────────────
    from lab.evaluation.runner import ComparisonResult as _CR2
    result = _CR2(
        comparison_id=str(uuid.uuid4()),
        created_at=now_iso(),
        n_samples=cfg.compare_n_samples,
        llm_budget=cfg.compare_llm_budget,
        baseline=baseline_result,
        single_agent=single_result,
        multi_agent=multi_result,
    )

    click.echo(f"\n[5/5] Persisting comparison to database ...")
    d = result.to_dict()
    with conn:
        save_comparison(conn, Comparison(
            id=result.comparison_id,
            created_at=result.created_at,
            n_samples=result.n_samples,
            llm_budget=result.llm_budget,
            baseline_json=d["baseline"],
            single_json=d["single_agent"],
            multi_json=d["multi_agent"],
            completed_at=now_iso(),
        ))

    _print_comparison_summary(result)


# ─────────────────────────────────────────────────────────────────────────────
# history command
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--limit", default=5, show_default=True, help="Number of recent runs to show.")
@click.option("--run-id", default=None, help="Show full detail for a specific run ID.")
@click.option("--env-file", default=None)
def history(limit: int, run_id: str | None, env_file: str | None) -> None:
    """Inspect stored research runs and their outcomes."""
    cfg = load_config(env_file)
    conn = init_db(cfg.db_path)

    if run_id:
        details = get_run_details(conn, run_id)
        if details is None:
            click.echo(f"No run found with ID: {run_id}")
            return
        click.echo(json.dumps(details, indent=2))
        return

    runs = get_recent_runs(conn, limit=limit)
    if not runs:
        click.echo("No research runs found.  Run `python -m lab research` to start one.")
        return

    click.echo(f"\n{'ID':8}  {'Status':12}  {'Model':20}  {'Created':30}")
    click.echo("─" * 74)
    for r in runs:
        click.echo(
            f"{r['id'][:8]}  {r['status']:12}  {r['model']:20}  {r['created_at']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _persist_result(conn, run_id: str, result: PipelineResult) -> None:
    """Save the complete pipeline result to SQLite.

    Persists the full provenance chain: hypothesis → experiment → all eight
    agent arguments → judge decision → recommendation → summary.
    """
    hyp_data = result.hypothesis
    da_data = result.devils_advocate
    rebuttal_data = result.researcher_rebuttal
    exp_result_data = result.experiment_result
    exp_meta = result.experiment_meta
    det_data = result.detractor_arguments
    judge_data = result.judge_decision
    rec_data = result.recommendation
    summary_data = result.summarizer_output

    hyp_id = new_id()
    exp_id = new_id()
    ts = now_iso()
    params = hyp_data.get("experiment_params", {})

    with conn:
        save_hypothesis(conn, Hypothesis(
            id=hyp_id,
            run_id=run_id,
            statement=hyp_data.get("hypothesis_statement", ""),
            experiment_type=hyp_data.get("experiment_type", ""),
            parameters=params,
            created_at=ts,
        ))

        save_experiment(conn, Experiment(
            id=exp_id,
            hypothesis_id=hyp_id,
            experiment_type=hyp_data.get("experiment_type", ""),
            parameters=params,
            result=exp_result_data,
            calculation_method=exp_meta.get("method", ""),
            candles_used=exp_meta.get("candles_used", 0),
            created_at=ts,
        ))

        _save_arg(conn, exp_id, "researcher",
                  hyp_data.get("hypothesis_statement", ""), hyp_data, ts)
        _save_arg(conn, exp_id, "devils_advocate",
                  da_data.get("objection", ""), da_data, ts)
        _save_arg(conn, exp_id, "researcher_rebuttal",
                  rebuttal_data.get("rebuttal", ""), rebuttal_data, ts)
        _save_arg(conn, exp_id, "detractor",
                  det_data.get("primary_challenge", ""), det_data, ts)
        _save_arg(conn, exp_id, "judge",
                  judge_data.get("rationale", ""), judge_data, ts)

        save_judge_decision(conn, JudgeDecision(
            id=new_id(),
            experiment_id=exp_id,
            decision=judge_data.get("decision", "inconclusive"),
            rationale=judge_data.get("rationale", ""),
            confidence=judge_data.get("confidence", "low"),
            created_at=ts,
        ))

        if rec_data:
            save_recommendation(conn, Recommendation(
                id=new_id(),
                run_id=run_id,
                action=rec_data.get("action", "needs_more_data"),
                rationale=rec_data.get("rationale", ""),
                created_at=ts,
            ))

        if summary_data:
            save_summary(conn, Summary(
                id=new_id(),
                run_id=run_id,
                summary_text=summary_data.get("summary", ""),
                created_at=ts,
            ))


def _save_arg(conn, experiment_id: str, role: str,
              text: str, raw: dict, ts: str) -> None:
    """Insert one agent_arguments row."""
    save_agent_argument(conn, AgentArgument(
        id=new_id(),
        experiment_id=experiment_id,
        agent_role=role,
        argument_text=text,
        raw_response=json.dumps(raw),
        created_at=ts,
    ))


def _print_summary(result: PipelineResult) -> None:
    """Print a human-readable research summary to the console."""
    hyp = result.hypothesis
    da = result.devils_advocate
    rebuttal = result.researcher_rebuttal
    exp = result.experiment_result
    det = result.detractor_arguments
    judge = result.judge_decision
    rec = result.recommendation
    summary = result.summarizer_output

    click.echo("═" * 60)
    click.echo("  RESEARCH SUMMARY")
    click.echo("═" * 60)

    click.echo("\n▶ HYPOTHESIS")
    click.echo(f"  {hyp.get('hypothesis_statement', 'N/A')}")
    params = hyp.get("experiment_params", {})
    click.echo(
        f"  Parameters: short={params.get('short_window')} | "
        f"long={params.get('long_window')} | "
        f"horizon={params.get('horizon_periods')} bars"
    )

    click.echo("\n▶ DEVIL'S ADVOCATE  (pre-experiment)")
    da_conf = da.get("confidence", 0)
    click.echo(f"  Objection ({da_conf:.0%} confidence):  {da.get('objection', 'N/A')}")
    click.echo(f"  Targeted assumption:              {da.get('targeted_assumption', 'N/A')}")

    concedes = rebuttal.get("concedes_point", False)
    concedes_label = click.style("concedes point", fg="yellow") if concedes else "stands firm"
    click.echo(f"\n▶ RESEARCHER REBUTTAL  ({concedes_label})")
    click.echo(f"  {rebuttal.get('rebuttal', 'N/A')}")

    click.echo("\n▶ DETERMINISTIC EVIDENCE")
    click.echo(f"  Events found:          {exp.get('event_count', 'N/A')}")
    click.echo(f"  Positive outcomes:     {exp.get('positive_count', 'N/A')}")
    win_pct = exp.get("win_rate", 0) * 100
    click.echo(f"  Win rate:              {win_pct:.1f}%")
    click.echo(f"  p-value:               {exp.get('p_value', 'N/A'):.4f}")
    click.echo(f"  Significant (p<0.05):  {exp.get('statistically_significant', 'N/A')}")
    click.echo(f"  Min sample met (n≥10): {exp.get('minimum_events_met', 'N/A')}")
    click.echo(f"  Candles analysed:      {exp.get('candles_used', 'N/A')}")

    click.echo("\n▶ DETRACTOR")
    click.echo(f"  Primary challenge:  {det.get('primary_challenge', 'N/A')}")
    click.echo(f"  Methodological:     {det.get('methodological_concern', 'N/A')}")
    click.echo(f"  Alternative:        {det.get('alternative_explanation', 'N/A')}")

    decision = judge.get("decision", "N/A").upper()
    confidence = judge.get("confidence", "N/A")
    decision_colour = {"ACCEPT": "green", "REJECT": "red"}.get(decision, "yellow")
    click.echo("\n▶ JUDGE DECISION")
    click.echo(f"  Decision:   {click.style(decision, fg=decision_colour, bold=True)}  (confidence: {confidence})")
    click.echo(f"  Rationale:  {judge.get('rationale', 'N/A')}")

    integrity = result.result_integrity_check
    if integrity.get("judge_altered_result") is False:
        click.echo(
            click.style(
                "\n  ✓ Result integrity confirmed — Judge cannot alter the deterministic findings.",
                fg="green",
            )
        )

    if rec:
        action = rec.get("action", "N/A")
        action_colour = {
            "archive_confirmed": "green",
            "archive_rejected": "red",
            "run_variant": "cyan",
            "needs_more_data": "yellow",
        }.get(action, "white")
        click.echo("\n▶ RECOMMENDATION")
        click.echo(f"  Action:    {click.style(action.upper().replace('_', ' '), fg=action_colour, bold=True)}")
        click.echo(f"  Rationale: {rec.get('rationale', 'N/A')}")

    if summary:
        click.echo("\n▶ SUMMARY")
        text = summary.get("summary", "")
        for line in textwrap.wrap(text, width=56):
            click.echo(f"  {line}")

    click.echo("\n" + "═" * 60)


def _error(message: str) -> None:
    click.echo(click.style(f"\n  ✗ {message}", fg="red"), err=True)


# ─────────────────────────────────────────────────────────────────────────────
# Verbose step printer
# ─────────────────────────────────────────────────────────────────────────────

_STEP_LABELS = {
    "researcher":          ("[1] RESEARCHER",          "cyan"),
    "devils_advocate":     ("[2] DEVIL'S ADVOCATE",    "yellow"),
    "researcher_rebuttal": ("[3] RESEARCHER REBUTTAL", "cyan"),
    "experiment":          ("[4] EXPERIMENT",          "blue"),
    "detractor":           ("[5] DETRACTOR",           "yellow"),
    "judge":               ("[6] JUDGE",               "magenta"),
    "recommender":         ("[7] RECOMMENDER",         "white"),
    "summarizer":          ("[8] SUMMARIZER",          "white"),
}

# Replay uses a 5-step pipeline
_REPLAY_STEP_LABELS = {
    "researcher": ("[1] RESEARCHER",   "cyan"),
    "experiment": ("[2] EXPERIMENT",   "blue"),
    "judge":      ("[3] JUDGE",        "magenta"),
    "recommender":("[4] RECOMMENDER",  "white"),
    "summarizer": ("[5] SUMMARIZER",   "white"),
}


def _verbose_on_step(step_name: str, output, labels: dict = _STEP_LABELS) -> None:
    """Print one agent step's output to the console."""
    label, colour = labels.get(step_name, (step_name.upper(), "white"))
    click.echo("\n  " + click.style(f"▶ {label}", fg=colour, bold=True))

    if step_name == "researcher":
        params = output.experiment_params
        click.echo(f"    Hypothesis: {output.hypothesis_statement}")
        click.echo(
            f"    Params: short={params.short_window}, "
            f"long={params.long_window}"
            + (f", horizon={params.horizon_periods}" if hasattr(params, 'horizon_periods') else "")
        )

    elif step_name == "devils_advocate":
        click.echo(f"    Objection ({output.confidence:.0%}): {output.objection}")
        click.echo(f"    Targeted assumption: {output.targeted_assumption}")

    elif step_name == "researcher_rebuttal":
        concedes = click.style("concedes point", fg="yellow") if output.concedes_point else "stands firm"
        click.echo(f"    Stance: {concedes}")
        click.echo(f"    {output.rebuttal}")

    elif step_name == "experiment":
        # output is a dict here (from run_experiment dispatcher or MultiHorizonResult)
        if "horizons" in output:
            # multi-horizon (replay)
            click.echo(f"    Multi-horizon results:")
            for h in output.get("horizons", []):
                sig = click.style("✓ sig", fg="green") if h.get("statistically_significant") else "✗ not sig"
                click.echo(
                    f"      horizon={h['horizon_periods']}bars  "
                    f"events={h['event_count']}  "
                    f"win={h['win_rate']:.1%}  "
                    f"p={h['p_value']:.4f}  {sig}"
                )
        else:
            sig = click.style("✓ significant", fg="green") if output.get("statistically_significant") else "✗ not significant"
            click.echo(f"    Events: {output.get('event_count')}  Win rate: {output.get('win_rate', 0):.1%}  p={output.get('p_value', 1):.4f}  {sig}")
            click.echo(f"    Min sample met (n≥10): {output.get('minimum_events_met')}")

    elif step_name == "detractor":
        click.echo(f"    Primary challenge:  {output.primary_challenge}")
        click.echo(f"    Methodological:     {output.methodological_concern}")
        click.echo(f"    Alternative:        {output.alternative_explanation}")

    elif step_name == "judge":
        decision = output.decision.upper()
        colour = {"ACCEPT": "green", "REJECT": "red"}.get(decision, "yellow")
        click.echo(
            f"    Decision: {click.style(decision, fg=colour, bold=True)}  "
            f"(confidence: {output.confidence})"
        )
        click.echo(f"    Rationale: {output.rationale}")
        if hasattr(output, "recommended_horizon"):
            click.echo(f"    Best horizon: {output.recommended_horizon} bars")

    elif step_name == "recommender":
        action = output.action
        colour = {
            "archive_confirmed": "green", "accept_signal": "green",
            "archive_rejected": "red",   "reject_signal":  "red",
            "run_variant": "cyan",
            "needs_more_data": "yellow",
        }.get(action, "white")
        click.echo(f"    Action: {click.style(action.upper().replace('_', ' '), fg=colour, bold=True)}")
        click.echo(f"    {output.rationale}")

    elif step_name == "summarizer":
        click.echo(f"    {output.summary}")


def _make_verbose_callback(labels: dict = _STEP_LABELS):
    """Return a step callback that uses the given label set."""
    def _cb(step_name: str, output) -> None:
        _verbose_on_step(step_name, output, labels)
    return _cb
