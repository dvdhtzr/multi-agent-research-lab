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
    get_run_details,
)
from lab.db.schema import init_db
from lab.pipeline.runner import PipelineResult, PipelineRunner, BudgetExceededError


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
def research(env_file: str | None) -> None:
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

    runner = PipelineRunner(cfg, load_result.candles)
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
