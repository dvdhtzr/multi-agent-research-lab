# SPY Research Lab

A local multi-agent research pipeline that debates, tests, and evaluates historical market hypotheses — entirely on your machine, with no paid APIs, no cloud services, and no external dependencies beyond Python and Ollama.

Eight AI agents argue with each other. One deterministic Python function is the only thing that gets to produce numbers. The agents can't change them.

---

## Quick start

```bash
git clone <your-repo-url>
cd spy-research-lab
pip install -e ".[dev]"
ollama pull gemma4
python -m lab research
```

That's it. No Docker. No Node.js. No configuration files required.

---

## What it does

Running `python -m lab research` kicks off a structured eight-agent deliberation:

```
[1] Researcher        Proposes a hypothesis: "When the 10-period SMA crosses above the
                      40-period SMA, price rises over the next 15 minutes."

[2] Devil's Advocate  Challenges it before any data is examined:
                      "Your fixed-window parameters ignore market regime changes."

[3] Rebuttal          Researcher responds — may concede the point or push back.

[4] EXPERIMENT        Deterministic Python: scipy.stats.binomtest on 1,200 candles.
                      17 events found. Win rate: 64.7%. p-value: 0.1662.
                      (The agents receive these numbers. They cannot change them.)

[5] Detractor         Challenges the interpretation post-experiment:
                      "p=0.1662 exceeds the 0.05 threshold — this could be chance."

[6] Judge             Makes the final call: REJECT (confidence: high).
                      Rationale must cite the actual p-value.

[7] Recommender       Suggests next action: ARCHIVE REJECTED.

[8] Summarizer        Writes a plain-English summary for non-technical readers.
```

The whole run takes about 50 seconds on an 8B model. Everything is saved to SQLite.

---

## The core idea: typed agents with a trust boundary

Every LLM call goes through [instructor](https://github.com/jxnl/instructor), which wraps Ollama's OpenAI-compatible API and enforces that the model's response matches a Pydantic schema before the pipeline continues. If the model returns malformed JSON or missing fields, instructor retries automatically — the pipeline never receives unvalidated data.

The trust boundary is enforced by schema design, not by prompting:

```python
class JudgeOutput(BaseModel):
    decision: Literal["accept", "reject", "inconclusive"]
    rationale: str
    confidence: Literal["high", "medium", "low"]
    # No p_value field. No event_count field.
    # The Judge structurally cannot alter the deterministic results.
```

The `JudgeOutput` schema has nowhere to put a different p-value. The integrity check at the end of every run is therefore guaranteed by the type system — not by trust.

```
  ✓ Result integrity confirmed — Judge cannot alter the deterministic findings.
```

> **The model proposes. Deterministic software validates.**

This is the central architectural lesson of the project: use typed schemas to enforce capability boundaries, not just policy prompts.

---

## Architecture

```
python -m lab research
        │
        ▼
   PipelineRunner  (src/lab/pipeline/runner.py)
        │
        ├─[1] Researcher ──────────────────────────────► Ollama
        │       └─ ResearcherOutput: hypothesis_statement, experiment_params
        │
        ├─[2] Devil's Advocate ────────────────────────► Ollama
        │       └─ DevilsAdvocateOutput: objection, targeted_assumption, confidence
        │
        ├─[3] Researcher Rebuttal ─────────────────────► Ollama
        │       └─ ResearcherRebuttalOutput: rebuttal, concedes_point
        │
        ├─[4] Experiment ─────────────────────────────── Python (no LLM)
        │       └─ scipy.stats.binomtest → event_count, win_rate, p_value, …
        │
        ├─[5] Detractor ───────────────────────────────► Ollama
        │       └─ DetractorOutput: primary_challenge, methodological_concern, …
        │
        ├─[6] Judge ───────────────────────────────────► Ollama
        │       └─ JudgeOutput: decision, rationale, confidence
        │
        ├─[7] Recommender ─────────────────────────────► Ollama
        │       └─ RecommenderOutput: action, rationale
        │
        └─[8] Summarizer ──────────────────────────────► Ollama
                └─ SummarizerOutput: summary

        ▼
   SQLite (research.db)
   Full provenance: hypothesis → experiment → all agent arguments → decision
```

Each arrow is a call to `instructor.from_openai(...)`, which hits Ollama's `/v1/chat/completions` endpoint and validates the response against the Pydantic schema before returning.

---

## Prerequisites

| Component | Version | Notes |
|---|---|---|
| Python | ≥ 3.12 | [python.org](https://python.org) |
| Ollama | any recent | [ollama.com](https://ollama.com) |

Nothing else.

---

## Installation

```bash
git clone <your-repo-url>
cd spy-research-lab
pip install -e ".[dev]"
ollama pull gemma4
```

`pip install -e ".[dev]"` installs the project plus `instructor`, `openai`, `pydantic`, `scipy`, `click`, and `pytest`.

**Any model that follows JSON instructions works.** `gemma4` (8B, ~9.6 GB) is the default. For faster runs, try `llama3.2:3b`. For better reasoning, try `gemma4:26b` or `qwen3:30b`. Switch with `LAB_DEFAULT_MODEL=qwen3:30b python -m lab research` or via `.env`.

---

## Running

Ollama must be running (`ollama serve` if it isn't already):

```bash
python -m lab research          # one 8-agent pipeline run
python -m lab campaign          # autonomous multi-run campaign
```

```bash
python -m lab history           # list recent single runs
python -m lab history --run-id ecba8236   # full JSON detail for one run
```

### Autonomous campaigns

`python -m lab campaign` loops the full pipeline repeatedly. After each run it:

1. Feeds the prior results back to the next Researcher ("don't repeat these")
2. Routes on the Recommender's action (`run_variant` or `needs_more_data` → continue; `archive_*` → stop)
3. Detects stagnation (3 consecutive runs with nearly identical SMA windows → stop)
4. Enforces a total call budget across all runs, including validation retries

```bash
python -m lab campaign --max-runs 5 --max-calls 60
python -m lab campaign --resume ecba8236   # resume an interrupted campaign
```

The campaign ID is printed at start. If interrupted, resume it by its 8-character prefix.

---

## Tests

```bash
pytest tests/unit/ tests/integration/ -v
```

**74 tests, ~3 seconds, no Ollama required.**

The integration tests replace the instructor client with a mock that returns canned typed responses, so the full 8-agent orchestration is exercised — including budget enforcement and experiment dispatch — without calling Ollama.

```bash
# Live end-to-end test (requires Ollama + model)
LAB_RUN_SMOKE_TEST=1 pytest tests/smoke/ -v -s
```

---

## Configuration

All settings have sensible defaults. A `.env` file is optional.

```bash
cp config.example.env .env
```

| Variable | Default | Description |
|---|---|---|
| `LAB_DEFAULT_MODEL` | `gemma4:latest` | Ollama model used by all agents |
| `LAB_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `LAB_DATA_PATH` | `data/synthetic/spy_sample.csv` | OHLCV CSV to analyse |
| `LAB_DB_PATH` | `research.db` | SQLite database file |
| `LAB_MAX_OLLAMA_CALLS` | `12` | Hard limit: Ollama calls **per single run** (including retries) |
| `LAB_MAX_WALL_CLOCK_SECONDS` | `300` | Hard limit: seconds per single run |
| `LAB_MAX_CAMPAIGN_RUNS` | `5` | Maximum pipeline runs in one campaign |
| `LAB_MAX_CAMPAIGN_CALLS` | `60` | Total LLM calls across an entire campaign (including retries) |

> **Cost-safety note:** the budget limits exist for locally-hosted, non-metered Ollama. If you ever adapt this project for a paid API provider, implement independently-enforced spending ceilings that no agent decision can override. The budget variables alone are not sufficient.

---

## Using your own SPY data

Place a CSV in `data/private/` (gitignored) with these columns:

```
timestamp,open,high,low,close,volume
```

Timestamps must be `datetime.fromisoformat()`-parseable (e.g. `2020-01-02 09:30:00`).

```bash
LAB_DATA_PATH=data/private/spy_2018_2024.csv python -m lab research
```

---

## Project structure

```
spy-research-lab/
├── src/lab/
│   ├── cli.py                  # Entry point: research, campaign, history
│   ├── config.py               # Pydantic BaseSettings — all configuration
│   ├── pipeline/
│   │   └── runner.py           # PipelineRunner — the 8-agent orchestration
│   ├── campaign/
│   │   └── runner.py           # CampaignRunner — multi-run autonomous loop
│   ├── agents/
│   │   ├── prompts.py          # System prompts for each agent role
│   │   └── schemas.py          # Pydantic output schemas (the typed contracts)
│   ├── research/
│   │   ├── experiments.py      # Deterministic SMA crossover calculation
│   │   └── validator.py        # Experiment spec validation
│   ├── data/
│   │   ├── loader.py           # CSV parsing, OHLCV validation
│   │   └── synthetic.py        # Reproducible synthetic data generator
│   └── db/
│       ├── schema.py           # SQLite CREATE TABLE statements
│       └── repository.py       # CRUD — no ORM, raw SQL
├── data/synthetic/
│   └── spy_sample.csv          # 1,200 synthetic candles — safe for GitHub
├── tests/
│   ├── unit/                   # Fast, no I/O, no Ollama
│   ├── integration/            # Mocked pipeline/campaign runners
│   └── smoke/                  # Live end-to-end — requires Ollama
├── config.example.env
└── pyproject.toml
```

Start with `pipeline/runner.py` to understand how the 8 agents connect, then `campaign/runner.py` to see how the multi-run loop works on top of it.

---

## Troubleshooting

**Ollama not reachable**
```
openai.APIConnectionError: Connection refused
```
Run `ollama serve`. Confirm with `curl http://localhost:11434`.

**Model not found**
```
model "gemma4:latest" not found
```
Run `ollama pull gemma4`, or set `LAB_DEFAULT_MODEL` to a model you have (`ollama list`).

**Runs are slow / wall-clock timeout**
Switch to a smaller model: `LAB_DEFAULT_MODEL=llama3.2:3b`. Or increase `LAB_MAX_WALL_CLOCK_SECONDS`.

**Budget exhausted mid-run**
```
BudgetExceededError: Ollama call budget exhausted (12 calls)
```
Increase `LAB_MAX_OLLAMA_CALLS` in `.env`. This usually means instructor is retrying often because the model isn't reliably producing valid JSON — a smaller, instruction-tuned model sometimes helps.

---

## Milestones

| Milestone | Status | Description |
|---|---|---|
| **1 — Minimal loop** | ✅ Complete | Researcher + Detractor + Judge, one experiment, full provenance |
| **2 — Full deliberation** | ✅ Complete | Devil's Advocate, Rebuttal, Recommender, Summarizer; typed pipeline via instructor |
| **3 — Autonomous campaigns** | ✅ Complete | Multi-run loop, adaptive budgets, stagnation detection, persistent memory, resumability |
| 4 — Historical replay | Planned | 5-minute cadence, four prediction horizons |
| 5 — Comparative evaluation | Planned | Baseline vs single-agent vs multi-agent |

---

## Licences

- **This project:** MIT
- **instructor:** MIT  
- **openai Python client:** MIT
- **Pydantic:** MIT
- **scipy:** BSD-3-Clause
- **Click:** BSD-3-Clause
- **Ollama:** MIT
