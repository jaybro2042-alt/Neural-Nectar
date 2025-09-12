
# API Agent Project (Clean Restart)

A single **API agent** with tool-calling that can *fix itself*, reorganize files, fetch data, run Backtrader, refine, and select the best strategy. It integrates the files you uploaded and outputs **Backtrader-format** specs and reports.

## Layout
```
api_agent_project/
  agent/api_agent.py          # API agent (OpenAI tool-calling; offline plan fallback)
  orchestrator/core.py        # fetch, validate, run, refine, select
  orchestrator/runner_fallback.py  # built-in Backtrader runner
  runners/                    # your runner/strategy templates (imported if present)
  tools/registry.py           # toolbelt (env, pip, read/write, reorg, validations)
  strategies/                 # scary_skeleton.json + your scary_* specs
  data/                       # CSVs get written here
  reports/                    # per-run JSON reports
  ui/console.py               # simple REPL front-end (optional)
```

## Install (Windows)
```
py -3 -m pip install --upgrade backtrader yfinance pandas openai
```

## Run with API agent (recommended)
Set your key first:
```
setx OPENAI_API_KEY "sk-..."
```
Then:
```
cd /mnt/data/api_agent_project
py -3 agent\api_agent.py -p "Use scary_skeleton to fetch data for EUR/USD 1m, run it, refine parameters, and report best metrics."
```

Interactive console:
```
py -3 ui\console.py
» Reorganize project from my uploaded files and run a full sweep on EUR/USD 1m.
```

## Reorganize the project
The agent can copy files into the canonical structure:
```
tools.reorganize_project(["/path/to/scary_*.json", "/path/to/bt_runner_template.py", "/path/to/king_tools_extra.py"])
```

## Backtrader data schema
CSV header must be:
```
datetime,open,high,low,close,volume
```
Datetime format: `YYYY-MM-DD HH:MM:SS` (UTC recommended). The orchestrator writes this layout automatically when fetching from Yahoo Finance.

## Notes
- If `runners/bt_runner_template.py` exists with `run_backtest(spec_path, csv_path)`, the agent uses it. Otherwise it falls back to a simple internal runner.
- All strategy specs are JSON and Backtrader-ready; `scary_skeleton.json` provides the shared parameters to keep children consistent.
- The agent provides a **plain-English explanation** of steps in its output. No cross-agents. Just one brain with tools.
