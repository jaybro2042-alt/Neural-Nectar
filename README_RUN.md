# Coherent Agent Project (patched)
## What's here
- `main_agent_entry.py` — simple REPL entry that loads `core.py` and provides fallback console commands.
- `core.py` — self-contained backtesting utilities (yfinance fetch, quick_refine, select_best, simple EMA strategy).
- `api_agent.patched.py` — orchestration layer using Agents SDK-style function tools, now importing local `core` instead of a missing `orchestrator.core`.

## Quick start (local, no SDK needed)
```powershell
# From this folder:
py -3 -m pip install --upgrade pandas numpy yfinance
py -3 .\main_agent_entry.py
# inside the REPL:
!!tools
ohlcv AUDCAD 5m 30d
# or backtest a spec (put one under strategies\demo_ema_rsi.json)
backtest strategies\demo_ema_rsi.json
refine strategies\demo_ema_rsi.json
```

## Optional: API-style agent
Requires `agents` package and OpenAI creds if you enable LLM chat. To run its console:
```powershell
py -3 .pi_agent.patched.py
```
It exposes function tools backed by `core.py` so you can still ensure OHLCV, run backtests, refine, and write reports.
