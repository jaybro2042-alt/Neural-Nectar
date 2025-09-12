"""
chat_api_agent.py — English-first chat entry for your API agent.

Run:
  py -3 .\chat_api_agent.py

This script provides a simple chat interface to your trading agent.  It allows
you to type natural language, and it will route your requests to either
local tools (backtest, refine, ohlcv, select_best) or, if you have an
OpenAI API key configured, to the OpenAI model.  If no API key is set or
the openai package is unavailable, it falls back to a heuristic parser
that understands certain English phrases and invokes the appropriate local
functions from core.py.
"""

import os
import re
import json
import traceback
import importlib.util
from pathlib import Path

# Try to import your orchestrator layer (api_agent.patched.py).  If it
# doesn't exist, ignore the error; the local router will still work.
api_agent = None
try:
    patched_path = Path(__file__).with_name("api_agent.patched.py")
    if patched_path.exists():
        spec = importlib.util.spec_from_file_location("api_agent_patched", str(patched_path))
        api_agent = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(api_agent)  # type: ignore
    else:
        import api_agent  # type: ignore  # noqa: F401
        api_agent = api_agent  # noqa: F401
except Exception:
    # If import fails, leave api_agent as None.  The local router will still work.
    api_agent = None

# Always import local core.  core.py should implement ensure_ohlcv,
# run_backtest_on_spec, quick_refine, select_best.
try:
    import core as CORE  # type: ignore  # noqa: F401
except Exception:
    CORE = None
    traceback.print_exc()


def _local_router(text: str) -> str:
    """Heuristic English → tool calls (no network needed).

    This function parses a natural-language request and attempts to map it to
    one of the supported functions in core.py.  If it recognizes a pattern,
    it calls the function and returns a JSON-formatted result; otherwise
    it returns a help message.
    """
    t = (text or "").strip()
    low = t.lower()

    # Try to interpret ohlcv requests: e.g. "get me ohlcv for AUDCAD 5m 30d"
    if "ohlcv" in low:
        parts = low.split()
        if len(parts) >= 4:
            sym, tf, period = parts[-3], parts[-2], parts[-1]
            try:
                result = CORE.ensure_ohlcv(sym, tf, period)  # type: ignore
                return json.dumps(result, indent=2)
            except Exception:
                traceback.print_exc()
                return "[error] failed to fetch OHLCV"

    # Interpret backtest requests when a JSON file is mentioned
    if "backtest" in low and ".json" in low:
        m = re.findall(r'([\w./\\-]+\.json)', t)
        if m:
            try:
                res = CORE.run_backtest_on_spec(m[-1])  # type: ignore
                return json.dumps(res, indent=2)
            except Exception:
                traceback.print_exc()
                return "[error] backtest failed"

    # Interpret refine requests when a JSON file is mentioned
    if "refine" in low and ".json" in low:
        m = re.findall(r'([\w./\\-]+\.json)', t)
        if m:
            try:
                res = CORE.quick_refine(m[-1])  # type: ignore
                return json.dumps(res, indent=2)
            except Exception:
                traceback.print_exc()
                return "[error] refine failed"

    # Interpret select_best requests
    if "select_best" in low:
        parts = t.split()
        if parts:
            last = parts[-1]
            try:
                res = CORE.select_best(last)  # type: ignore
                return json.dumps(res, indent=2)
            except Exception:
                traceback.print_exc()
                return "[error] select_best failed"

    # List commands/help
    if low in ("!!tools", "tools", "help"):
        return (
            "I can understand:\n"
            "  • ohlcv <SYMBOL> <TIMEFRAME> <PERIOD>\n"
            "  • backtest <path.json>\n"
            "  • refine <path.json>\n"
            "  • select_best <results.json-or-string>\n"
            "You can type English like: 'run a backtest on strategies/demo_ema_rsi.json'"
        )

    return (
        "I'm listening. Try: 'run a backtest on strategies/demo_ema_rsi.json', "
        "'get ohlcv for AUDCAD 5m 30d', or 'refine strategies/base_ema_rsi.json'."
    )


def repl() -> None:
    """Run a simple interactive chat loop."""
    print("API Agent (chat). Type naturally. Commands still work.")
    print("Type 'quit' to exit.")

    # Determine if we can use OpenAI.  If OPENAI_API_KEY is set and the
    # openai package is available, we'll use the model; otherwise we'll
    # stay offline and use the local router.
    use_llm = False
    client = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            import openai  # type: ignore
            client = openai.OpenAI()
            use_llm = True
            print("[online] OPENAI_API_KEY detected — will use model + tools if available.")
        except Exception:
            print("[offline] openai package not available; using local router.")
            use_llm = False

    while True:
        try:
            msg = input("you> ").strip()
        except EOFError:
            break
        if not msg:
            continue
        if msg.lower() in ("quit", "exit", "bye"):
            print("bye"); break

        # Offline / fallback: local routing
        if not use_llm:
            out = _local_router(msg)
            print(out)
            continue

        # Online: naive chat using model completions (no function-calling)
        try:
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful trading assistant. Keep answers concise. "
                            "If the user mentions backtest, refine, ohlcv, or select_best "
                            "with a JSON path or symbol/timeframe/period, clearly echo the suggested command and result."
                        ),
                    },
                    {"role": "user", "content": msg},
                ],
            )
            print(resp.choices[0].message.content)
        except Exception as e:
            print(f"[online error] {e}. Falling back to local.")
            print(_local_router(msg))


if __name__ == "__main__":
    repl()