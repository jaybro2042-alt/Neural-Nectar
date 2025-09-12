"""
agent_chat.py — English-first local agent with filesystem + shell + trading tools

This script combines your trading tools (via core.py) with common local file and shell
operations in a single chat-style interface.  It prioritizes local actions: if you
ask to list a directory, read or write a file, copy or move something, run a
shell command, or use trading functions like backtest/refine/ohlcv/select_best,
it will execute those directly.  Only if a request does not match any known
pattern does it optionally query the OpenAI API, provided an API key is set.

Usage:
    py -3 .\agent_chat.py

Examples:
    ls C:\\              # list contents of C:\
    read C:\\Windows\\System32\\drivers\\etc\\hosts
    write C:\\temp\\note.txt :: Hello world
    copy C:\\a.txt -> C:\\b.txt
    move C:\\dir -> D:\\dir
    rm C:\\temp\\note.txt
    mkdir C:\\temp\\logs
    shell echo hello
    backtest strategies\\base_ema_rsi.json
    ohlcv AUDCAD 5m 30d

If you set the environment variable OPENAI_API_KEY and have the openai package
installed, the agent will use the model for general chat queries that don't
match any local operation.  Otherwise it stays offline.
"""

import os
import re
import json
import shutil
import subprocess
import traceback
from pathlib import Path

# Optional online chat via OpenAI; only used when a message doesn't match
# any local command.  We detect the API key at runtime.
def _try_llm_chat(msg: str) -> str | None:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return None
    try:
        import openai  # type: ignore
        client = openai.OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise and helpful assistant. When the user asks for "
                        "file or shell actions, respond that you have executed them locally."
                    ),
                },
                {"role": "user", "content": msg},
            ],
        )
        return resp.choices[0].message.content or ""
    except Exception:
        # If the API call fails, fall back to local responses
        return None

# Import your trading core; it should define ensure_ohlcv, run_backtest_on_spec,
# quick_refine, select_best.  If import fails, trading functions will
# silently be unavailable.
try:
    import core as CORE  # type: ignore
except Exception:
    CORE = None
    traceback.print_exc()

# ----- Local filesystem helpers -----
def ls(path: str) -> str:
    p = Path(path or ".")
    if not p.exists():
        return f"[ls] not found: {p}"
    rows = []
    for entry in sorted(p.iterdir()):
        typ = "<DIR>" if entry.is_dir() else f"{entry.stat().st_size}B"
        rows.append(f"{typ:>8}  {entry.name}")
    return "\n".join(rows) or "(empty)"

def read(path: str, max_bytes: int = 200_000) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return f"[read] file not found: {p}"
    data = p.read_bytes()
    if len(data) > max_bytes:
        head = data[:max_bytes].decode(errors="replace")
        return f"[read] too large ({len(data)} bytes). Showing first {max_bytes} bytes:\n{head}"
    return data.decode(errors="replace")

def write(path: str, content: str, mkdirs: bool = True) -> str:
    p = Path(path)
    if mkdirs:
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"[write] ok → {p}"

def append(path: str, content: str, mkdirs: bool = True) -> str:
    p = Path(path)
    if mkdirs:
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(content)
    return f"[append] ok → {p}"

def copy(src: str, dst: str) -> str:
    s, d = Path(src), Path(dst)
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s, d)
    return f"[copy] {s} → {d}"

def move(src: str, dst: str) -> str:
    s, d = Path(src), Path(dst)
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(d))
    return f"[move] {s} → {d}"

def rm(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(p)
        return f"[rm] dir removed → {p}"
    if p.exists():
        p.unlink()
        return f"[rm] file removed → {p}"
    return f"[rm] not found: {p}"

def mkdir(path: str) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return f"[mkdir] ok → {p}"

def shell(cmd: str) -> str:
    if not cmd.strip():
        return "Usage: shell <command>"
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (out.stdout or "") + (out.stderr or "")

# ----- Local intent router for files + trading -----
def local_router(text: str) -> str | None:
    t = (text or "").strip()
    low = t.lower()

    # Filesystem commands
    m = re.match(r"^(?:ls|list)\s+(.+)$", low)
    if m:
        return ls(m.group(1))
    if low.startswith("read "):
        return read(t[5:].strip())
    if low.startswith("write "):
        parts = t.split("::", 1)
        if len(parts) == 2:
            return write(parts[0].split()[1].strip(), parts[1])
        return "Usage: write <path> :: <content>"
    if low.startswith("append "):
        parts = t.split("::", 1)
        if len(parts) == 2:
            return append(parts[0].split()[1].strip(), parts[1])
        return "Usage: append <path> :: <content>"
    if low.startswith("copy "):
        parts = re.split(r"\s*->\s*", t[5:].strip(), 1)
        if len(parts) == 2:
            return copy(parts[0], parts[1])
        return "Usage: copy <src> -> <dst>"
    if low.startswith("move "):
        parts = re.split(r"\s*->\s*", t[5:].strip(), 1)
        if len(parts) == 2:
            return move(parts[0], parts[1])
        return "Usage: move <src> -> <dst>"
    if low.startswith("rm "):
        return rm(t[3:].strip())
    if low.startswith("mkdir "):
        return mkdir(t[6:].strip())
    if low.startswith("shell "):
        return shell(t[6:].strip())

    # Trading commands via CORE
    if CORE:
        if "ohlcv" in low:
            parts = t.split()
            if len(parts) >= 4:
                sym, tf, period = parts[-3], parts[-2], parts[-1]
                try:
                    res = CORE.ensure_ohlcv(sym, tf, period)  # type: ignore
                    return json.dumps(res, indent=2)
                except Exception:
                    traceback.print_exc()
                    return "[ohlcv] error (see traceback)"
        if "backtest" in low and ".json" in low:
            m = re.findall(r'([\w./\\-]+\.json)', t)
            if m:
                try:
                    res = CORE.run_backtest_on_spec(m[-1])  # type: ignore
                    return json.dumps(res, indent=2)
                except Exception:
                    traceback.print_exc()
                    return "[backtest] error (see traceback)"
        if "refine" in low and ".json" in low:
            m = re.findall(r'([\w./\\-]+\.json)', t)
            if m:
                try:
                    res = CORE.quick_refine(m[-1])  # type: ignore
                    return json.dumps(res, indent=2)
                except Exception:
                    traceback.print_exc()
                    return "[refine] error (see traceback)"
        if "select_best" in low:
            last = t.split()[-1]
            try:
                res = CORE.select_best(last)  # type: ignore
                return json.dumps(res, indent=2)
            except Exception:
                traceback.print_exc()
                return "[select_best] error (see traceback)"

    return None  # Let the LLM handle anything else

# ----- Chat loop -----
def main() -> None:
    print("Agent ready. Talk in plain English. Type 'quit' to exit.")
    print("Examples:")
    print("  ls C:\\        |  read C:\\path\\file.txt")
    print("  write C:\\tmp\\note.txt :: hello world")
    print("  copy C:\\a.txt -> C:\\b.txt   |  move C:\\a -> D:\\a")
    print("  rm C:\\tmp\\note.txt            |  mkdir C:\\tmp\\logs")
    print("  shell echo ok")
    print("  backtest strategies\\demo_ema_rsi.json")
    print("  ohlcv AUDCAD 5m 30d")

    while True:
        try:
            msg = input("you> ").strip()
        except EOFError:
            break
        if not msg:
            continue
        if msg.lower() in ("quit", "exit", "bye"):
            print("bye")
            break

        # 1) Local actions always take priority
        out = local_router(msg)
        if out is not None:
            print(out)
            continue

        # 2) Fallback: call the model if configured
        llm = _try_llm_chat(msg)
        if llm:
            print(llm)
            continue

        # 3) If nothing else matched
        print("I'm listening. Try local commands like ls, read, copy, backtest, etc.")

if __name__ == "__main__":
    main()