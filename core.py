import os
import json
import pandas as pd
import numpy as np
import datetime
from pathlib import Path

"""
Core trading and utility functions with master configuration awareness.

This module now reads a master configuration file (``master_config.json`` by
default, override with ``MASTER_CONFIG_PATH`` environment variable) to
determine default symbols, timeframes, periods, metrics and data/report
directories.  It exposes helper functions ``PATHS()`` and ``DEFAULTS()``
to access these values and functions for modifying the config at runtime.
Additional wrapper functions matching the names declared in the master
configuration are provided (e.g. ``ensure_ohlcv_plus``, ``build_deep_csv``,
``forward_monitor``, ``write_report``, ``organize_project``, ``patch_file``,
``append_line``, ``pip_install``, ``local_shell``).  See the bottom of
this module for their basic implementations.
"""

###############################################################################
# Master configuration handling
###############################################################################


def _resolve_master_config_path() -> Path | None:
    """Resolve the path to the master configuration file.

    Order of resolution:
      1. Environment variable ``MASTER_CONFIG_PATH`` if set and the file exists.
      2. ``master_config.json`` in the current working directory.
      3. ``None`` if no file found, causing defaults to be used.
    """
    env = os.getenv("MASTER_CONFIG_PATH")
    if env:
        p = Path(env)
        if p.exists():
            return p
    p = Path("master_config.json")
    if p.exists():
        return p
    return None


class _MasterConfig:
    """Load, access, modify and persist the master configuration."""

    def __init__(self) -> None:
        self.path: Path | None = _resolve_master_config_path()
        self.data: dict = {}
        self.reload()

    def reload(self) -> None:
        """Reload the configuration from disk, falling back to defaults."""
        if self.path and self.path.exists():
            try:
                raw = self.path.read_text(encoding="utf-8")
                self.data = json.loads(raw)
            except Exception as e:
                # If parse fails record error and fall back to minimal defaults
                self.data = {
                    "paths": {
                        "data_dir": "data",
                        "strategies_dir": "strategies",
                        "reports_dir": "reports",
                    },
                    "tools": {
                        "defaults": {
                            "primary_metric": "profit_factor",
                            "symbol": "AUDCAD",
                            "timeframe": "5m",
                            "period": "30d",
                        }
                    },
                }
                self.data.setdefault("_errors", []).append(
                    f"Failed to parse master config: {e}"
                )
        else:
            # Use base defaults if no file
            self.data = {
                "paths": {
                    "data_dir": "data",
                    "strategies_dir": "strategies",
                    "reports_dir": "reports",
                },
                "tools": {
                    "defaults": {
                        "primary_metric": "profit_factor",
                        "symbol": "AUDCAD",
                        "timeframe": "5m",
                        "period": "30d",
                    }
                },
            }

    def get(self, *keys: str, default: object = None) -> object:
        """Retrieve a nested value or return ``default`` if missing."""
        cur: object = self.data
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def set(self, *keys_and_value: object) -> object:
        """Set a nested configuration key and persist to disk."""
        if len(keys_and_value) < 2:
            raise ValueError("set requires at least one key and a value")
        *keys, value = keys_and_value
        cur = self.data
        for key in keys[:-1]:
            if key not in cur or not isinstance(cur[key], dict):
                cur[key] = {}
            cur = cur[key]
        cur[keys[-1]] = value
        # Persist if we have a valid path
        if self.path:
            try:
                self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            except Exception:
                pass
        return value


# Global config instance
_MASTER = _MasterConfig()


def PATHS() -> dict:
    """Return canonical directories (data/strategies/reports) from the config."""
    return {
        "data_dir": _MASTER.get("paths", "data_dir", default="data"),
        "strategies_dir": _MASTER.get("paths", "strategies_dir", default="strategies"),
        "reports_dir": _MASTER.get("paths", "reports_dir", default="reports"),
    }


def DEFAULTS() -> dict:
    """Return default trading parameters from the config."""
    return {
        "primary_metric": _MASTER.get("tools", "defaults", "primary_metric", default="profit_factor"),
        "symbol": _MASTER.get("tools", "defaults", "symbol", default="AUDCAD"),
        "timeframe": _MASTER.get("tools", "defaults", "timeframe", default="5m"),
        "period": _MASTER.get("tools", "defaults", "period", default="30d"),
    }


###############################################################################
# Utility functions matching tool names in the config
###############################################################################

def get_master_config() -> dict:
    """Return the current in‑memory master configuration."""
    return _MASTER.data


def set_master_config(new_obj: dict) -> bool:
    """Replace the entire configuration with ``new_obj`` and persist it."""
    if not isinstance(new_obj, dict):
        raise ValueError("set_master_config expects a dict")
    _MASTER.data = new_obj
    # Persist to disk if we know where to write
    if _MASTER.path:
        try:
            _MASTER.path.write_text(json.dumps(new_obj, indent=2), encoding="utf-8")
        except Exception:
            pass
    return True


def reload_master_config() -> bool:
    """Reload the configuration from disk (discarding unsaved changes)."""
    _MASTER.reload()
    return True


def repair_master_config() -> bool:
    """Ensure essential keys exist in the configuration and persist it."""
    # Guarantee presence of minimal keys
    _MASTER.data.setdefault("paths", {}).setdefault("data_dir", "data")
    _MASTER.data.setdefault("paths", {}).setdefault("strategies_dir", "strategies")
    _MASTER.data.setdefault("paths", {}).setdefault("reports_dir", "reports")
    _MASTER.data.setdefault("tools", {}).setdefault("defaults", {}).setdefault("primary_metric", "profit_factor")
    _MASTER.data.setdefault("tools", {}).setdefault("defaults", {}).setdefault("symbol", "AUDCAD")
    _MASTER.data.setdefault("tools", {}).setdefault("defaults", {}).setdefault("timeframe", "5m")
    _MASTER.data.setdefault("tools", {}).setdefault("defaults", {}).setdefault("period", "30d")
    # Persist
    if _MASTER.path:
        try:
            _MASTER.path.write_text(json.dumps(_MASTER.data, indent=2), encoding="utf-8")
        except Exception:
            pass
    return True

try:
    import yfinance as yf
except Exception:
    yf = None


def ensure_ohlcv(symbol: str | None = None,
                 timeframe: str | None = None,
                 period: str | None = None,
                 auto_adjust: bool = True) -> dict:
    """Fetch OHLCV via yfinance and write a CSV into the configured data directory.

    Any parameter may be omitted; missing values fall back to the defaults
    defined in the master configuration.  The output CSV is named
    ``<symbol>_<timeframe>_<period>.csv`` and placed inside the ``data_dir``
    specified in the configuration.  Multiple ticker variants are tried
    when downloading data.

    Raises:
        RuntimeError if ``yfinance`` is unavailable or no data could be
        retrieved.
    """
    if yf is None:
        raise RuntimeError("yfinance not available in this environment")
    d = DEFAULTS()
    symbol = symbol or d["symbol"]
    timeframe = timeframe or d["timeframe"]
    period = period or d["period"]
    safe_symbol = str(symbol).replace('/', '_')
    outdir = PATHS()["data_dir"]
    os.makedirs(outdir, exist_ok=True)
    fname = f"{safe_symbol}_{timeframe}_{period}.csv"
    outpath = os.path.join(outdir, fname)
    candidates = [str(symbol)]
    if not str(symbol).endswith("=X"):
        candidates.append(str(symbol) + "=X")
    candidates.append(str(symbol).replace("/", "").upper())
    candidates.append(str(symbol).replace("/", "").upper() + "=X")
    last_err: str | None = None
    for tick in candidates:
        try:
            df = yf.download(tick, period=period, interval=timeframe, progress=False, auto_adjust=auto_adjust)
            if df is None or df.empty:
                last_err = f"No data for {tick}"
                continue
            df.index = pd.to_datetime(df.index)
            df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            }, inplace=True)
            cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df[cols]
            df.to_csv(outpath)
            return {'path': outpath, 'rows': len(df), 'symbol_used': tick}
        except Exception as e:
            last_err = repr(e)
            continue
    raise RuntimeError(f"Failed to fetch OHLCV for {symbol}. Tried {candidates}. Last error: {last_err}")

# -----------------------------------------------------------------------------
# Extended OHLCV and CSV helpers
# -----------------------------------------------------------------------------

def ensure_ohlcv_plus(symbol: str | None = None,
                      timeframe: str | None = None,
                      period: str | None = None,
                      indicators: list[str] | None = None,
                      strict: bool = False) -> dict:
    """Fetch OHLCV and optionally add technical indicators.

    This function simply delegates to ``ensure_ohlcv`` at present but exists
    to satisfy the tool interface described by the master configuration.
    Future versions may enrich the resulting CSV with EMA/SMA/RSI/ATR/etc.

    Args:
        symbol: Currency or stock symbol; falls back to config default.
        timeframe: Bar interval; falls back to config default.
        period: Historical period; falls back to config default.
        indicators: List of indicator names to compute (ignored currently).
        strict: If true, raise on unknown indicator names (unused currently).

    Returns:
        A dict with keys ``path``, ``rows`` and ``symbol_used`` analogous to
        ``ensure_ohlcv``.
    """
    # At the moment we do not compute indicators; simply fetch base data
    return ensure_ohlcv(symbol=symbol, timeframe=timeframe, period=period)


def build_deep_csv(symbol: str | None = None,
                   timeframe: str | None = None,
                   period: str | None = None) -> dict:
    """Create a "deep" CSV, potentially including indicators and pre‑processing.

    Currently this is just a wrapper around ``ensure_ohlcv_plus``.  A deep
    CSV may include additional computed features in the future.
    """
    return ensure_ohlcv_plus(symbol=symbol, timeframe=timeframe, period=period)


def validate_csv(path: str) -> dict:
    """Validate that a CSV contains required OHLCV columns.

    Loads the CSV located at ``path`` and checks for the presence of the
    columns ``datetime``, ``open``, ``high``, ``low``, ``close`` and
    ``volume`` (case‑insensitive).  If any are missing returns a
    dict with ``ok=False`` and a list of missing column names.  Otherwise
    returns ``ok=True`` and the number of rows.
    """
    if not os.path.exists(path):
        return {'ok': False, 'error': 'file not found'}
    df = pd.read_csv(path)
    expected = {'datetime', 'open', 'high', 'low', 'close', 'volume'}
    cols = {c.lower() for c in df.columns}
    missing = expected - cols
    if missing:
        return {'ok': False, 'missing': list(missing)}
    return {'ok': True, 'rows': len(df)}


def _simple_ema_strategy(df, params):
    # coerce numeric columns for safety
    df = df.copy()
    for col in ['open','high','low','close','volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # drop rows without close price
    if 'close' in df.columns:
        df = df.dropna(subset=['close'])

    fast = int(params.get('ema_fast', 10))
    slow = int(params.get('ema_slow', 20))
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['pos'] = np.where(df['ema_fast'] > df['ema_slow'], 1, 0)
    df['signal'] = df['pos'].diff()
    cash = 10000.0
    position = 0.0
    entry_price = 0.0
    trades = []
    equity = []
    eq = cash
    for i, row in df.iterrows():
        price = row['close']
        if row['signal'] == 1 and position == 0:
            position = eq / price
            entry_price = price
            eq = position * price
            trades.append({'type': 'buy', 'price': price, 'time': str(i)})
        elif row['signal'] == -1 and position > 0:
            pnl = position * price - position * entry_price
            cash = position * price
            trades.append({'type': 'sell', 'price': price, 'time': str(i), 'pnl': pnl})
            position = 0
            entry_price = 0
            eq = cash
        else:
            if position > 0:
                eq = position * price
            else:
                eq = cash
        equity.append(eq)
    if position > 0:
        last_price = df['close'].iloc[-1]
        pnl = position * last_price - position * entry_price
        trades.append({'type': 'sell', 'price': last_price, 'time': str(df.index[-1]), 'pnl': pnl})
        cash = position * last_price
        position = 0
        eq = cash
        if equity:
            equity[-1] = eq
    profits = [t['pnl'] for t in trades if 'pnl' in t and t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if 'pnl' in t and t['pnl'] < 0]
    total_profit = sum(profits) if profits else 0.0
    total_loss = sum(losses) if losses else 0.0
    gross_loss = abs(total_loss)
    profit_factor = (total_profit / gross_loss) if gross_loss > 0 else float('inf')
    ntrades = len([t for t in trades if 'pnl' in t])
    wins = len([p for p in profits if p > 0])
    winrate = (wins / ntrades) if ntrades > 0 else 0.0
    expectancy = ((total_profit - gross_loss) / ntrades) if ntrades > 0 else 0.0
    peak = np.maximum.accumulate(equity) if equity else np.array([0])
    drawdowns = (peak - equity) / peak if len(peak) > 0 else np.array([0])
    max_dd = float(np.max(drawdowns)) * 100 if len(drawdowns) > 0 else 0.0
    total_return = (equity[-1] / equity[0] - 1) if len(equity) > 0 and equity[0] > 0 else 0.0
    rtot = total_return
    ravg = (total_return / ntrades) if ntrades > 0 else 0.0
    result = {
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'winrate': winrate,
        'rtot': rtot,
        'ravg': ravg,
        'max_drawdown_pct': max_dd,
        'ntrades': ntrades,
        'trades': trades
    }
    return result


def run_backtest_on_spec(spec_path):
    if not os.path.exists(spec_path):
        return {'ok': False, 'error': 'spec not found', 'path': spec_path}
    with open(spec_path, 'r', encoding='utf-8') as f:
        try:
            spec = json.load(f)
        except Exception as e:
            return {'ok': False, 'error': f'json load error: {e}'}
    symbol = spec.get('symbol', 'AUDCAD')
    timeframe = spec.get('timeframe', '5m')
    period = spec.get('period', '60d')
    data_res = ensure_ohlcv(symbol, timeframe, period)
    df = pd.read_csv(data_res['path'], index_col=0, parse_dates=True)
    # ensure 'close' exists and coerce to numeric
    if 'close' not in df.columns:
        return {'ok': False, 'error': 'data missing required columns', 'data_csv': data_res.get('path')}
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])
    params = spec.get('strategy', {}).get('params', spec.get('params', {}))
    try:
        res = _simple_ema_strategy(df, params)
    except Exception as e:
        # return diagnostic info
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        head = df.head().to_dict()
        return {'ok': False, 'error': str(e), 'spec': spec_path, 'data_csv': data_res.get('path'), 'dtypes': dtypes, 'head': head}
    res['ok'] = True
    res['spec'] = spec_path
    res['data_csv'] = data_res['path']
    return res


def quick_refine(spec_path, max_runs=6):
    if not os.path.exists(spec_path):
        return {'ok': False, 'error': 'spec not found'}
    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)
    opt = spec.get('optimization', {})
    params = opt.get('params', {})
    runs = []
    grids = {}
    for k, v in params.items():
        if isinstance(v, dict) and 'min' in v and 'max' in v and 'step' in v:
            grids[k] = list(range(v['min'], v['max'] + 1, v['step']))
        else:
            grids[k] = [v]
    import itertools
    keys = list(grids.keys())
    combos = list(itertools.product(*(grids[k] for k in keys)))
    combos = combos[:max_runs]
    for combo in combos:
        p = dict(zip(keys, combo))
        tname = spec_path + '.run.' + '_'.join([f"{k}{v}" for k, v in p.items()]) + '.json'
        spec2 = dict(spec)
        if 'strategy' not in spec2:
            spec2['strategy'] = {}
        spec2['strategy']['params'] = p
        with open(tname, 'w', encoding='utf-8') as fo:
            json.dump(spec2, fo)
        res = run_backtest_on_spec(tname)
        res['run_params'] = p
        runs.append(res)
    return {'ok': True, 'runs': runs}


def select_best(results_json, primary: str | None = None):
    if isinstance(results_json, str):
        try:
            data = json.loads(results_json)
        except Exception:
            if os.path.exists(results_json):
                with open(results_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                return {'ok': False, 'error': 'invalid input to select_best'}
    else:
        data = results_json
    # Use config default metric if primary not provided
    if not primary:
        primary = DEFAULTS()["primary_metric"]
    runs = data.get('runs', [])
    if not runs:
        return {'ok': False, 'error': 'no runs'}
    best = None
    best_val = None
    for r in runs:
        val = r.get(primary, None)
        if val is None:
            val = r.get('result', {}).get(primary) if isinstance(r.get('result'), dict) else None
        if val is None:
            continue
        try:
            valf = float(val)
        except Exception:
            continue
        if best is None or valf > best_val:
            best = r
            best_val = valf
    return {'ok': True, 'best': best}

# -----------------------------------------------------------------------------
# Additional tool implementations referenced by the master configuration
# -----------------------------------------------------------------------------

def forward_monitor(spec_path: str, period: str | None = None, refresh_secs: int = 300, cycles: int = 6) -> dict:
    """Periodically re-fetch data and re-run a strategy spec.

    This function runs ``run_backtest_on_spec`` on ``spec_path`` repeatedly
    ``cycles`` times, waiting ``refresh_secs`` between runs.  The optional
    ``period`` parameter overrides the period in the spec for each run.  If
    any run fails the error is recorded and the loop continues.  Returns a
    dict containing a list of runs with timestamps.
    """
    history: list = []
    for i in range(int(cycles)):
        try:
            # If a period override is provided, load spec and set period
            spath = spec_path
            if period:
                # Load spec and override period
                with open(spec_path, 'r', encoding='utf-8') as f:
                    spec = json.load(f)
                spec['period'] = period
                tmp = f"{spec_path}.tmp_monitor.json"
                with open(tmp, 'w', encoding='utf-8') as fo:
                    json.dump(spec, fo)
                spath = tmp
            res = run_backtest_on_spec(spath)
            history.append({'cycle': i + 1, 'timestamp': datetime.datetime.utcnow().isoformat() + 'Z', 'result': res})
        except Exception as e:
            history.append({'cycle': i + 1, 'timestamp': datetime.datetime.utcnow().isoformat() + 'Z', 'error': str(e)})
        # Sleep between cycles except after last
        if i < cycles - 1:
            try:
                import time
                time.sleep(refresh_secs)
            except Exception:
                pass
    return {'ok': True, 'history': history}


def write_report(content: str, name: str | None = None) -> dict:
    """Write a text report to the reports directory.

    Creates the reports directory if necessary.  The file name defaults
    to ``report_<timestamp>.txt`` but can be overridden via ``name``.
    Returns a dict with the written path.
    """
    reports_dir = PATHS()["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    if not name:
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name = f"report_{ts}.txt"
    out = os.path.join(reports_dir, name)
    try:
        with open(out, 'w', encoding='utf-8') as f:
            f.write(content)
        return {'ok': True, 'report_path': out}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def organize_project(dry_run: bool = True) -> dict:
    """Ensure canonical directories exist and optionally move stray files.

    When ``dry_run`` is true this function only reports which files
    _would_ be moved.  When false it actually moves files found in the
    project root into ``strategies_dir`` (for ``*.json``), ``data_dir``
    (for ``*.csv``) and ``reports_dir`` (for ``*.md`` except ``README.md``).
    Returns a dict describing the moves.
    """
    moves = []
    paths = PATHS()
    strat_dir = Path(paths["strategies_dir"])
    data_dir = Path(paths["data_dir"])
    reports_dir = Path(paths["reports_dir"])
    strat_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    root = Path('.')
    for f in root.glob("*.json"):
        dst = strat_dir / f.name
        if f != dst:
            moves.append({'from': str(f), 'to': str(dst)})
            if not dry_run:
                shutil.move(str(f), str(dst))
    for f in root.glob("*.csv"):
        dst = data_dir / f.name
        if f != dst:
            moves.append({'from': str(f), 'to': str(dst)})
            if not dry_run:
                shutil.move(str(f), str(dst))
    for f in root.glob("*.md"):
        if f.name.lower() == 'readme.md':
            continue
        dst = reports_dir / f.name
        if f != dst:
            moves.append({'from': str(f), 'to': str(dst)})
            if not dry_run:
                shutil.move(str(f), str(dst))
    return {'dry_run': dry_run, 'moves': moves}


def patch_file(path: str, find: str, replace: str, backup: bool = True) -> dict:
    """Perform a string replacement within a text file.

    If ``find`` does not occur in the file the function returns with
    ``replacements`` set to 0.  When replacements occur and ``backup`` is
    true, a ``.bak`` file is written before modifying the original.
    """
    p = Path(path)
    if not p.exists():
        return {'ok': False, 'error': 'file not found'}
    text = p.read_text(encoding='utf-8')
    count = text.count(find)
    if count == 0:
        return {'ok': True, 'replacements': 0}
    backup_path = ''
    if backup:
        bak = p.with_suffix(p.suffix + '.bak')
        bak.write_text(text, encoding='utf-8')
        backup_path = str(bak)
    new_text = text.replace(find, replace)
    p.write_text(new_text, encoding='utf-8')
    return {'ok': True, 'replacements': count, 'backup': backup_path}


def append_line(path: str, line: str) -> dict:
    """Append a line to a file, creating it if it doesn't exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(line + '\n')
    return {'ok': True, 'path': str(p)}


def pip_install(pkg: str) -> str:
    """Install (or upgrade) a Python package using pip."""
    import subprocess, sys
    out = subprocess.run([sys.executable, '-m', 'pip', 'install', '-U', pkg], capture_output=True, text=True)
    return out.stdout + out.stderr


def local_shell(cmd: str) -> str:
    """Execute a shell command and return combined stdout+stderr."""
    import subprocess
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (out.stdout or '') + (out.stderr or '')
# --- Minimal builder for the console agent ---

import shlex
from types import SimpleNamespace

def build_main_agent(cfg):
    """
    Returns a simple agent object with:
      - .tools (for !!tools)
      - .run(text) that understands a few console commands
    Commands:
      bt|backtest <spec.json>
      refine <spec.json>
      best|select_best <results.json-or-string>
      ohlcv <symbol> <timeframe> <period>
    """
    class _ConsoleAgent:
        def __init__(self, cfg):
            self.cfg = cfg
            self.tools = [
                SimpleNamespace(name="backtest"),
                SimpleNamespace(name="refine"),
                SimpleNamespace(name="select_best"),
                SimpleNamespace(name="ohlcv"),
            ]

        def run(self, text: str):
            """
            Parse and execute a single console command. Supports both trading
            commands (backtest, refine, select_best, ohlcv) and a few utility
            commands ("!!tools", "!!reload", "!shell", "!py"). Returns a
            string that should be printed by the caller.
            """
            # Normalize to a stripped string to simplify checks
            text = (text or "").strip()
            if not text:
                return "Say: backtest <spec.json> | refine <spec.json> | select_best <json> | ohlcv <sym> <tf> <period>"

            # --- Utility commands ---
            # List available commands (trading ops + utilities)
            if text in ("!!tools", "tools"):
                return ("Commands:\n"
                        "  backtest <spec.json>\n"
                        "  refine <spec.json>\n"
                        "  select_best <results.json-or-string>\n"
                        "  ohlcv <symbol> <timeframe> <period>\n"
                        "Utilities:\n"
                        "  !!reload\n"
                        "  !py <code>\n"
                        "  !shell <cmd>")

            # Reload hint. In this simple console agent we can't really
            # hot‑reload core code safely, so advise a restart.
            if text in ("!!reload", "reload"):
                return "[reload] Use Ctrl+C then re-run main_agent_entry.py"

            # Execute a shell command
            if text.startswith("!shell"):
                import subprocess
                cmd = text.partition("!shell")[2].strip()
                if not cmd:
                    return "Usage: !shell <command>"
                try:
                    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    return (out.stdout or "") + (out.stderr or "")
                except Exception as e:
                    return f"shell error: {e}"

            # Execute inline Python code
            if text.startswith("!py"):
                code = text.partition("!py")[2]
                # Provide only core functions in a limited namespace
                ns = {}
                ns["ensure_ohlcv"] = ensure_ohlcv
                ns["run_backtest_on_spec"] = run_backtest_on_spec
                ns["quick_refine"] = quick_refine
                ns["select_best"] = select_best
                try:
                    exec(code, ns, ns)
                    return "[py done]"
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"[py error] {e}"

            # --- Trading commands ---
            try:
                args = shlex.split(text)
            except Exception:
                args = text.split()

            if not args:
                return "Say: backtest <spec.json> | refine <spec.json> | select_best <json> | ohlcv <sym> <tf> <period>"

            cmd = args[0].lower()

            if cmd in ("bt", "backtest"):
                if len(args) < 2:
                    return "usage: backtest <spec.json>"
                res = run_backtest_on_spec(args[1])
                return json.dumps(res, indent=2)

            if cmd == "refine":
                if len(args) < 2:
                    return "usage: refine <spec.json>"
                res = quick_refine(args[1])
                return json.dumps(res, indent=2)

            if cmd in ("best", "select_best"):
                if len(args) < 2:
                    return "usage: select_best <results.json-or-string>"
                res = select_best(args[1])
                return json.dumps(res, indent=2)

            if cmd == "ohlcv":
                if len(args) < 4:
                    return "usage: ohlcv <symbol> <timeframe> <period>   e.g. ohlcv SPY 1d 60d"
                sym, tf, period = args[1], args[2], args[3]
                res = ensure_ohlcv(sym, tf, period)
                return json.dumps(res, indent=2)

            return f"Unknown command: {cmd}"

        def chat(self, text: str):
            """
            Chat-first interface:
            - If the text contains keywords that map to existing commands,
              route to self.run() with the appropriate command string.
            - Otherwise return a friendly help message.
            """
            t = (text or "").strip()
            if not t:
                return ""
            low = t.lower()
            # If user mentions backtest and a JSON file, run backtest
            if "backtest" in low and ".json" in low:
                try:
                    import re
                    matches = re.findall(r'([\w./\\-]+\.json)', t)
                    if matches:
                        try:
                            return self.run(f"backtest {matches[-1]}")
                        except Exception as e:
                            return str(e)
                except Exception:
                    pass
            # Refine
            if "refine" in low and ".json" in low:
                try:
                    import re
                    matches = re.findall(r'([\w./\\-]+\.json)', t)
                    if matches:
                        try:
                            return self.run(f"refine {matches[-1]}")
                        except Exception as e:
                            return str(e)
                except Exception:
                    pass
            # select_best
            if "select_best" in low:
                parts = t.split()
                if parts:
                    last = parts[-1]
                    try:
                        return self.run(f"select_best {last}")
                    except Exception as e:
                        return str(e)
            # ohlcv pattern: user might say "ohlcv AUDCAD 5m 30d"
            if "ohlcv" in low:
                parts = t.split()
                # If there are at least 4 parts, treat last three as args
                if len(parts) >= 4:
                    sym, tf, period = parts[-3], parts[-2], parts[-1]
                    try:
                        return self.run(f"ohlcv {sym} {tf} {period}")
                    except Exception as e:
                        return str(e)
            # If we didn't route to a command, return a help message
            return ("I’m in chat mode. Here's what I can do:\n"
                    "  • backtest <spec.json>\n"
                    "  • refine <spec.json>\n"
                    "  • select_best <results.json-or-string>\n"
                    "  • ohlcv <symbol> <timeframe> <period>\n"
                    "You can also run !!tools to see available commands.")

    return _ConsoleAgent(cfg)

