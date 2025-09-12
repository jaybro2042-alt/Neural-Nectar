import os, sys, json, math, traceback, subprocess, time, itertools, random
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# -------------------- cfg --------------------
def load_cfg():
    with open(ROOT/"master_config.json","r",encoding="utf-8") as f:
        return json.load(f)
CFG = load_cfg()

# -------------------- utils ------------------
def ensure_dirs(*paths):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def interval_to_yf(itv): return {"1h":"60m"}.get(itv, itv)
def pair_to_yf(sym):
    s = sym.replace("/","").upper()
    return s if s.endswith("=X") else s+"=X"

def run_local_shell(cmd):
    try:
        if os.name == "nt":
            p = subprocess.run(["powershell","-NoProfile","-Command",cmd], capture_output=True, text=True)
        else:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        out, err = (p.stdout or "").strip(), (p.stderr or "").strip()
        return json.dumps({"ok": True, "stdout": out, "stderr": err, "rc": p.returncode})
    except Exception as e:
        return json.dumps({"ok": False, "error": repr(e)})

def python_exec(code):
    import io, contextlib, math
    safe_globals = {"__builtins__": {"range":range,"len":len,"min":min,"max":max,"sum":sum,"abs":abs,"round":round,"sorted":sorted,"print":print}, "math":math}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code,"<user>","exec"), safe_globals, {})
        return json.dumps({"ok": True, "stdout": buf.getvalue()})
    except Exception as e:
        return json.dumps({"ok": False, "error": repr(e), "stdout": buf.getvalue()})

# -------------------- indicators --------------------
def compute_indicators(df, p):
    import numpy as np
    # EMA/SMA
    if "ema_fast" in p: df["EMA_F"] = df["Close"].ewm(span=int(p["ema_fast"]), adjust=False).mean()
    if "ema_slow" in p: df["EMA_S"] = df["Close"].ewm(span=int(p["ema_slow"]), adjust=False).mean()
    if "sma" in p:      df["SMA"]   = df["Close"].rolling(int(p["sma"])).mean()

    # MACD
    if {"macd_fast","macd_slow","macd_signal"} <= p.keys():
        fast = df["Close"].ewm(span=int(p["macd_fast"]), adjust=False).mean()
        slow = df["Close"].ewm(span=int(p["macd_slow"]), adjust=False).mean()
        macd = fast - slow
        signal = macd.ewm(span=int(p["macd_signal"]), adjust=False).mean()
        df["MACD"], df["MACD_SIG"], df["MACD_H"] = macd, signal, macd - signal

    # Bollinger Bands
    if {"bb_period","bb_dev"} <= p.keys():
        r = int(p["bb_period"]); dev = float(p["bb_dev"])
        ma = df["Close"].rolling(r).mean()
        sd = df["Close"].rolling(r).std(ddof=0)
        df["BB_MID"], df["BB_UP"], df["BB_DN"] = ma, ma + dev*sd, ma - dev*sd

    # ATR
    if "atr_period" in p:
        n = int(p["atr_period"])
        high, low, close = df["High"], df["Low"], df["Close"]
        prev_close = close.shift(1)
        tr = (high - low).abs()
        tr = tr.combine((high - prev_close).abs(), max)
        tr = tr.combine((low - prev_close).abs(), max)
        df["ATR"] = tr.ewm(alpha=1/n, adjust=False).mean()

    # RSI
    if "rsi_length" in p:
        n = int(p["rsi_length"])
        delta = df["Close"].diff()
        up = delta.clip(lower=0.0); down = -delta.clip(upper=0.0)
        roll_up = up.ewm(alpha=1/n, adjust=False).mean()
        roll_down = down.ewm(alpha=1/n, adjust=False).mean()
        rs = roll_up / (roll_down.replace(0, np.nan))
        df["RSI"] = (100 - (100/(1+rs))).fillna(50)

    # StochRSI
    if {"stoch_len","stoch_k","stoch_d","rsi_length"} <= p.keys():
        rsi = df.get("RSI")
        if rsi is None:
            temp = p.copy()
            temp["rsi_length"] = int(p["rsi_length"])
            compute_indicators(df, temp); rsi = df["RSI"]
        n = int(p["stoch_len"])
        min_rsi = rsi.rolling(n).min()
        max_rsi = rsi.rolling(n).max()
        stoch = 100 * (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)
        k = int(p["stoch_k"]); d = int(p["stoch_d"])
        df["StochRSI_K"] = stoch.rolling(k).mean()
        df["StochRSI_D"] = df["StochRSI_K"].rolling(d).mean()

    # ADX
    if "adx_period" in p:
        n = int(p["adx_period"])
        up_move = df["High"].diff()
        down_move = -df["Low"].diff()
        plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0)
        minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0)
        tr1 = (df["High"]-df["Low"]).abs()
        tr2 = (df["High"]-df["Close"].shift(1)).abs()
        tr3 = (df["Low"]-df["Close"].shift(1)).abs()
        tr = tr1.combine(tr2, max).combine(tr3, max)
        atr = tr.ewm(alpha=1/n, adjust=False).mean()
        pdi = 100 * (plus_dm.ewm(alpha=1/n, adjust=False).mean()/atr)
        mdi = 100 * (minus_dm.ewm(alpha=1/n, adjust=False).mean()/atr)
        dx = ( (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-9) ) * 100
        df["ADX"] = dx.ewm(alpha=1/n, adjust=False).mean()

    # OBV
    if "obv" in p and p["obv"]:
        obv = ( (df["Close"].diff() > 0).astype(int) - (df["Close"].diff() < 0).astype(int) ) * df["Volume"].fillna(0)
        df["OBV"] = obv.cumsum()

    return df

# -------------------- data fetch --------------------
def fetch_bars(symbol, interval, period):
    import pandas as pd, yfinance as yf
    yf_symbol = pair_to_yf(symbol)
    yf_interval = interval_to_yf(interval)
    df = yf.download(tickers=yf_symbol, interval=yf_interval, period=period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol} ({yf_symbol}, {interval}, {period})")
    df = df.rename(columns=str.title).dropna().copy()
    return df

# -------------------- strategies --------------------
def strat_ema_rsi_band(df, params):
    rsi_lo = float(params.get("rsi_entry_low", 45))
    rsi_hi = float(params.get("rsi_entry_high", 65))
    df["LongCond"] = (df["EMA_F"] > df["EMA_S"]) & (df["RSI"] >= rsi_lo) & (df["RSI"] <= rsi_hi)
    return df

def strat_macd_bb_trend(df, params):
    # long when MACD>signal and price > mid band; exit if MACD<signal or close<mid
    above_mid = df["Close"] > df["BB_MID"]
    macd_up = df["MACD"] > df["MACD_SIG"]
    df["LongCond"] = above_mid & macd_up
    return df

STRATS = {
    "ema_rsi_band": {
        "needs": ["ema_fast","ema_slow","rsi_length","rsi_entry_low","rsi_entry_high"],
        "func": strat_ema_rsi_band
    },
    "macd_bb_trend": {
        "needs": ["macd_fast","macd_slow","macd_signal","bb_period","bb_dev"],
        "func": strat_macd_bb_trend
    }
}

# -------------------- backtest core --------------------
def run_backtest(df, pos_col="LongCond"):
    df = df.copy()
    df["Pos"] = df[pos_col].shift(1).fillna(False).astype(int)
    df["Ret"] = df["Close"].pct_change().fillna(0.0)
    df["StratRet"] = df["Pos"] * df["Ret"]
    equity = (1 + df["StratRet"]).cumprod()

    # trades list
    edge = df["Pos"].diff().fillna(0)
    entries = df.index[edge == 1]
    exits   = df.index[edge == -1]
    ex_list = list(exits)
    if len(entries) and len(ex_list) < len(entries): ex_list.append(df.index[-1])
    trades = []
    for en, ex in itertools.zip_longest(entries, ex_list):
        if en is None or ex is None or ex <= en: continue
        p_en = float(df.loc[en, "Close"]); p_ex = float(df.loc[ex, "Close"])
        trades.append({"entry": en.isoformat(), "exit": ex.isoformat(), "p_entry": p_en, "p_exit": p_ex, "ret": (p_ex/p_en)-1.0})

    # metrics
    total_ret = float(equity.iloc[-1] - 1.0)
    ann_scale = max(1, (len(df) / 252.0) ** 0.5)
    vol = float(df["StratRet"].std(ddof=0)) or 1e-9
    sharpe_like = float((df["StratRet"].mean() / vol) * ann_scale)
    wins = [t for t in trades if t["ret"] > 0]; losses = [t for t in trades if t["ret"] <= 0]
    win_rate = (len(wins) / len(trades))*100.0 if trades else 0.0
    gp = sum(t["ret"] for t in wins); gl = -sum(t["ret"] for t in losses) if losses else 0.0
    pf = (gp / gl) if gl > 0 else None
    # MDD
    hi, dd = -1e18, 0.0
    for v in equity.values: hi = max(hi, v); dd = max(dd, (hi - v) / hi if hi > 0 else 0.0)

    return {
        "equity": equity,
        "metrics": {
            "total_return_pct": round(total_ret*100,3),
            "sharpe_like": round(sharpe_like,3),
            "win_rate_pct": round(win_rate,2),
            "profit_factor": None if pf is None else round(pf,3),
            "max_drawdown_pct": round(dd*100.0,2),
            "trades": len(trades)
        },
        "trades": trades
    }

# -------------------- pipeline & optimize --------------------
def pipeline_run(over=None):
    over = over or {}
    dfl = CFG.get("tools",{}).get("defaults",{})
    symbol   = over.get("symbol",   dfl.get("symbol","AUDCAD"))
    interval = over.get("timeframe",dfl.get("timeframe","5m"))
    period   = over.get("period",   dfl.get("period","30d"))
    min_bars = int(over.get("min_bars", dfl.get("min_bars", 1200)))

    # choose strategy
    strat_name = over.get("strategy", CFG.get("strategy",{}).get("name","ema_rsi_band"))
    strat_cfg  = CFG.get("strategy",{}).get("params",{}).copy()
    strat_cfg.update(over)  # allow direct overrides

    # fetch
    df = fetch_bars(symbol, interval, period)
    if len(df) < min_bars: pass

    # indicators needed by chosen strat
    needs = set(STRATS[strat_name]["needs"])
    # include common knobs if present
    common = ["sma","atr_period","obv","bb_period","bb_dev","rsi_length","stoch_len","stoch_k","stoch_d","adx_period"]
    want = needs.union([k for k in common if k in strat_cfg])
    params = {k: strat_cfg[k] for k in want if k in strat_cfg}
    df = compute_indicators(df, params)

    # run strategy
    df = STRATS[strat_name]["func"](df, strat_cfg)
    res = run_backtest(df)

    # write artifacts
    ensure_dirs(ROOT/"data", ROOT/"reports")
    csv_name = f"{symbol}_{interval}.csv".replace("/","")
    (ROOT/"data"/csv_name).write_text(df.to_csv(index=True), encoding="utf-8")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rep_json = (ROOT/"reports"/f"{ts}_summary.json").resolve()
    rep_trades = (ROOT/"reports"/f"{ts}_trades.csv").resolve()
    import pandas as pd
    pd.DataFrame(res["trades"]).to_csv(rep_trades, index=False)

    summary = {
        "symbol":symbol, "interval":interval, "period":period, "bars":int(len(df)),
        "strategy": strat_name, "params": params,
        "metrics": res["metrics"],
        "files":{"bars_csv":str((ROOT/"data"/csv_name).resolve()),
                 "trades_csv":str(rep_trades),
                 "summary_json":str(rep_json)},
        "created_utc": ts
    }
    with open(rep_json,"w",encoding="utf-8") as f: json.dump(summary,f,indent=2)
    return summary

def optimize(args=None):
    args = args or {}
    # space from CFG or quick sane defaults
    space = CFG.get("optimize",{}).get("space",{
        "ema_rsi_band":{
            "ema_fast":[8,12,16], "ema_slow":[20,26,34],
            "rsi_length":[10,14,21], "rsi_entry_low":[40,45,50], "rsi_entry_high":[60,65,70]
        },
        "macd_bb_trend":{
            "macd_fast":[8,12], "macd_slow":[20,26,32], "macd_signal":[9,12],
            "bb_period":[20,30], "bb_dev":[1.8,2.0,2.2]
        }
    })
    strat = args.get("strategy", CFG.get("strategy",{}).get("name","ema_rsi_band"))
    space = space.get(strat, {})
    # constraints
    max_dd = float(args.get("max_dd_pct", CFG.get("risk",{}).get("max_drawdown_pct_limit", 40)))
    min_trades = int(args.get("min_trades", 8))
    max_combos = int(args.get("max_combos", 60))

    # base defaults
    dfl = CFG.get("tools",{}).get("defaults",{})
    base = {"symbol": dfl.get("symbol","AUDCAD"), "timeframe": dfl.get("timeframe","5m"), "period": dfl.get("period","30d")}
    # create product of parameter options, cap count
    keys = list(space.keys())
    choices = [space[k] for k in keys]
    combos = list(itertools.product(*choices))
    random.shuffle(combos)
    combos = combos[:max_combos]

    best = None
    tried = 0
    for comb in combos:
        tried += 1
        over = base.copy()
        over["strategy"] = strat
        for k,v in zip(keys, comb): over[k] = v
        try:
            res = pipeline_run(over)
        except Exception as e:
            continue
        m = res["metrics"]
        if m["trades"] < min_trades: continue
        if m["max_drawdown_pct"] > max_dd: continue
        score = (m["sharpe_like"], m["profit_factor"] or -1, m["total_return_pct"])
        if best is None or score > best["score"]:
            best = {"score":score, "result":res, "params":{k:over[k] for k in keys}}
    return {"tried": tried, "best": best}

# -------------------- tools for openai --------------------
def tool_specs():
    return [
        {"type":"function","function":{"name":"local_shell","description":"Run a local shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}},
        {"type":"function","function":{"name":"python_exec","description":"Run a short Python snippet","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}},
        {"type":"function","function":{"name":"run_pipeline","description":"Run backtest with given overrides","parameters":{"type":"object","properties":{},"required":[]}}},
        {"type":"function","function":{"name":"optimize","description":"Grid-search params for a strategy under constraints","parameters":{"type":"object","properties":{"strategy":{"type":"string"},"max_dd_pct":{"type":"number"},"min_trades":{"type":"integer"},"max_combos":{"type":"integer"}},"required":[]}}}
    ]

def call_tool(name, args):
    if name=="local_shell":  return run_local_shell(args.get("cmd",""))
    if name=="python_exec":  return python_exec(args.get("code",""))
    if name=="run_pipeline": return json.dumps({"ok":True,"summary":pipeline_run(args)})
    if name=="optimize":     return json.dumps({"ok":True,"result":optimize(args)})
    return json.dumps({"ok":False,"error":"unknown tool"})

def get_openai_client():
    try:
        from openai import OpenAI
        return OpenAI()
    except Exception:
        return None

SYSTEM_PROMPT = """You are Jay's trading/dev agent.
Use tools to fetch data, run backtests, and optimize.
Constrain results: prefer Sharpe-like; reject candidates if MaxDD > limit or trades too few.
Be brief; show key metrics and file paths."""

# -------------------- REPL --------------------
def repl():
    print("Main Trading Orchestrator ready.")
    print("Commands:  run pipeline   optimize {json}   !!reload   !!tools   !!mode (openai|local)   !!quit")
    print("Utilities: !shell <cmd>   !py <code>")
    client = get_openai_client() if os.getenv("OPENAI_API_KEY") else None
    mode = "openai" if client else "local"
    messages = [{"role":"system","content":SYSTEM_PROMPT}]

    def llm_round(text):
        nonlocal messages, client
        messages.append({"role":"user","content":text})
        for _ in range(8):
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                messages=messages, tools=tool_specs(), tool_choice="auto"
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                messages.append({"role":"assistant","tool_calls":msg.tool_calls})
                for tc in msg.tool_calls:
                    out = call_tool(tc.function.name, json.loads(tc.function.arguments or "{}"))
                    messages.append({"role":"tool","tool_call_id":tc.id,"name":tc.function.name,"content":out})
                continue
            content = msg.content or ""
            messages.append({"role":"assistant","content":content})
            return content
        return "(tool loop cap)"

    while True:
        try: raw = input("you> ").strip()
        except (EOFError,KeyboardInterrupt): print("\nbye."); break
        if not raw: continue
        if raw in ("!!quit","!!exit"): print("bye."); break
        if raw=="!!tools": print("tools: local_shell, python_exec, run_pipeline, optimize"); continue
        if raw=="!!reload":
            global CFG; CFG = load_cfg(); print("reloaded."); continue
        if raw.startswith("!shell "): print(json.loads(run_local_shell(raw[7:]))); continue
        if raw.startswith("!py "):    print(json.loads(python_exec(raw[4:]))); continue
        if raw=="run pipeline":
            s = pipeline_run({})
            print("[pipeline] metrics:", s["metrics"]); print("[pipeline] files:", s["files"]); continue
        if raw.startswith("optimize"):
            j = {}
            if "{" in raw:
                try: j = json.loads(raw[raw.index("{"):])
                except: print("bad JSON after optimize"); continue
            r = optimize(j); print(r); continue
        if raw.startswith("!!mode"):
            parts = raw.split()
            if len(parts)==2 and parts[1] in ("openai","local"):
                mode = parts[1]; print(f"mode → {mode}")
                if mode=="openai" and not client:
                    client = get_openai_client()
                    if not client: print("no OPENAI_API_KEY; staying local.")
            else:
                print("usage: !!mode openai|local")
            continue
        # interactive LLM
        if mode=="openai":
            try: print(llm_round(raw))
            except Exception as e: print("LLM error:",e); traceback.print_exc()
        else:
            print("[local] " + raw)

if __name__ == "__main__":
    import json
    repl()
