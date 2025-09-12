import json, math, itertools, argparse, pathlib, sys
import pandas as pd
import backtrader as bt

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bt_core.strategy_flex import StrategyFlex
from bt_core.metrics_ext   import MetricsExt

DATA    = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# -------- robust CSV loader (same logic as run_bt.py) --------
def _load_bars_csv(csv_path):
    df = pd.read_csv(csv_path)

    # 1) find timestamp column
    ts_col = None
    for c in df.columns:
        lc = str(c).lower()
        if ("time" in lc) or ("date" in lc) or ("timestamp" in lc):
            ts_col = c; break
    if ts_col is None:
        ts_col = df.columns[0]

    # 2) parse datetime, sort, tz-naive
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    # 3) normalize column names
    mapping = {}
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc in ("open","o"): mapping[c] = "Open"
        elif lc in ("high","h"): mapping[c] = "High"
        elif lc in ("low","l"): mapping[c] = "Low"
        elif lc in ("close","c","adj close","adj_close"): mapping[c] = "Close"
        elif lc in ("volume","vol","v"): mapping[c] = "Volume"
        elif lc in ("openinterest","oi","open_interest"): mapping[c] = "OpenInterest"
    if mapping: df = df.rename(columns=mapping)

    # 4) ensure columns exist
    for k in ["Open","High","Low","Close","Volume"]:
        if k not in df.columns:
            df[k] = 0.0 if k == "Volume" else df["Close"]

    # 5) coerce numerics & drop junk
    for k in ["Open","High","Low","Close","Volume"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df = df.dropna(subset=["Open","High","Low","Close"])
    df["Volume"] = df["Volume"].fillna(0.0)

    return df[["Open","High","Low","Close","Volume"]]

# -------- single run on a dataframe --------
def _run_on_df(df, initial_capital, commission, params):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(float(initial_capital))
    cerebro.broker.setcommission(commission=float(commission))
    datafeed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(datafeed)
    cerebro.addstrategy(StrategyFlex, **params)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(MetricsExt,            _name="metrics")
    res = cerebro.run()
    strat = res[0]
    return strat.analyzers.metrics.get_analysis()

def _score(m):
    # Prefer daily compounding speed, then PF, then raw total return
    ppd = m.get("profit_per_day_pct", -9e9) or -9e9
    pf  = m.get("profit_factor", None)
    pfv = -1 if pf in (None, float("inf")) else pf
    tr  = m.get("total_return_pct", -9e9) or -9e9
    return (ppd, pfv, tr)

# -------- grid product from dict of lists --------
def _grid(space):
    keys = list(space.keys())
    vals = [space[k] if isinstance(space[k], (list, tuple)) else [space[k]] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

# -------- walk-forward main --------
def walkforward(spec_path, k=1, train_pct=0.7, name=None):
    spec_p = (ROOT / spec_path).resolve()
    spec   = json.loads(spec_p.read_text(encoding="utf-8-sig"))

    data_cfg = spec.get("data", {})
    csv_path = data_cfg.get("csv")
    if not csv_path:
        csv_path = str((DATA / f"{data_cfg.get('symbol','AUDCAD')}_{data_cfg.get('interval','5m')}.csv").resolve())
    df = _load_bars_csv(csv_path)

    name = name or spec.get("name", spec_p.stem)
    initial_capital = spec.get("initial_capital", 10000)
    commission      = spec.get("commission", 0.0002)
    base_params     = dict(stake=1.0, commission=commission)
    base_params.update(spec.get("params", {}))

    strategy_name = spec.get("strategy", "ema_rsi_band")  # for reporting only

    # Search space (from spec.search_space or sensible defaults)
    space = spec.get("search_space")
    if not space:
        # modest defaults so it runs quickly; expand later
        space = {
            "ema_fast":       [8, 12, 16],
            "ema_slow":       [20, 26, 34],
            "rsi_length":     [10, 14],
            "rsi_lo":         [40, 45],
            "rsi_hi":         [60, 65],
            "enable_longs":   [True],
            "enable_shorts":  [True]
        }

    N = len(df)
    if N < 200:
        raise RuntimeError(f"Not enough bars for walk-forward: have {N}, need at least ~200.")

    results = []
    # Build k folds: rolling windows with fixed train:test ratio
    # e.g., k=1 => single split; k>1 => slide forward evenly
    split_points = []
    if k <= 1:
        split_points = [int(N * train_pct)]
    else:
        step = int((N * (1.0 - train_pct)) / k) or 1
        start = int(N * train_pct)
        split_points = [start + i*step for i in range(k)]
        split_points = [s for s in split_points if s < N-10]  # leave at least some test bars

    for i, sp in enumerate(split_points):
        tr_lo = 0
        tr_hi = sp
        te_lo = sp
        te_hi = N if i == len(split_points)-1 else min(N, sp + (N - sp)//k or 50)

        df_tr = df.iloc[tr_lo:tr_hi]
        df_te = df.iloc[te_lo:te_hi]
        if len(df_tr) < 100 or len(df_te) < 50:
            continue

        # grid search on train
        best = None
        for cand in _grid(space):
            params = base_params.copy()
            params.update(cand)
            m_tr = _run_on_df(df_tr, initial_capital, commission, params)
            s = _score(m_tr)
            if (best is None) or (s > best["score"]):
                best = {"score": s, "params": cand, "train": m_tr}

        # evaluate on test
        use_params = base_params.copy()
        use_params.update(best["params"])
        m_te = _run_on_df(df_te, initial_capital, commission, use_params)

        results.append({
            "fold": i+1,
            "train_idx": [int(df_tr.index[0].timestamp()), int(df_tr.index[-1].timestamp())],
            "test_idx":  [int(df_te.index[0].timestamp()), int(df_te.index[-1].timestamp())],
            "best_params": best["params"],
            "train_metrics": best["train"],
            "test_metrics":  m_te
        })

    # aggregate
    def _avg(key):
        vals = [r["test_metrics"].get(key) for r in results if r["test_metrics"].get(key) is not None]
        return None if not vals else round(sum(vals)/len(vals), 4)

    summary = {
        "spec": name,
        "strategy": strategy_name,
        "csv": csv_path,
        "k_folds": len(results),
        "train_pct": train_pct,
        "avg_test": {
            "profit_per_day_pct": _avg("profit_per_day_pct"),
            "total_return_pct":   _avg("total_return_pct"),
            "win_rate_pct":       _avg("win_rate_pct"),
            "profit_factor":      _avg("profit_factor"),
            "max_drawdown_pct":   _avg("max_drawdown_pct"),
        },
        "folds": results
    }

    outpath = REPORTS / f"wf_{name}.json"
    outpath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", default=r"strategies\base_ema_rsi.json")
    ap.add_argument("--k", type=int, default=1, help="number of walk-forward folds (1=single split)")
    ap.add_argument("--train", type=float, default=0.7, help="training fraction per fold (0-1)")
    ap.add_argument("--name", type=str, default=None, help="override run name")
    args = ap.parse_args()
    walkforward(args.spec, k=args.k, train_pct=args.train, name=args.name)
