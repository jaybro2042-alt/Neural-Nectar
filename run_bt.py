import json, pathlib, sys
import pandas as pd
import backtrader as bt

# Make sure we can import bt_core.*
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bt_core.strategy_flex import StrategyFlex
from bt_core.metrics_ext import MetricsExt

DATA    = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# ---------- robust CSV loader ----------
def _load_bars_csv(csv_path):
    df = pd.read_csv(csv_path)

    # 1) timestamp column: pick any col with date/time in name, else first col
    ts_col = None
    for c in df.columns:
        lc = str(c).lower()
        if "time" in lc or "date" in lc or "timestamp" in lc:
            ts_col = c
            break
    if ts_col is None:
        ts_col = df.columns[0]

    # 2) parse datetime, sort, tz-naive (Backtrader wants naive)
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
    if mapping:
        df = df.rename(columns=mapping)

    # 4) ensure required columns exist
    for k in ["Open","High","Low","Close","Volume"]:
        if k not in df.columns:
            df[k] = 0.0 if k == "Volume" else df["Close"]

    # 5) coerce to numeric and drop junk rows
    for k in ["Open","High","Low","Close","Volume"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df = df.dropna(subset=["Open","High","Low","Close"])
    df["Volume"] = df["Volume"].fillna(0.0)

    # 6) keep only what Backtrader needs
    return df[["Open","High","Low","Close","Volume"]]

# ---------- runner ----------
def run_backtest_from_spec(spec_path: str):
    spec_p = (ROOT / spec_path).resolve()
    spec = json.loads(spec_p.read_text(encoding="utf-8-sig"))  # tolerant to BOM

    data_cfg = spec.get("data", {})
    csv_path = data_cfg.get("csv")
    if not csv_path:
        csv_path = str((DATA / f"{data_cfg.get('symbol','AUDCAD')}_{data_cfg.get('interval','5m')}.csv").resolve())
    df = _load_bars_csv(csv_path)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(float(spec.get("initial_capital", 10000)))
    cerebro.broker.setcommission(commission=float(spec.get("commission", 0.0002)))

    datafeed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(datafeed)

    params = dict(stake=1.0, commission=float(spec.get("commission", 0.0002)))
    params.update(spec.get("params", {}))
    cerebro.addstrategy(StrategyFlex, **params)

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(MetricsExt, _name="metrics")

    res = cerebro.run()
    strat = res[0]
    metrics = strat.analyzers.metrics.get_analysis()

    name = spec.get("name", spec_p.stem)
    out = {
        "spec": name,
        "metrics": metrics,
        "files": {
            "summary_json": str((REPORTS / f"bt_summary_{name}.json").resolve()),
            "bars_csv": csv_path
        }
    }
    (REPORTS / f"bt_summary_{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out

if __name__ == "__main__":
    import sys as _sys
    spec = _sys.argv[1] if len(_sys.argv) > 1 else "strategies\\base_ema_rsi.json"
    run_backtest_from_spec(spec)
