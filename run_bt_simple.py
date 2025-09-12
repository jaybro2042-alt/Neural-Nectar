import json, pathlib, datetime
from strategies.backtrader_runner import run_backtest  # assumes you have this module

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True, parents=True)

# Strategy specs to try
strategy_files = [
    "strategies/default_ema_rsi.json",
    "strategies/scary_omni.json",
    "strategies/scary_osprey.json",
    "strategies/scary_roughdeluxe.json",
    "strategies/scary_skeleton.json",
]

# Bars CSV your pipeline writes (make sure it exists first)
csv_path = DATA / "AUDCAD_5m.csv"

results = []
for sf in strategy_files:
    spec_path = ROOT / sf
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as e:
        results.append({"file": sf, "error": f"read/parse: {e}"})
        continue

    try:
        # Run your backtrader backtest
        metrics = run_backtest(
            spec,
            csv_path=str(csv_path),
            cash=spec.get("initial_capital", 10_000),
            commission=spec.get("commission", 0.0002),
            timeframe="5m",
        )
        # Add profit_per_day if runner doesn’t provide it
        if "profit_per_day_pct" not in metrics:
            try:
                total = float(metrics.get("total_return_pct", 0.0)) / 100.0
                bars = int(metrics.get("bars", 0)) or None
                # If your runner returns start/end times, use them; otherwise estimate by bars
                if "first_ts" in metrics and "last_ts" in metrics:
                    t0 = datetime.datetime.fromisoformat(metrics["first_ts"])
                    t1 = datetime.datetime.fromisoformat(metrics["last_ts"])
                    days = max((t1 - t0).total_seconds() / 86400.0, 1e-9)
                else:
                    # crude fallback: if 5m bars, 288 bars/day
                    days = (bars / 288.0) if bars else 1.0
                metrics["profit_per_day_pct"] = round((total * 100.0) / days, 3)
            except Exception:
                pass

        out = {"file": sf, "metrics": metrics}
        results.append(out)

        # Write per-file summary
        outp = REPORTS / f"bt_summary_{spec.get('name', spec_path.stem)}.json"
        outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    except Exception as e:
        results.append({"file": sf, "error": f"run_backtest: {e}"})

# Consolidated dump
consolidated = REPORTS / "bt_consolidated.json"
consolidated.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("done; wrote", consolidated)
