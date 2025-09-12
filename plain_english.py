def plain_english_from_summary(summary):
    """Return a concise plain-English summary string from a pipeline summary dict."""
    s = summary or {}
    sym = s.get("symbol", "unknown")
    interval = s.get("interval", "")
    params = s.get("params", {})
    m = s.get("metrics", {})
    pf = m.get("profit_factor")
    dd = m.get("max_drawdown_pct")
    ret = m.get("total_return_pct")
    trades = m.get("trades")

    parts = []
    parts.append(f"Backtest: {sym} on {interval}.")
    parts.append(f"Params: EMA fast={params.get('ema_fast')}, slow={params.get('ema_slow')}, RSI={params.get('rsi_len')}.")
    if ret is not None:
        parts.append(f"Total return: {ret:+.2f}%.")
    if pf is not None:
        if pf < 1.0:
            parts.append(f"Profit factor {pf:.2f} — unprofitable under these settings.")
        else:
            parts.append(f"Profit factor {pf:.2f} — profitable.")
    if dd is not None:
        parts.append(f"Max drawdown: {dd:.2f}%.")
    if trades is not None:
        parts.append(f"Trades: {trades}.")

    if pf is not None and pf < 1.0:
        parts.append("Recommendation: run a parameter sweep (EMA/RSI ranges) or try a different symbol/timeframe.")
    else:
        parts.append("Performance looks acceptable; consider walk-forward tests or longer period validation.")

    return " ".join(parts)
