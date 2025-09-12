import backtrader as bt

class MetricsExt(bt.Analyzer):
    def start(self):
        self._trades = []
        self._first_dt = None
        self._last_dt  = None

    def notify_trade(self, trade):
        if not trade.isclosed: return
        self._trades.append(dict(pnlcomm=float(trade.pnlcomm)))

    def next(self):
        dt = bt.num2date(self.strategy.data.datetime[0])
        if self._first_dt is None: self._first_dt = dt
        self._last_dt = dt

    def get_analysis(self):
        startcash = float(getattr(self.strategy, "startcash", 10000.0))
        endvalue  = float(self.strategy.broker.getvalue())
        total_ret = (endvalue / startcash) - 1.0

        dur_days = 1e-9
        if self._first_dt and self._last_dt:
            dur_days = max((self._last_dt - self._first_dt).total_seconds()/86400.0, 1e-9)
        ppd = (total_ret * 100.0) / dur_days

        wins   = [t for t in self._trades if t["pnlcomm"] > 0]
        losses = [t for t in self._trades if t["pnlcomm"] <= 0]
        winrate = (len(wins)/len(self._trades))*100.0 if self._trades else 0.0
        gp = sum(t["pnlcomm"] for t in wins)
        gl = -sum(t["pnlcomm"] for t in losses) if losses else 0.0
        pf = (gp/gl) if gl > 0 else None

        dd = self.strategy.analyzers.drawdown.get_analysis() if hasattr(self.strategy.analyzers,'drawdown') else {}
        maxdd_pct = float(dd.get('max', {}).get('drawdown', 0.0))

        return dict(
            bars = int(len(self.strategy.data)),
            first_ts = self._first_dt.isoformat() if self._first_dt else None,
            last_ts  = self._last_dt.isoformat() if self._last_dt else None,
            total_return_pct = round(total_ret*100.0, 3),
            profit_per_day_pct = round(ppd, 3),
            win_rate_pct = round(winrate, 2),
            profit_factor = None if pf is None else round(pf, 3),
            max_drawdown_pct = round(maxdd_pct, 2),
            trades = len(self._trades),
            start_cash = startcash,
            end_value = endvalue
        )
