import math
import backtrader as bt

def _finite(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None

class StrategyFlex(bt.Strategy):
    params = dict(
        stake=1.0, commission=0.0002,
        enable_longs=True, enable_shorts=True,

        use_ema_rsi=True, ema_fast=12, ema_slow=26, rsi_length=14, rsi_lo=45, rsi_hi=65,
        use_macd_bb=False, macd_fast=12, macd_slow=26, macd_signal=9, bb_period=20, bb_dev=2.0,
        use_supertrend=False, st_period=10, st_mult=3.0,
        use_adx_filter=False, adx_period=14, adx_min=20,
        use_stochrsi=False, stoch_len=14, stoch_k=3, stoch_d=3, stoch_buy=20, stoch_sell=80,
        use_keltner=False, kel_ema=20, kel_atr=10, kel_mult=1.5,
        use_donchian=False, donch_period=20,
        use_vwap=False, vwap_period=None,

        use_or_logic=False,  # False -> ALL conditions (AND); True -> ANY (OR)
        flat_on_cross=True
    )

    def __init__(self):
        p = self.p
        self.startcash = self.broker.getvalue()

        # Build only what we use; import inside to avoid circular init.
        if p.use_ema_rsi:
            self.ema_f = bt.ind.EMA(period=p.ema_fast)
            self.ema_s = bt.ind.EMA(period=p.ema_slow)
            self.rsi   = bt.ind.RSI_Safe(period=p.rsi_length)

        if p.use_macd_bb:
            self.macd = bt.ind.MACD(self.data.close, period_me1=p.macd_fast,
                                    period_me2=p.macd_slow, period_signal=p.macd_signal)
            self.bb   = bt.ind.BollingerBands(period=p.bb_period, devfactor=p.bb_dev)

        if p.use_adx_filter:
            self.adx = bt.ind.ADX(period=p.adx_period)

        if p.use_stochrsi:
            from .indicators_ext import StochRSI
            self.stochrsi = StochRSI(rsi_length=p.rsi_length, stoch_len=p.stoch_len,
                                     k=p.stoch_k, d=p.stoch_d)

        if p.use_supertrend:
            from .indicators_ext import Supertrend
            self.super = Supertrend(period=p.st_period, multiplier=p.st_mult)

        if p.use_keltner:
            from .indicators_ext import KeltnerChannel
            self.keltner = KeltnerChannel(ema_period=p.kel_ema, atr_period=p.kel_atr, mult=p.kel_mult)

        if p.use_donchian:
            from .indicators_ext import DonchianChannel
            self.don = DonchianChannel(period=p.donch_period)

        if p.use_vwap and p.vwap_period is not None:
            from .indicators_ext import VWAP
            self.vwap = VWAP(period=p.vwap_period)

    # --------- boolean signal evaluators (use Python bools, not LineOps) ---------
    def _long_ok(self):
        p = self.p
        conds = []

        c = _finite(self.data.close[0])

        if p.use_ema_rsi:
            emaf = _finite(self.ema_f[0]); emas = _finite(self.ema_s[0]); rsi = _finite(self.rsi[0])
            if None not in (emaf, emas): conds.append(emaf > emas)
            if rsi is not None: conds += [rsi >= p.rsi_lo, rsi <= p.rsi_hi]

        if p.use_macd_bb:
            macd = _finite(self.macd.macd[0]); sig = _finite(self.macd.signal[0]); mid = _finite(self.bb.mid[0])
            if None not in (macd, sig): conds.append(macd > sig)
            if None not in (c, mid):    conds.append(c > mid)

        if p.use_supertrend:
            d = _finite(self.super.dir[0])
            if d is not None: conds.append(d > 0)

        if p.use_adx_filter:
            adx = _finite(self.adx[0])
            if adx is not None: conds.append(adx > p.adx_min)

        if p.use_stochrsi:
            k = _finite(self.stochrsi.k[0])
            if k is not None: conds.append(k < p.stoch_buy)

        if p.use_keltner:
            mid = _finite(self.keltner.mid[0])
            if None not in (c, mid): conds.append(c > mid)

        if p.use_donchian:
            mid = _finite(self.don.mid[0])
            if None not in (c, mid): conds.append(c > mid)

        if p.use_vwap and p.vwap_period is not None:
            vv = _finite(self.vwap.vwap[0])
            if None not in (c, vv): conds.append(c > vv)

        if not conds: return False
        return any(conds) if p.use_or_logic else all(conds)

    def _short_ok(self):
        p = self.p
        conds = []

        c = _finite(self.data.close[0])

        if p.use_ema_rsi:
            emaf = _finite(self.ema_f[0]); emas = _finite(self.ema_s[0]); rsi = _finite(self.rsi[0])
            if None not in (emaf, emas): conds.append(emaf < emas)
            if rsi is not None: conds += [rsi <= (100 - p.rsi_lo), rsi >= (100 - p.rsi_hi)]

        if p.use_macd_bb:
            macd = _finite(self.macd.macd[0]); sig = _finite(self.macd.signal[0]); mid = _finite(self.bb.mid[0])
            if None not in (macd, sig): conds.append(macd < sig)
            if None not in (c, mid):    conds.append(c < mid)

        if p.use_supertrend:
            d = _finite(self.super.dir[0])
            if d is not None: conds.append(d < 0)

        if p.use_adx_filter:
            adx = _finite(self.adx[0])
            if adx is not None: conds.append(adx > p.adx_min)

        if p.use_stochrsi:
            k = _finite(self.stochrsi.k[0])
            if k is not None: conds.append(k > p.stoch_sell)

        if p.use_keltner:
            mid = _finite(self.keltner.mid[0])
            if None not in (c, mid): conds.append(c < mid)

        if p.use_donchian:
            mid = _finite(self.don.mid[0])
            if None not in (c, mid): conds.append(c < mid)

        if p.use_vwap and p.vwap_period is not None:
            vv = _finite(self.vwap.vwap[0])
            if None not in (c, vv): conds.append(c < vv)

        if not conds: return False
        return any(conds) if p.use_or_logic else all(conds)

    def next(self):
        long_ok  = self._long_ok()  if self.p.enable_longs  else False
        short_ok = self._short_ok() if self.p.enable_shorts else False

        if not self.position:
            if long_ok:
                size = self.p.stake * (self.broker.getvalue() / max(_finite(self.data.close[0]) or 1.0, 1e-9))
                self.buy(size=size)
            elif short_ok:
                size = self.p.stake * (self.broker.getvalue() / max(_finite(self.data.close[0]) or 1.0, 1e-9))
                self.sell(size=size)
        else:
            exit_now = False
            if self.getposition().size > 0:
                if self.p.use_ema_rsi and self.p.flat_on_cross and (_finite(self.ema_f[0]) is not None) and (_finite(self.ema_s[0]) is not None) and (self.ema_f[0] <= self.ema_s[0]):
                    exit_now = True
                if self.p.use_stochrsi and (_finite(self.stochrsi.k[0]) is not None) and (self.stochrsi.k[0] > self.p.stoch_sell):
                    exit_now = True
            else:
                if self.p.use_ema_rsi and self.p.flat_on_cross and (_finite(self.ema_f[0]) is not None) and (_finite(self.ema_s[0]) is not None) and (self.ema_f[0] >= self.ema_s[0]):
                    exit_now = True
                if self.p.use_stochrsi and (_finite(self.stochrsi.k[0]) is not None) and (self.stochrsi.k[0] < self.p.stoch_buy):
                    exit_now = True
            if exit_now:
                self.close()
