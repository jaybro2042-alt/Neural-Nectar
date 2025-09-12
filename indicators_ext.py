import backtrader as bt
from .indicators_ext import StochRSI, Supertrend, KeltnerChannel, DonchianChannel, VWAP

class StrategyFlex(bt.Strategy):
    params = dict(
        stake=1.0, commission=0.0002,
        use_ema_rsi=True, ema_fast=12, ema_slow=26, rsi_length=14, rsi_lo=45, rsi_hi=65,
        use_macd_bb=False, macd_fast=12, macd_slow=26, macd_signal=9, bb_period=20, bb_dev=2.0,
        use_supertrend=False, st_period=10, st_mult=3.0,
        use_adx_filter=False, adx_period=14, adx_min=20,
        use_stochrsi=False, stoch_len=14, stoch_k=3, stoch_d=3, stoch_buy=20, stoch_sell=80,
        use_keltner=False, kel_ema=20, kel_atr=10, kel_mult=1.5,
        use_donchian=False, donch_period=20,
        use_vwap=False, vwap_period=None,
        use_or_logic=False, flat_on_cross=True
    )

    def __init__(self):
        p = self.p
        self.startcash = self.broker.getvalue()
        self.ema_f = bt.ind.EMA(period=p.ema_fast)
        self.ema_s = bt.ind.EMA(period=p.ema_slow)
        self.rsi   = bt.ind.RSI_Safe(period=p.rsi_length)
        self.macd  = bt.ind.MACD(self.data.close, period_me1=p.macd_fast, period_me2=p.macd_slow, period_signal=p.macd_signal)
        self.bb    = bt.ind.BollingerBands(period=p.bb_period, devfactor=p.bb_dev)
        self.adx   = bt.ind.ADX(period=p.adx_period)
        self.stochrsi = StochRSI(rsi_length=p.rsi_length, stoch_len=p.stoch_len, k=p.stoch_k, d=p.stoch_d)
        self.super    = Supertrend(period=p.st_period, multiplier=p.st_mult)
        self.keltner  = KeltnerChannel(ema_period=p.kel_ema, atr_period=p.kel_atr, mult=p.kel_mult)
        self.don      = DonchianChannel(period=p.donch_period)
        self.vwap     = VWAP(period=p.vwap_period)

    def long_condition(self):
        p = self.p; conds = []
        if p.use_ema_rsi:
            conds += [self.ema_f > self.ema_s, self.rsi >= p.rsi_lo, self.rsi <= p.rsi_hi]
        if p.use_macd_bb:
            conds += [self.macd.macd > self.macd.signal, self.data.close > self.bb.mid]
        if p.use_supertrend:  conds.append(self.super.dir > 0)
        if p.use_adx_filter:  conds.append(self.adx > p.adx_min)
        if p.use_stochrsi:    conds.append(self.stochrsi.k < p.stoch_buy)
        if p.use_keltner:     conds.append(self.data.close > self.keltner.mid)
        if p.use_donchian:    conds.append(self.data.close > self.don.mid)
        if p.use_vwap and p.vwap_period: conds.append(self.data.close > self.vwap.vwap)
        if not conds: return False
        return bt.ind.Or(*conds) if p.use_or_logic else bt.ind.And(*conds)

    def exit_condition(self):
        p=self.p; exits=[]
        if p.use_ema_rsi and p.flat_on_cross: exits.append(self.ema_f <= self.ema_s)
        if p.use_stochrsi: exits.append(self.stochrsi.k > p.stoch_sell)
        if not exits: return False
        return bt.ind.Or(*exits)

    def next(self):
        if not self.position:
            if self.long_condition():
                size = self.p.stake * (self.broker.getvalue() / max(self.data.close[0], 1e-9))
                self.buy(size=size)
        else:
            if self.exit_condition():
                self.close()

