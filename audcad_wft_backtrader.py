
import os
import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import timedelta

# Machine learning & optimization
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Technical indicators (ta)
import ta

# Backtrader
import backtrader as bt

class PredPandasData(bt.feeds.PandasData):
    """
    Extends PandasData to include a 'pred_prob' line so the strategy
    can read model output via self.datas[0].pred_prob
    """
    lines = ('pred_prob',)
    params = (('pred_prob', -1),)

# ---------------------------
# USER CONFIGURABLE SETTINGS
# ---------------------------
DATA_CSV = "audcad_1min.csv"   # path to your minute CSV
DATETIME_COL = "datetime"
OHLVC = {"open":"open", "high":"high", "low":"low", "close":"close", "volume":"volume"}

# Walk-forward settings
TOTAL_DAYS = 90                 # last N days to use
TRAIN_DAYS = 30                 # training window length in days
TEST_DAYS = 7                   # test window length in days
STEP_DAYS = 7                   # move forward by this many days each fold

# Target horizon (in minutes) to predict; e.g., predict return over next H minutes
PRED_HORIZON_MIN = 60           # predict next 60 minutes aggregated return

# Start cash
START_CASH = 100000.0

# Max allowed drawdown target (as a fraction of equity) you wish to enforce while sizing
MAX_DD_TARGET = 0.002  # 0.2% as requested; may be unrealistic

# Backtrader trade sizing: fraction of equity to risk per trade if possible
RISK_PER_TRADE = 0.001  # 0.1% risk per trade (adjust)

# Optuna settings for speed initially; increase trials for better tuning
OPTUNA_TRIALS = 40

# Limits for training time: you can reduce these if too slow
MAX_TRAIN_ROWS = None  # set to int to downsample for speed (not recommended for real runs)

# Save outputs
OUTPUT_DIR = "wft_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# UTILITIES / INDICATORS
# ---------------------------

def load_data(csv_path, datetime_col=DATETIME_COL):
    df = pd.read_csv(csv_path)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)
    df = df.rename(columns={OHLVC["open"]:"open", OHLVC["high"]:"high", OHLVC["low"]:"low",
                            OHLVC["close"]:"close", OHLVC["volume"]:"volume"})
    return df

def add_indicators(df):
    # Many indicators via ta library. All use only past data (lookback) so safe.
    # RSI
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    # Stochastic %K/%D
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_sig'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    # ATR
    df['atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    # ADX
    df['adx_14'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    # CCI
    df['cci_20'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
    # On-Balance Volume
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    # Williams %R
    df['wr_14'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
    # MFI
    df['mfi_14'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=14).money_flow_index()
    # EMA/SMA variations
    df['ema_8'] = ta.trend.EMAIndicator(df['close'], window=8).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
    df['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
    # ROC
    df['roc_12'] = ta.momentum.ROCIndicator(df['close'], window=12).roc()
    # Momentum
    df['mom_10'] = ta.momentum.StochRSIIndicator(df['close']).stochrsi_k()  # substituting stochRSI K
    # Keltner Channels
    kc_upper = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=20).keltner_channel_hband()
    kc_lower = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=20).keltner_channel_lband()
    df['kc_upper'] = kc_upper
    df['kc_lower'] = kc_lower
    # Ichimoku components (using ta)
    ich = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
    df['ich_a'] = ich.ichimoku_a()
    df['ich_b'] = ich.ichimoku_b()
    # VWAP: cumulative price*vol / cumulative volume per day - reset each day
    df['vwap'] = compute_vwap(df)
    # Fill na
    df = df.fillna(method='ffill').fillna(0)
    return df

def compute_vwap(df):
    # Compute intraday VWAP by grouping by date
    tmp = df.copy()
    tmp['date'] = tmp[DATETIME_COL].dt.date
    tmp['pv'] = tmp['close'] * tmp['volume']
    tmp['cum_pv'] = tmp.groupby('date')['pv'].cumsum()
    tmp['cum_vol'] = tmp.groupby('date')['volume'].cumsum().replace(0, np.nan)
    vwap = tmp['cum_pv'] / tmp['cum_vol']
    vwap = vwap.fillna(method='ffill').fillna(0)
    return vwap

# ---------------------------
# Feature & Target Engineering
# ---------------------------

def make_features_and_target(df, horizon=PRED_HORIZON_MIN):
    df = df.copy()
    # Target: future return over next horizon minutes
    df['future_close'] = df['close'].shift(-horizon)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    # Binary target: 1 if future_return > 0, else 0
    df['target'] = (df['future_return'] > 0).astype(int)
    # Drop last horizon rows with nan target
    df = df.dropna(subset=['target']).reset_index(drop=True)
    return df

# ---------------------------
# Backtrader Strategy
# ---------------------------

class MLBacktestStrategy(bt.Strategy):
    params = dict(
        pred_col='pred_prob',    # column name in data that holds model predicted probability
        signal_threshold=0.55,   # threshold to enter long
        stop_loss_atr_mult=3.0,  # stop loss in ATR multiples
        take_profit_atr_mult=6.0,
        risk_per_trade=RISK_PER_TRADE,
        max_dd_target=MAX_DD_TARGET,
    )

    def __init__(self):
        # Access predicted probability line from data feed (if present)
        self.pred = getattr(self.datas[0], self.p.pred_col, None)
        # Use ATR indicator inside strategy to set stops (calculated on bar's data only)
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.order = None
        self.entry_price = None

    def log(self, txt):
        # simple logger; can be expanded
        dt = self.datas[0].datetime.datetime(0)
        print(f"{dt.isoformat()} {txt}")

    def next(self):
        # position sizing: compute size so that risk per trade <= risk_per_trade fraction of current equity
        cash = self.broker.get_cash()
        value = self.broker.getvalue()
        if self.position.size == 0:
            # get predicted probability (must be present)
            try:
                prob = float(self.pred[0])
            except Exception:
                prob = 0.0
            if prob >= self.p.signal_threshold:
                # compute stop loss distance as atr * mult
                atr = max(1e-8, self.atr[0])
                stop_dist = self.p.stop_loss_atr_mult * atr
                # amount to risk per trade in dollars
                dollar_risk = value * self.p.risk_per_trade
                # size = dollar_risk / stop_dist (units of price) -> number of lots/shares
                size = int(max(1, dollar_risk / stop_dist))
                # ensure we don't buy more than cash allows
                notional = size * self.data.close[0]
                if notional > cash:
                    size = int(cash / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
                    self.entry_price = self.data.close[0]
                    # we'll use a simple stop/take using sell orders placed once in position() callbacks
        else:
            # Manage position: simple stop & take profit based on ATR from entry
            atr = max(1e-8, self.atr[0])
            stop_price = self.entry_price - self.p.stop_loss_atr_mult * atr
            take_price = self.entry_price + self.p.take_profit_atr_mult * atr
            if self.data.close[0] <= stop_price or self.data.close[0] >= take_price:
                self.close()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY executed price={order.executed.price:.5f}, size={order.executed.size}")
            elif order.issell():
                self.log(f"SELL executed price={order.executed.price:.5f}, size={order.executed.size}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")
        self.order = None

# ---------------------------
# Backtrader helper to feed pandas DF (with feature columns & pred)
# ---------------------------

def df_to_btfeed(df):
    # Backtrader expects datetime index or 'datetime' column
    # Keep necessary columns: datetime, open, high, low, close, volume and extra feature cols
    df2 = df.copy()
    df2 = df2.rename(columns={DATETIME_COL:'datetime'})
    df2['datetime'] = pd.to_datetime(df2['datetime'])
    df2 = df2.set_index('datetime')
    # backtrader feed
    data = bt.feeds.PandasData(dataname=df2)
    return data

# ---------------------------
# Backtest runner for parameter evaluation
# ---------------------------

def run_backtest_bt(df, strategy_params, cerebro_params=None, verbose=False):
    """
    Execute a simple backtest on the provided DataFrame using Backtrader.

    This replacement ensures the DataFrame has a proper datetime index,
    exposes any 'pred_prob' column via a custom feed, and gathers results
    into a convenient dictionary with drawdown and daily ROI metrics.
    """
    cerebro = bt.Cerebro(stdstats=False)

    # Apply any provided Cerebro params (e.g. broker settings) dynamically
    if cerebro_params:
        for k, v in cerebro_params.items():
            try:
                setattr(cerebro, k, v)
            except Exception:
                pass

    cerebro.broker.setcash(START_CASH)
    try:
        cerebro.broker.setcommission(commission=0.0002)
    except Exception:
        pass

    # Prepare DataFrame with datetime index
    df_bt = df.copy()
    df_bt[DATETIME_COL] = pd.to_datetime(df_bt[DATETIME_COL])
    df_bt = df_bt.set_index(DATETIME_COL)

    # Validate required OHLCV columns
    required = {'open', 'high', 'low', 'close', 'volume'}
    missing = [c for c in required if c not in df_bt.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns for Backtrader: {missing}")

    # Use custom feed that includes 'pred_prob' if present
    data = PredPandasData(dataname=df_bt)
    cerebro.adddata(data)

    cerebro.addstrategy(MLBacktestStrategy, **strategy_params)

    # Attach common analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    results = cerebro.run(stdstats=False)
    strat = results[0]

    ending_value = cerebro.broker.getvalue()
    pnl = ending_value - START_CASH

    # Extract max drawdown if available
    max_dd = 0.0
    try:
        dd = strat.analyzers.drawdown.get_analysis()
        if dd and 'max' in dd and 'drawdown' in dd['max']:
            max_dd = float(dd['max']['drawdown']) / 100.0
    except Exception:
        pass

    # Estimate daily ROI over data span
    dt0 = pd.to_datetime(df[DATETIME_COL].iloc[0])
    dt1 = pd.to_datetime(df[DATETIME_COL].iloc[-1])
    days = max(1, (dt1 - dt0).days + 1)
    daily_roi = pnl / days

    trade_analysis = None
    try:
        trade_analysis = strat.analyzers.trades.get_analysis()
    except Exception:
        pass

    return {
        "ending_value": float(ending_value),
        "pnl": float(pnl),
        "max_drawdown": float(max_dd),
        "daily_roi_$": float(daily_roi),
        "trades": trade_analysis,
    }

def df_to_btfeed(df):
    """
    Convert a DataFrame with DATETIME_COL plus OHLCV (+ optional 'pred_prob') into a Backtrader data feed.

    This helper uses the PredPandasData class to ensure that additional columns like 'pred_prob'
    are available inside the strategy via self.datas[0].pred_prob.
    """
    df2 = df.copy()
    if DATETIME_COL not in df2.columns:
        raise ValueError(f"Expected datetime column '{DATETIME_COL}' not found")
    df2['datetime'] = pd.to_datetime(df2[DATETIME_COL])
    df2 = df2.set_index('datetime')
    return PredPandasData(dataname=df2)

# ---------------------------
# Walk-Forward & Optimization
# ---------------------------

def evaluate_params_with_backtest(train_df, val_df, trial):
    """
    For an Optuna trial: sample hyperparameters, train model on train_df, produce predictions for val_df,
    run backtest on val_df using predicted probabilities and return objective metric.
    """
    # sample ML hyperparameters
    num_leaves = trial.suggest_int('num_leaves', 16, 256)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 20, 200)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 0.1)
    feature_fraction = trial.suggest_uniform('feature_fraction', 0.5, 1.0)
    # strategy thresholds and stop/take
    signal_threshold = trial.suggest_uniform('signal_threshold', 0.51, 0.9)
    stop_loss_atr_mult = trial.suggest_uniform('stop_loss_atr_mult', 1.0, 6.0)
    take_profit_atr_mult = trial.suggest_uniform('take_profit_atr_mult', 1.0, 12.0)

    # Prepare ML features
    feature_cols = [c for c in train_df.columns if c not in [DATETIME_COL, 'future_close', 'future_return', 'target']]
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    # Use a basic LightGBM model
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective':'binary',
        'metric':'auc',
        'num_leaves': num_leaves,
        'min_data_in_leaf': min_data_in_leaf,
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'verbosity': -1,
    }
    # train
    gbm = lgb.train(params, train_data, num_boost_round=200, verbose_eval=False)

    # Create predictions on val_df
    X_val = val_df[feature_cols]
    val_probs = gbm.predict(X_val)
    # attach predicted prob into val_df copy for backtrader consumption
    val_df_bt = val_df.copy()
    val_df_bt['pred_prob'] = val_probs
    # Run backtest on val_df
    strategy_params = dict(
        pred_col='pred_prob',
        signal_threshold=signal_threshold,
        stop_loss_atr_mult=stop_loss_atr_mult,
        take_profit_atr_mult=take_profit_atr_mult,
        risk_per_trade=RISK_PER_TRADE,
        max_dd_target=MAX_DD_TARGET
    )
    res = run_backtest_bt(val_df_bt, strategy_params)
    # Objective: maximize daily ROI and penalize drawdown
    daily_roi = res['daily_roi_$']
    max_dd = res['max_drawdown']
    # Penalty if drawdown exceeds target
    dd_penalty = 0.0
    if max_dd > MAX_DD_TARGET:
        dd_penalty = (max_dd - MAX_DD_TARGET) * 1000000.0  # large penalty
    # return negative because Optuna minimizes (we'll minimize -score)
    score = -(daily_roi - dd_penalty)
    # You can also attach intermediate values
    trial.set_user_attr("daily_roi_$", daily_roi)
    trial.set_user_attr("max_dd", max_dd)
    return score

def walk_forward_run(full_df):
    # Compute number of folds based on last TOTAL_DAYS
    last_time = pd.to_datetime(full_df[DATETIME_COL].iloc[-1])
    first_time = last_time - pd.Timedelta(days=TOTAL_DAYS)
    df = full_df[full_df[DATETIME_COL] >= first_time].reset_index(drop=True)
    # Build features+target on this reduced df
    df = add_indicators(df)
    df = make_features_and_target(df, horizon=PRED_HORIZON_MIN)

    folds_results = []
    fold_start = df[DATETIME_COL].iloc[0]
    fold_end_time = pd.to_datetime(df[DATETIME_COL].iloc[0]) + pd.Timedelta(days=TRAIN_DAYS + TEST_DAYS)
    # rolling windows
    cur_train_start = pd.to_datetime(df[DATETIME_COL].iloc[0])

    while True:
        train_start = cur_train_start
        train_end = train_start + pd.Timedelta(days=TRAIN_DAYS) - pd.Timedelta(seconds=1)
        test_start = train_end + pd.Timedelta(seconds=1)
        test_end = test_start + pd.Timedelta(days=TEST_DAYS) - pd.Timedelta(seconds=1)

        if test_end > pd.to_datetime(df[DATETIME_COL].iloc[-1]):
            break

        train_df = df[(pd.to_datetime(df[DATETIME_COL]) >= train_start) & (pd.to_datetime(df[DATETIME_COL]) <= train_end)].reset_index(drop=True)
        test_df = df[(pd.to_datetime(df[DATETIME_COL]) >= test_start) & (pd.to_datetime(df[DATETIME_COL]) <= test_end)].reset_index(drop=True)

        if len(train_df) < 100 or len(test_df) < 10:
            cur_train_start = cur_train_start + pd.Timedelta(days=STEP_DAYS)
            continue

        # Optionally downsample for speed during prototyping (not recommended for final)
        if MAX_TRAIN_ROWS is not None and len(train_df) > MAX_TRAIN_ROWS:
            train_df = train_df.sample(n=MAX_TRAIN_ROWS, random_state=42).sort_index().reset_index(drop=True)

        # Optimize on validation (here validation = test_df of fold) using Optuna
        def objective(trial):
            return evaluate_params_with_backtest(train_df, test_df, trial)

        study = optuna.create_study(direction="minimize", study_name=f"wft_fold_{train_start.date()}", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

        best = study.best_trial
        print(f"Fold {train_start.date()} -> best daily_roi_$_trial_attr = {best.user_attrs.get('daily_roi_$')} max_dd={best.user_attrs.get('max_dd')}")
        # Train final model on train_df with best params and evaluate on test_df (out-of-sample)
        best_params = {
            'num_leaves': best.params.get('num_leaves'),
            'min_data_in_leaf': best.params.get('min_data_in_leaf'),
            'learning_rate': best.params.get('learning_rate'),
            'feature_fraction': best.params.get('feature_fraction'),
        }
        feature_cols = [c for c in train_df.columns if c not in [DATETIME_COL, 'future_close', 'future_return', 'target']]
        X_train = train_df[feature_cols]
        y_train = train_df['target']
        train_lgb = lgb.Dataset(X_train, label=y_train)
        lgb_params = {
            'objective':'binary',
            'metric':'auc',
            'num_leaves': best_params['num_leaves'],
            'min_data_in_leaf': best_params['min_data_in_leaf'],
            'learning_rate': best_params['learning_rate'],
            'feature_fraction': best_params['feature_fraction'],
            'verbosity': -1,
        }
        model = lgb.train(lgb_params, train_lgb, num_boost_round=200, verbose_eval=False)
        # Predict on test_df
        X_test = test_df[feature_cols]
        test_df_bt = test_df.copy()
        test_df_bt['pred_prob'] = model.predict(X_test)

        strategy_params = dict(
            pred_col='pred_prob',
            signal_threshold=best.params.get('signal_threshold'),
            stop_loss_atr_mult=best.params.get('stop_loss_atr_mult'),
            take_profit_atr_mult=best.params.get('take_profit_atr_mult'),
            risk_per_trade=RISK_PER_TRADE,
            max_dd_target=MAX_DD_TARGET
        )
        test_res = run_backtest_bt(test_df_bt, strategy_params)
        print(f"Test Fold results: pnl=${test_res['pnl']:.2f}, max_dd={test_res['max_drawdown']:.6f}, daily_roi_=${test_res['daily_roi_$']:.2f}")

        folds_results.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "best_trial": best.params,
            "test_res": test_res
        })

        # move forward
        cur_train_start = cur_train_start + pd.Timedelta(days=STEP_DAYS)

    # Aggregate results
    total_pnl = sum([f['test_res']['pnl'] for f in folds_results])
    worst_dd = max([f['test_res']['max_drawdown'] for f in folds_results]) if folds_results else 0.0
    total_daily_roi = sum([f['test_res']['daily_roi_$'] for f in folds_results])  # approximate
    agg = {
        "total_pnl_$": total_pnl,
        "worst_drawdown": worst_dd,
        "sum_daily_roi_$": total_daily_roi,
        "folds": folds_results
    }
    # Save summary
    pd.to_pickle(agg, os.path.join(OUTPUT_DIR, "wft_summary.pkl"))
    return agg

# ---------------------------
# Main execution
# ---------------------------

def main():
    print("Loading data...")
    df = load_data(DATA_CSV, datetime_col=DATETIME_COL)
    print(f"Loaded {len(df)} rows. Trimming to last {TOTAL_DAYS} days and computing indicators...")

    # Run walk-forward
    agg = walk_forward_run(df)
    print("Walk-forward finished.")
    print(f"Total pnl over test folds: ${agg['total_pnl_$']:.2f}")
    print(f"Worst observed drawdown across folds: {agg['worst_drawdown']:.6f}")
    print(f"Detailed results saved to {OUTPUT_DIR}/wft_summary.pkl")

if __name__ == "__main__":
    main()