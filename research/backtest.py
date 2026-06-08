"""
Backtest for the cross-sectional XGBoost ensemble.

At each day, raw technical features are computed for the whole universe, then
ranked cross-sectionally (per day) into [0, 1] - the same transform used in
training. The model predicts P(outperform the universe); we go long the
highest-conviction names. Trading rules (ATR stops, 50MA filter, conviction-drop
exit, earnings avoidance) are unchanged from the baseline so the comparison
isolates the effect of the cross-sectional signal.

Saves trade + daily logs to backtest_results_xgb/.
"""

import os
import json
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import ta as ta_lib
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


# config
START_DATE = "2025-11-03"
END_DATE = datetime.today().strftime("%Y-%m-%d")
OUTPUT_DIR = "backtest_results_xgb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# model paths
XGB_MODEL_PATH = "xgb_model.pkl"
LR_MODEL_PATH = "lr_model.pkl"
TOP_FEATURES_PATH = "top_features.json"
FEAT_COLS_PATH = "data/processed_xgb/feature_cols.json"

# trading parameters
MIN_CONF = 0.55
EXTEND_THRESHOLD = 0.55
BASE_HOLD_DAYS = 1
MAX_HOLD_DAYS = 2
MAX_HOLD_DAYS_EXTENDED = 4
MAX_POSITIONS = 5
CONVICTION_DROP_THRESHOLD = 0.50
MIN_HOLD_DAYS_BEFORE_DROP = 1

# ensemble weights (must match XGB.py)
ENSEMBLE_XGB_WEIGHT = 0.55
ENSEMBLE_LR_WEIGHT = 0.45

POSITION_SIZE = 20_000   # flat $20k per position

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "CRM", "ORCL",
    "ADBE", "NOW", "INTU", "AMAT", "KLAC", "SNPS", "CDNS", "FTNT",
    "JPM", "GS", "BAC", "WFC", "MS", "BLK", "V", "MA",
    "TRV", "AON", "ICE", "CME", "CB",
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY",
    "ELV", "CI", "HUM", "ISRG", "BSX", "TMO", "MDT",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC",
    "WMT", "HD", "MCD", "KO", "PEP", "COST",
    "SBUX", "TGT", "TJX", "ROST", "YUM", "CMG", "DG",
    "HON", "CAT", "ITW", "EMR", "ETN", "PH", "ROK",
    "SHW", "ECL", "APD",
]


def get_trading_days(start, end):
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start, end=end)
    dates = pd.bdate_range(start=start, end=end)
    return [d for d in dates if d not in holidays]


# global feature cache, filled by preload_all_data()
FEATURE_CACHE = {}


def compute_features(df):
    """Identical to data.py compute_features()."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    df["rsi_14"] = ta_lib.momentum.RSIIndicator(close, window=14).rsi()
    df["rsi_7"] = ta_lib.momentum.RSIIndicator(close, window=7).rsi()
    df["rsi_28"] = ta_lib.momentum.RSIIndicator(close, window=28).rsi()

    macd = ta_lib.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["roc_5"] = ta_lib.momentum.ROCIndicator(close, window=5).roc()
    df["roc_10"] = ta_lib.momentum.ROCIndicator(close, window=10).roc()
    df["roc_20"] = ta_lib.momentum.ROCIndicator(close, window=20).roc()

    sma_10 = ta_lib.trend.SMAIndicator(close, window=10).sma_indicator()
    sma_20 = ta_lib.trend.SMAIndicator(close, window=20).sma_indicator()
    sma_50 = ta_lib.trend.SMAIndicator(close, window=50).sma_indicator()
    sma_200 = ta_lib.trend.SMAIndicator(close, window=200).sma_indicator()
    ema_12 = ta_lib.trend.EMAIndicator(close, window=12).ema_indicator()
    ema_26 = ta_lib.trend.EMAIndicator(close, window=26).ema_indicator()

    df["close_vs_sma10"] = (close - sma_10) / sma_10
    df["close_vs_sma20"] = (close - sma_20) / sma_20
    df["close_vs_sma50"] = (close - sma_50) / sma_50
    df["close_vs_sma200"] = (close - sma_200) / sma_200
    df["sma10_vs_sma50"] = (sma_10 - sma_50) / sma_50
    df["sma20_vs_sma50"] = (sma_20 - sma_50) / sma_50
    df["ema_cross"] = (ema_12 - ema_26) / close

    bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pct"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

    df["atr_14"] = ta_lib.volatility.AverageTrueRange(
        high, low, close, window=14).average_true_range() / close

    log_ret = np.log(close / close.shift(1))
    df["realised_vol_5"] = log_ret.rolling(5).std() * np.sqrt(252)
    df["realised_vol_20"] = log_ret.rolling(20).std() * np.sqrt(252)
    df["vol_ratio"] = df["realised_vol_5"] / (df["realised_vol_20"] + 1e-8)

    vol_sma20 = vol.rolling(20).mean()
    df["volume_ratio"] = vol / (vol_sma20 + 1e-8)
    df["volume_ratio_5"] = vol.rolling(5).mean() / (vol_sma20 + 1e-8)

    df["log_ret_1d"] = np.log(close / close.shift(1))
    df["log_ret_3d"] = np.log(close / close.shift(3))
    df["log_ret_5d"] = np.log(close / close.shift(5))
    df["log_ret_10d"] = np.log(close / close.shift(10))
    df["log_ret_20d"] = np.log(close / close.shift(20))

    df["rsi_mean_14"] = df["rsi_14"].rolling(14).mean()
    df["rsi_std_14"] = df["rsi_14"].rolling(14).std()
    df["rsi_trend"] = df["rsi_14"] - df["rsi_14"].shift(5)

    df["ret_mean_10"] = log_ret.rolling(10).mean()
    df["ret_mean_20"] = log_ret.rolling(20).mean()
    df["ret_std_10"] = log_ret.rolling(10).std()
    df["ret_std_20"] = log_ret.rolling(20).std()
    df["ret_skew_20"] = log_ret.rolling(20).skew()

    def price_slope(series, window):
        log_p = np.log(series)
        x = np.arange(window, dtype=float)
        x -= x.mean()
        x_sq = (x ** 2).sum()
        return log_p.rolling(window).apply(
            lambda y: np.nan if np.isnan(y).any() else (x * (y - y.mean())).sum() / x_sq,
            raw=True,
        )

    df["slope_5"] = price_slope(close, 5)
    df["slope_10"] = price_slope(close, 10)
    df["slope_20"] = price_slope(close, 20)

    df["vol_trend"] = df["volume_ratio"] - df["volume_ratio"].shift(5)
    df["high_low_range"] = (high - low) / close
    df["high_vs_close"] = (high - close) / close
    df["close_vs_low"] = (close - low) / close

    up_days = (log_ret > 0).astype(int)
    df["consec_up"] = up_days.groupby(
        (up_days != up_days.shift()).cumsum()
    ).cumcount() + 1
    df["consec_up"] = df["consec_up"] * up_days
    df["consec_down"] = ((log_ret < 0).astype(int)).groupby(
        ((log_ret < 0).astype(int) != (log_ret < 0).astype(int).shift()).cumsum()
    ).cumcount() + 1
    df["consec_down"] = df["consec_down"] * (log_ret < 0).astype(int)

    return df


def preload_all_data():
    """Download all ticker history and precompute features up front."""
    global FEATURE_CACHE
    print("\nPreloading historical data...")

    global_start = (
        datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=365)
    ).strftime("%Y-%m-%d")

    for ticker in TICKERS:
        try:
            df = yf.download(
                ticker, start=global_start, end=END_DATE,
                interval="1d", auto_adjust=True, progress=False,
            )
            if df.empty:
                print(f"  [data] {ticker} EMPTY")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            df.index.name = "date"
            FEATURE_CACHE[ticker] = compute_features(df)
        except Exception as e:
            print(f"  [data] {ticker} FAILED: {e}")

    print(f"Preloaded {len(FEATURE_CACHE)} tickers")


def get_atr_stop(df, entry_price, multiplier=2.0):
    try:
        atr = ta_lib.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=14
        ).average_true_range().iloc[-1]
        atr_stop = entry_price - (atr * multiplier)
        hard_stop = entry_price * 0.97
        return round(max(atr_stop, hard_stop), 2)
    except:
        return round(entry_price * 0.97, 2)


def is_above_50ma(df):
    try:
        sma_50 = ta_lib.trend.SMAIndicator(df["close"], window=50).sma_indicator()
        return float(df["close"].iloc[-1]) > float(sma_50.iloc[-1])
    except:
        return True


def get_earnings_dates(ticker):
    """Return set of earnings dates for this ticker (as date strings)."""
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is not None and "Earnings Date" in cal:
            dates = cal["Earnings Date"]
            if isinstance(dates, list):
                return {pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates}
            else:
                return {pd.Timestamp(dates).strftime("%Y-%m-%d")}
    except:
        pass
    return set()


def predict_prob(xgb_model, lr_model, imputer, feature_vector, top_feat_idx):
    """Ensemble probability for a single (already cross-sectionally ranked) vector."""
    x = feature_vector.reshape(1, -1)
    xgb_prob = xgb_model.predict_proba(x)[0][1]
    x_lr = imputer.transform(x[:, top_feat_idx])
    lr_prob = lr_model.predict_proba(x_lr)[0][1]
    return ENSEMBLE_XGB_WEIGHT * xgb_prob + ENSEMBLE_LR_WEIGHT * lr_prob


def get_raw_features(ticker, backtest_date, feature_cols):
    """Latest raw (un-ranked) feature row for a ticker up to backtest_date, plus
    close / 50MA / ATR-stop. Returns None if not enough history. Cross-sectional
    ranking happens later, once all tickers for the day are collected."""
    full_df = FEATURE_CACHE.get(ticker)
    if full_df is None or full_df.empty:
        return None
    df = full_df.loc[:backtest_date]
    if len(df) < 210:
        return None
    df = df.dropna(subset=["rsi_14", "macd", "slope_20", "rsi_mean_14"])
    if df.empty:
        return None
    latest = df.iloc[-1]
    close = float(df["close"].iloc[-1])
    raw = {col: (float(latest[col]) if col in latest.index and pd.notna(latest[col]) else np.nan)
           for col in feature_cols}
    return {
        "raw": raw,
        "close": close,
        "above_50ma": is_above_50ma(df),
        "stop_price": get_atr_stop(df, close),
    }


def predict_all_for_day(today, feature_cols, top_feat_idx, xgb_model, lr_model, imputer):
    """Build cross-sectionally ranked features for every ticker on `today` and
    return {ticker: {prob, close, above_50ma, stop_price}}."""
    raws = {}
    for ticker in TICKERS:
        r = get_raw_features(ticker, today, feature_cols)
        if r is not None:
            raws[ticker] = r
    if not raws:
        return {}

    # cross-sectional percentile rank per feature, across the universe today
    feat_df = pd.DataFrame({tk: raws[tk]["raw"] for tk in raws}).T[feature_cols]
    ranked = feat_df.rank(pct=True)

    results = {}
    for tk in raws:
        fv = np.nan_to_num(ranked.loc[tk].to_numpy(dtype=np.float32), nan=0.5)
        prob = predict_prob(xgb_model, lr_model, imputer, fv, top_feat_idx)
        results[tk] = {
            "prob": float(prob),
            "close": raws[tk]["close"],
            "above_50ma": raws[tk]["above_50ma"],
            "stop_price": raws[tk]["stop_price"],
        }
    return results


def main():
    print("=" * 70)
    print("  Cross-sectional XGBoost Backtest")
    print(f"  {START_DATE} -> {END_DATE}")
    print("=" * 70)

    # load models
    print("\nLoading models...")
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    try:
        xgb_model.set_params(device="cpu")   # predict on CPU regardless of train device
    except Exception:
        pass
    with open(LR_MODEL_PATH, "rb") as f:
        lr_model = pickle.load(f)
    with open(TOP_FEATURES_PATH) as f:
        top_features = json.load(f)
    with open(FEAT_COLS_PATH) as f:
        feature_cols = json.load(f)

    top_feat_idx = [feature_cols.index(f) for f in top_features]

    # features are cross-sectional ranks; any residual NaN -> neutral, so the
    # imputer is effectively a no-op but kept for the LR transform interface.
    imputer = SimpleImputer(strategy="constant", fill_value=0.5)
    imputer.fit(np.full((1, len(top_features)), 0.5))

    print(f"  XGB loaded ({len(feature_cols)} features), LR top {len(top_features)}")

    preload_all_data()

    # earnings dates for the exit rule
    print("\nFetching earnings dates...")
    earnings_cache = {}
    for ticker in TICKERS:
        earnings_cache[ticker] = get_earnings_dates(ticker)
        time.sleep(0.1)
    print(f"  Done ({len(earnings_cache)} tickers)")

    trading_days = get_trading_days(START_DATE, END_DATE)
    print(f"\nTrading days: {len(trading_days)}")

    # state
    open_trades = []
    cooldown_tickers = {}
    trade_log = []
    daily_log = []

    print(f"\n{'='*70}\n  RUNNING BACKTEST\n{'='*70}\n")

    all_results = {}
    for day in trading_days:
        today = day.strftime("%Y-%m-%d")
        today_dt = pd.Timestamp(today)

        all_results = predict_all_for_day(
            today, feature_cols, top_feat_idx, xgb_model, lr_model, imputer
        )
        print(f"{today} -> {len(all_results)} valid tickers")
        if not all_results:
            continue

        # check exits
        to_keep = []
        for trade in open_trades:
            entry_dt = pd.Timestamp(trade["entry_date"])
            days_held = len(pd.bdate_range(entry_dt, today_dt)) - 1
            ticker = trade["ticker"]
            result = all_results.get(ticker)
            current_prob = result["prob"] if result else 0.5
            current_price = result["close"] if result else trade["entry_price"]
            stop_price = trade.get("stop_price", 0)

            exit_reason = None

            # earnings exit - sell the day before earnings
            next_bday = (today_dt + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
            if next_bday in earnings_cache.get(ticker, set()):
                exit_reason = "EARNINGS EXIT"
            elif stop_price and current_price <= stop_price:
                exit_reason = "STOP LOSS"
            elif days_held >= MIN_HOLD_DAYS_BEFORE_DROP and current_prob < CONVICTION_DROP_THRESHOLD:
                exit_reason = "CONVICTION DROP"
            else:
                max_days = trade.get("max_days", BASE_HOLD_DAYS)
                if current_prob >= EXTEND_THRESHOLD and days_held >= BASE_HOLD_DAYS - 1:
                    new_max = min(days_held + 2, MAX_HOLD_DAYS_EXTENDED)
                    if new_max > max_days:
                        trade["max_days"] = new_max
                        trade["extended"] = True
                if days_held >= trade.get("max_days", BASE_HOLD_DAYS):
                    exit_reason = "TIME EXIT"

            if exit_reason:
                raw_pnl_pct = (current_price - trade["entry_price"]) / trade["entry_price"] * 100
                if exit_reason == "STOP LOSS" and raw_pnl_pct < -3.0:
                    pnl_pct = -3.0
                else:
                    pnl_pct = raw_pnl_pct
                pnl = pnl_pct / 100 * trade["position_size"]
                costs = trade["position_size"] * 0.0003 * 2 + 10
                net_pnl = pnl - costs

                trade_log.append({
                    "action": "SELL", "date": today, "ticker": ticker,
                    "days_held": days_held, "entry_date": trade["entry_date"],
                    "entry_price": trade["entry_price"], "exit_price": round(current_price, 2),
                    "position_size": trade["position_size"], "gross_pnl": round(pnl, 2),
                    "costs": round(costs, 2), "net_pnl": round(net_pnl, 2),
                    "pnl_pct": round(pnl_pct, 2), "exit_reason": exit_reason,
                    "prob_at_entry": trade["prob"], "prob_at_exit": round(current_prob, 4),
                    "extended": trade.get("extended", False),
                    "result": "WIN" if net_pnl > 0 else "LOSS",
                })

                if exit_reason == "STOP LOSS":
                    cooldown_tickers[ticker] = (
                        today_dt + pd.tseries.offsets.BDay(5)
                    ).strftime("%Y-%m-%d")

                print(f"  {today}  SELL  {ticker:<6}  {exit_reason:<16}  "
                      f"entry=${trade['entry_price']:.2f}  exit=${current_price:.2f}  "
                      f"P&L=${net_pnl:+.2f} ({pnl_pct:+.1f}%)")
            else:
                trade_log.append({
                    "action": "HOLD", "date": today, "ticker": ticker,
                    "days_held": days_held, "entry_date": trade["entry_date"],
                    "entry_price": trade["entry_price"], "exit_price": round(current_price, 2),
                    "position_size": trade["position_size"], "gross_pnl": "", "costs": "",
                    "net_pnl": "", "pnl_pct": round((current_price - trade["entry_price"]) / trade["entry_price"] * 100, 2),
                    "exit_reason": "", "prob_at_entry": trade["prob"],
                    "prob_at_exit": round(current_prob, 4), "extended": trade.get("extended", False),
                    "result": "",
                })
                to_keep.append(trade)

        open_trades = to_keep

        cooldown_tickers = {
            t: until for t, until in cooldown_tickers.items()
            if pd.Timestamp(until) > today_dt
        }

        # find new entries
        held_tickers = {t["ticker"] for t in open_trades}
        slots_available = MAX_POSITIONS - len(open_trades)

        candidates = {}
        for ticker, result in all_results.items():
            if result["prob"] < MIN_CONF:
                continue
            if ticker in held_tickers or ticker in cooldown_tickers:
                continue
            if not result["above_50ma"]:
                continue
            candidates[ticker] = result

        ranked_cands = sorted(candidates.items(), key=lambda x: x[1]["prob"], reverse=True)

        for ticker, result in ranked_cands[:slots_available]:
            prob = result["prob"]
            entry_price = result["close"]
            stop_price = result["stop_price"]

            open_trades.append({
                "ticker": ticker, "entry_date": today, "position_size": POSITION_SIZE,
                "prob": round(prob, 4), "entry_price": round(entry_price, 2),
                "stop_price": stop_price, "extended": False, "max_days": BASE_HOLD_DAYS,
            })

            trade_log.append({
                "action": "BUY", "date": today, "ticker": ticker, "days_held": 0,
                "entry_date": today, "entry_price": round(entry_price, 2), "exit_price": "",
                "position_size": POSITION_SIZE, "gross_pnl": "", "costs": "", "net_pnl": "",
                "pnl_pct": "", "exit_reason": "", "prob_at_entry": round(prob, 4),
                "prob_at_exit": "", "extended": False, "result": "",
            })

            print(f"  {today}  BUY   {ticker:<6}  P={prob:.3f}  "
                  f"${POSITION_SIZE:,}  entry~${entry_price:.2f}  stop=${stop_price:.2f}")

        # daily summary
        day_sells = [r for r in trade_log if r["date"] == today and r["action"] == "SELL"]
        day_pnl = sum(r["net_pnl"] for r in day_sells if isinstance(r["net_pnl"], (int, float)))
        daily_log.append({
            "date": today, "open_positions": len(open_trades),
            "tickers_held": ", ".join(t["ticker"] for t in open_trades),
            "new_buys": sum(1 for r in trade_log if r["date"] == today and r["action"] == "BUY"),
            "sells": len(day_sells), "day_pnl": round(day_pnl, 2),
        })

    # close any remaining open trades at the last price
    last_day = trading_days[-1].strftime("%Y-%m-%d")
    for trade in open_trades:
        ticker = trade["ticker"]
        last_result = all_results.get(ticker)
        current_price = last_result["close"] if last_result else trade["entry_price"]
        pnl_pct = (current_price - trade["entry_price"]) / trade["entry_price"] * 100
        pnl = pnl_pct / 100 * trade["position_size"]
        costs = trade["position_size"] * 0.0003 * 2 + 10
        net_pnl = pnl - costs
        trade_log.append({
            "action": "SELL", "date": last_day, "ticker": ticker, "days_held": 0,
            "entry_date": trade["entry_date"], "entry_price": trade["entry_price"],
            "exit_price": round(current_price, 2), "position_size": trade["position_size"],
            "gross_pnl": round(pnl, 2), "costs": round(costs, 2), "net_pnl": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct, 2), "exit_reason": "END OF BACKTEST",
            "prob_at_entry": trade["prob"], "prob_at_exit": "",
            "extended": trade.get("extended", False),
            "result": "WIN" if net_pnl > 0 else "LOSS",
        })

    # save
    trade_df = pd.DataFrame(trade_log)
    daily_df = pd.DataFrame(daily_log)
    trade_path = os.path.join(OUTPUT_DIR, "backtest_trade_log.csv")
    daily_path = os.path.join(OUTPUT_DIR, "backtest_daily_log.csv")
    trade_df.to_csv(trade_path, index=False)
    daily_df.to_csv(daily_path, index=False)

    if trade_df.empty:
        print("\nNo trades generated.")
        return

    sells = trade_df[trade_df["action"] == "SELL"].copy()
    sells["net_pnl"] = pd.to_numeric(sells["net_pnl"], errors="coerce")
    wins = sells[sells["net_pnl"] > 0]
    losses = sells[sells["net_pnl"] <= 0]
    total_pnl = sells["net_pnl"].sum()
    daily_df["cumulative_pnl"] = daily_df["day_pnl"].cumsum()

    print(f"\n{'='*70}")
    print(f"  BACKTEST COMPLETE - cross-sectional ensemble")
    print(f"{'='*70}")
    print(f"  Period:        {START_DATE} -> {END_DATE}")
    print(f"  Trading days:  {len(trading_days)}")
    print(f"  Total trades:  {len(sells)}")
    if len(wins):
        print(f"  Wins:          {len(wins)}  (avg ${wins['net_pnl'].mean():.2f})")
    if len(losses):
        print(f"  Losses:        {len(losses)}  (avg ${losses['net_pnl'].mean():.2f})")
    if len(sells):
        print(f"  Win rate:      {len(wins)/len(sells)*100:.1f}%")
    print(f"  Total net P&L: ${total_pnl:+,.2f}")

    print(f"\n  Exit breakdown:")
    for reason, grp in sells.groupby("exit_reason"):
        print(f"    {reason:<18} {len(grp):3d} trades  avg P&L ${grp['net_pnl'].mean():+.2f}")

    print(f"\n  Trade log -> {trade_path}")
    print(f"  Daily log -> {daily_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
