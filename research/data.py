"Cross-sectional data pipeline - labels stocks by relative performance vs the universe."

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import ta as ta_lib

warnings.filterwarnings("ignore")


# config
TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "CRM", "ORCL",
    "ADBE", "NOW", "INTU", "AMAT", "KLAC", "SNPS", "CDNS", "FTNT",
    # Finance
    "JPM", "GS", "BAC", "WFC", "MS", "BLK", "V", "MA",
    "TRV", "AON", "ICE", "CME", "CB",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY",
    "ELV", "CI", "HUM", "ISRG", "BSX", "TMO", "MDT",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC",
    # Consumer
    "WMT", "HD", "MCD", "KO", "PEP", "COST",
    "SBUX", "TGT", "TJX", "ROST", "YUM", "CMG", "DG",
    # Industrials
    "HON", "CAT", "ITW", "EMR", "ETN", "PH", "ROK",
    # Materials
    "SHW", "ECL", "APD",
    # Market context (proxy only, not a training target)
    "SPY",
]

START_DATE = "2015-01-01"
END_DATE = "2025-09-01"

TRAIN_END = "2023-01-01"
PURGE_END = "2023-07-01"
VAL_END = "2024-06-01"

HORIZON = 5

PROCESSED_DIR = "data/processed_xgb"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# the per-stock technical features the model trains on. These get converted to
# cross-sectional percentile ranks. Macro / analyst columns are deliberately
# excluded - they carry no cross-sectional (stock-vs-stock) information.
TECH_FEATURES = [
    "rsi_14", "rsi_7", "rsi_28",
    "macd", "macd_signal", "macd_hist",
    "roc_5", "roc_10", "roc_20",
    "close_vs_sma10", "close_vs_sma20", "close_vs_sma50", "close_vs_sma200",
    "sma10_vs_sma50", "sma20_vs_sma50", "ema_cross",
    "bb_pct", "bb_width", "atr_14",
    "realised_vol_5", "realised_vol_20", "vol_ratio",
    "volume_ratio", "volume_ratio_5",
    "log_ret_1d", "log_ret_3d", "log_ret_5d", "log_ret_10d", "log_ret_20d",
    "rsi_mean_14", "rsi_std_14", "rsi_trend",
    "ret_mean_10", "ret_mean_20", "ret_std_10", "ret_std_20", "ret_skew_20",
    "slope_5", "slope_10", "slope_20",
    "vol_trend", "high_low_range", "high_vs_close", "close_vs_low",
    "consec_up", "consec_down",
]


def download_prices(ticker):
    print(f"  [prices] {ticker}")
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index.name = "date"
    return df


def compute_features(df):
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


def compute_macro_features(spy_df, vix_df):
    """SPY/VIX regime features. Stored for later use (e.g. regime gating) but not
    part of the cross-sectional model - they are identical across all tickers."""
    macro = pd.DataFrame(index=spy_df.index)
    spy_close = spy_df["close"]

    macro["spy_ret_5d"] = np.log(spy_close / spy_close.shift(5))
    macro["spy_ret_20d"] = np.log(spy_close / spy_close.shift(20))
    spy_sma200 = ta_lib.trend.SMAIndicator(spy_close, window=200).sma_indicator()
    macro["spy_vs_sma200"] = (spy_close - spy_sma200) / spy_sma200
    macro["spy_rsi_14"] = ta_lib.momentum.RSIIndicator(spy_close, window=14).rsi()
    spy_log_ret = np.log(spy_close / spy_close.shift(1))
    macro["spy_vol_20"] = spy_log_ret.rolling(20).std() * np.sqrt(252)

    if vix_df is not None and not vix_df.empty:
        vix_close = vix_df["close"].reindex(macro.index, method="ffill")
        macro["vix_level"] = vix_close
        macro["vix_vs_ma20"] = vix_close / (vix_close.rolling(20).mean() + 1e-8) - 1
    else:
        macro["vix_level"] = np.nan
        macro["vix_vs_ma20"] = np.nan

    return macro


def add_forward_return(df):
    """HORIZON-day forward log return (the raw target before cross-sectional ranking)."""
    df["fwd_ret"] = np.log(df["close"].shift(-HORIZON) / df["close"])
    return df


def rank_features_cross_sectionally(combined, feature_cols):
    """Convert each feature to its per-day percentile rank across the universe.
    Uses .values (positional) to avoid alignment issues with the duplicated date index."""
    ranked = combined.groupby(level=0)[feature_cols].rank(pct=True)
    combined[feature_cols] = ranked.values
    return combined


def add_cross_sectional_label(combined, low=0.34, high=0.66):
    """Label = relative performance. Rank forward return across the universe each
    day; top tercile -> 1 (outperform), bottom tercile -> 0 (underperform), middle
    -> NaN (dropped). Ranking removes the common market move, so the label is purely
    about which stocks beat which."""
    rank_vals = combined.groupby(level=0)["fwd_ret"].rank(pct=True).values
    label = np.full(len(combined), np.nan)
    label[rank_vals >= high] = 1.0
    label[rank_vals <= low] = 0.0
    combined["fwd_rank"] = rank_vals
    combined["label"] = label
    return combined


def main():
    print("=" * 60)
    print("Data pipeline - cross-sectional (relative-performance) reframe")
    print("=" * 60)

    # SPY + VIX for macro columns (stored, not used as model features here)
    print("\nDownloading SPY + VIX for macro context...")
    spy_df = download_prices("SPY")
    try:
        vix_df = download_prices("^VIX")
        print(f"  VIX rows: {len(vix_df)}")
    except Exception as e:
        print(f"  VIX failed ({e}) - using NaN")
        vix_df = None
    macro_df = compute_macro_features(spy_df, vix_df)
    print(f"  Macro columns: {macro_df.shape[1]}")

    # per-ticker: technical features + forward return
    print("\nProcessing tickers...")
    all_raw = []
    for ticker in TICKERS:
        if ticker == "SPY":
            continue   # market proxy only, never a training target
        df = download_prices(ticker)
        if df.empty or len(df) < 250:
            print(f"  skip {ticker} (insufficient data)")
            continue
        df = compute_features(df)
        df = add_forward_return(df)
        df = df.join(macro_df, how="left", rsuffix="_macro")
        df["ticker"] = ticker
        # drop indicator warmup rows and the last HORIZON rows (no forward return)
        df.dropna(subset=["fwd_ret", "rsi_14", "macd", "log_ret_1d",
                          "slope_20", "rsi_mean_14"], inplace=True)
        all_raw.append(df)
        print(f"  {ticker}: {len(df)} rows")
        time.sleep(0.3)

    if not all_raw:
        print("No data downloaded.")
        return

    combined = pd.concat(all_raw).sort_index()
    print(f"\nCombined: {combined.shape[0]:,} rows, {combined['ticker'].nunique()} tickers")

    # cross-sectional feature ranking (per day, over the full universe)
    print("Ranking features cross-sectionally...")
    combined = rank_features_cross_sectionally(combined, TECH_FEATURES)

    # cross-sectional label
    combined = add_cross_sectional_label(combined)
    n_before = len(combined)
    combined.dropna(subset=["label"], inplace=True)
    print(f"Labeled rows: {len(combined):,}/{n_before:,} (middle tercile dropped)")
    print(f"Label balance: {combined['label'].mean()*100:.1f}% outperform")

    # split by date, with a purge gap between train and val
    train = combined[combined.index < TRAIN_END].copy()
    val = combined[(combined.index >= PURGE_END) & (combined.index < VAL_END)].copy()
    test = combined[combined.index >= VAL_END].copy()
    purged = combined[(combined.index >= TRAIN_END) & (combined.index < PURGE_END)]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train):,} rows  ({train.index.min().date()} -> {train.index.max().date()})")
    print(f"  Purge: {len(purged):,} rows discarded")
    print(f"  Val:   {len(val):,} rows  ({val.index.min().date()} -> {val.index.max().date()})")
    print(f"  Test:  {len(test):,} rows  ({test.index.min().date()} -> {test.index.max().date()})")

    # save
    train_path = os.path.join(PROCESSED_DIR, "train.parquet")
    val_path = os.path.join(PROCESSED_DIR, "val.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test.parquet")
    train.to_parquet(train_path)
    val.to_parquet(val_path)
    test.to_parquet(test_path)

    feature_path = os.path.join(PROCESSED_DIR, "feature_cols.json")
    with open(feature_path, "w") as f:
        json.dump(TECH_FEATURES, f, indent=2)

    print(f"\nSaved train/val/test parquet + feature_cols.json ({len(TECH_FEATURES)} features)")
    print("\nClass balance per split:")
    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"  {name:6s}: {df['label'].mean()*100:.1f}% outperform  ({len(df):,} rows)")

    print("\nDone. Features are cross-sectional ranks in [0, 1]; no separate")
    print("normalisation step is needed (ranking is the normalisation).")


if __name__ == "__main__":
    main()
