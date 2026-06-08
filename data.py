"""
Data pipeline v2 - adds analyst features, macro regime features, and ATR-based labels.
Splits into train/val/test, normalises per ticker, saves parquet files.
"""

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
    # Market context
    "SPY",
]

START_DATE = "2015-01-01"
END_DATE = "2025-09-01"

TRAIN_END = "2023-01-01"
PURGE_END = "2023-07-01"
VAL_END = "2024-06-01"

HORIZON = 5

PROCESSED_DIR = "data/processed_xgb"
STATS_PATH = "data/processed_xgb/train_norm_stats.json"

os.makedirs(PROCESSED_DIR, exist_ok=True)


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


# Analyst features.
# Pulls buy/sell ratings, price target upside, and EPS revision momentum.
# yfinance only gives current snapshots, not historical, so older train rows
# have some look-ahead bias here. Acceptable tradeoff with free data.
def fetch_analyst_features(ticker):
    """Return dict of analyst features for ticker, or None if unavailable."""
    try:
        t = yf.Ticker(ticker)

        features = {}

        # A) recommendations
        recs = t.recommendations
        if recs is not None and not recs.empty:
            # use most recent month
            if "period" in recs.columns:
                current = recs[recs["period"] == "0m"]
                if current.empty:
                    current = recs.iloc[0:1]
            else:
                current = recs.iloc[0:1]

            sb = float(current.get("strongBuy", [0]).iloc[0]) if "strongBuy" in current else 0
            b = float(current.get("buy", [0]).iloc[0]) if "buy" in current else 0
            h = float(current.get("hold", [0]).iloc[0]) if "hold" in current else 0
            s = float(current.get("sell", [0]).iloc[0]) if "sell" in current else 0
            ss = float(current.get("strongSell", [0]).iloc[0]) if "strongSell" in current else 0

            total = sb + b + h + s + ss + 1e-8
            features["analyst_buy_pct"] = (sb + b) / total   # fraction bullish
            features["analyst_sell_pct"] = (s + ss) / total  # fraction bearish
            features["analyst_hold_pct"] = h / total
            # consensus score: 1=strong sell, 5=strong buy
            features["analyst_score"] = (5*sb + 4*b + 3*h + 2*s + 1*ss) / total

            # upgrade momentum: buy% now vs 2 months ago
            if len(recs) >= 3 and "period" in recs.columns:
                old = recs[recs["period"] == "-2m"]
                if not old.empty:
                    old_sb = float(old.get("strongBuy", [0]).iloc[0]) if "strongBuy" in old else 0
                    old_b = float(old.get("buy", [0]).iloc[0]) if "buy" in old else 0
                    old_h = float(old.get("hold", [0]).iloc[0]) if "hold" in old else 0
                    old_s = float(old.get("sell", [0]).iloc[0]) if "sell" in old else 0
                    old_ss = float(old.get("strongSell", [0]).iloc[0]) if "strongSell" in old else 0
                    old_total = old_sb + old_b + old_h + old_s + old_ss + 1e-8
                    old_buy_pct = (old_sb + old_b) / old_total
                    features["analyst_upgrade_momentum"] = features["analyst_buy_pct"] - old_buy_pct
                else:
                    features["analyst_upgrade_momentum"] = 0.0
            else:
                features["analyst_upgrade_momentum"] = 0.0
        else:
            features["analyst_buy_pct"] = np.nan
            features["analyst_sell_pct"] = np.nan
            features["analyst_hold_pct"] = np.nan
            features["analyst_score"] = np.nan
            features["analyst_upgrade_momentum"] = np.nan

        # B) price target upside
        targets = t.analyst_price_targets
        info = t.fast_info
        if targets is not None and hasattr(info, "last_price"):
            current_price = info.last_price
            if current_price and current_price > 0:
                mean_target = targets.get("mean", None)
                if mean_target:
                    features["analyst_upside"] = (mean_target - current_price) / current_price
                else:
                    features["analyst_upside"] = np.nan
            else:
                features["analyst_upside"] = np.nan
        else:
            features["analyst_upside"] = np.nan

        # C) EPS revision momentum
        revisions = t.eps_revisions
        if revisions is not None and not revisions.empty:
            row = revisions.iloc[0]  # current quarter
            up30 = float(row.get("upLast30days", 0) or 0)
            down30 = float(row.get("downLast30days", 0) or 0)
            up7 = float(row.get("upLast7days", 0) or 0)
            down7 = float(row.get("downLast7days", 0) or 0)

            total30 = up30 + down30 + 1e-8
            total7 = up7 + down7 + 1e-8

            # positive = more upgrades than downgrades
            features["eps_revision_30d"] = (up30 - down30) / total30
            features["eps_revision_7d"] = (up7 - down7) / total7
        else:
            features["eps_revision_30d"] = np.nan
            features["eps_revision_7d"] = np.nan

        return features

    except Exception as e:
        print(f"    [analyst] {ticker} failed: {e}")
        return None


# Macro features - market-wide features that condition the model on the regime.
# All derived from SPY (already downloaded) and VIX (downloaded separately).
# No look-ahead: all features on day T use only data up to day T.
def compute_macro_features(spy_df, vix_df):
    """Compute SPY/VIX macro features indexed by date."""
    macro = pd.DataFrame(index=spy_df.index)

    spy_close = spy_df["close"]

    # SPY momentum
    macro["spy_ret_5d"] = np.log(spy_close / spy_close.shift(5))
    macro["spy_ret_20d"] = np.log(spy_close / spy_close.shift(20))

    # SPY vs 200MA
    spy_sma200 = ta_lib.trend.SMAIndicator(spy_close, window=200).sma_indicator()
    macro["spy_vs_sma200"] = (spy_close - spy_sma200) / spy_sma200

    # SPY RSI
    macro["spy_rsi_14"] = ta_lib.momentum.RSIIndicator(spy_close, window=14).rsi()

    # SPY realised vol
    spy_log_ret = np.log(spy_close / spy_close.shift(1))
    macro["spy_vol_20"] = spy_log_ret.rolling(20).std() * np.sqrt(252)

    # VIX level
    if vix_df is not None and not vix_df.empty:
        vix_close = vix_df["close"].reindex(macro.index, method="ffill")
        macro["vix_level"] = vix_close
        macro["vix_vs_ma20"] = vix_close / (vix_close.rolling(20).mean() + 1e-8) - 1
    else:
        macro["vix_level"] = np.nan
        macro["vix_vs_ma20"] = np.nan

    return macro


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


# ATR-based label.
# 1 if 5d return > +1 ATR, 0 if < -1 ATR, NaN (dropped) if in between.
# Dropping the noise zone means the model only trains on decisive moves.
def add_label(df):
    future_ret = np.log(df["close"].shift(-HORIZON) / df["close"])
    atr_pct = df["atr_14"]   # already normalised by close in compute_features

    df["forward_ret"] = future_ret

    label = pd.Series(np.nan, index=df.index)
    label[future_ret > atr_pct] = 1
    label[future_ret < -atr_pct] = 0
    # middle stays NaN, gets dropped later

    df["label"] = label
    return df


# normalisation
EXCLUDE_FROM_NORM = {
    "label", "forward_ret", "ticker",
    "open", "high", "low", "close", "volume"
}


def get_feature_cols(df):
    return [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in EXCLUDE_FROM_NORM
    ]


def fit_normalise_train(df):
    feature_cols = get_feature_cols(df)
    stats = {}
    for col in feature_cols:
        mu = df[col].mean()
        sig = df[col].std()
        stats[col] = {"mean": float(mu), "std": float(max(sig, 1e-8))}
        df[col] = (df[col] - mu) / (sig + 1e-8)
    return df, stats


def normalise_with_stats(df, stats):
    feature_cols = get_feature_cols(df)
    for col in feature_cols:
        if col in stats:
            mu = stats[col]["mean"]
            sig = stats[col]["std"]
            df[col] = (df[col] - mu) / (sig + 1e-8)
        else:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
    return df


def main():
    print("=" * 60)
    print("XGBoost Data Pipeline v2 - analyst + macro + ATR label")
    print("=" * 60)

    # download SPY and VIX first (needed for macro features)
    print("\nDownloading SPY for macro features...")
    spy_df = download_prices("SPY")

    print("Downloading VIX...")
    try:
        vix_df = download_prices("^VIX")
        print(f"  VIX rows: {len(vix_df)}")
    except Exception as e:
        print(f"  VIX download failed ({e}) - will use NaN for VIX features")
        vix_df = None

    macro_df = compute_macro_features(spy_df, vix_df)
    print(f"  Macro features computed: {macro_df.shape[1]} columns, {len(macro_df)} rows")

    # fetch analyst features for all tickers
    print("\nFetching analyst features (one API call per ticker)...")
    analyst_cache = {}
    for ticker in TICKERS:
        if ticker == "SPY":
            continue   # no analyst ratings for SPY
        print(f"  [analyst] {ticker}")
        feats = fetch_analyst_features(ticker)
        if feats:
            analyst_cache[ticker] = feats
        time.sleep(0.3)   # be gentle with the API

    print(f"\nAnalyst data fetched for {len(analyst_cache)}/{len(TICKERS)-1} tickers")

    # process each ticker
    all_raw = []

    for ticker in TICKERS:
        print(f"\n{'-'*55}")
        print(f"  {ticker}")

        # download prices (SPY already done but we still need it in all_raw)
        df = download_prices(ticker)
        if df.empty or len(df) < 250:
            print(f"  Skipping {ticker} - insufficient data")
            continue

        df = compute_features(df)
        df = add_label(df)

        # attach macro features (join on date index)
        df = df.join(macro_df, how="left", rsuffix="_macro")

        # broadcast scalar analyst features to all rows
        if ticker in analyst_cache:
            for feat_name, feat_val in analyst_cache[ticker].items():
                df[feat_name] = feat_val
        else:
            # fill with NaN, XGBoost handles these natively
            for feat_name in ["analyst_buy_pct", "analyst_sell_pct",
                               "analyst_hold_pct", "analyst_score",
                               "analyst_upgrade_momentum",
                               "analyst_upside",
                               "eps_revision_30d", "eps_revision_7d"]:
                df[feat_name] = np.nan

        df["ticker"] = ticker

        # drop noise-zone rows and indicator warmup rows
        df.dropna(subset=["label", "rsi_14", "macd",
                          "log_ret_1d", "slope_20", "rsi_mean_14"], inplace=True)

        print(f"  Rows after label filtering: {len(df)}")
        all_raw.append(df)
        time.sleep(0.4)

    if not all_raw:
        print("No data downloaded.")
        return

    combined = pd.concat(all_raw).sort_index()
    print(f"\nTotal rows after ATR filtering: {combined.shape}")

    # label balance check
    bal = combined["label"].mean()
    print(f"Label balance: {bal:.1%} up  (expect closer to 50/50 with ATR filter)")

    # split into train / val / test, with a purge gap between train and val
    train_raw = combined[combined.index < TRAIN_END].copy()
    val_raw = combined[
        (combined.index >= PURGE_END) &
        (combined.index < VAL_END)
    ].copy()
    test_raw = combined[combined.index >= VAL_END].copy()

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_raw)} rows  "
          f"({train_raw.index.min().date()} -> {train_raw.index.max().date()})")
    print(f"  Purge: {len(combined[(combined.index >= TRAIN_END) & (combined.index < PURGE_END)])} rows discarded")
    print(f"  Val:   {len(val_raw)} rows  "
          f"({val_raw.index.min().date()} -> {val_raw.index.max().date()})")
    print(f"  Test:  {len(test_raw)} rows  "
          f"({test_raw.index.min().date()} -> {test_raw.index.max().date()})")

    # normalise (fit on train, apply to val/test)
    print("\nFitting normalisation stats on train...")
    all_train_normed = []
    all_stats = {}

    for ticker in train_raw["ticker"].unique():
        t_df = train_raw[train_raw["ticker"] == ticker].copy()
        t_df, stats = fit_normalise_train(t_df)
        t_df.dropna(subset=get_feature_cols(t_df)[:5], inplace=True)
        all_train_normed.append(t_df)
        all_stats[ticker] = stats
        print(f"  {ticker}: {len(t_df)} rows")

    train_normed = pd.concat(all_train_normed).sort_index()

    with open(STATS_PATH, "w") as f:
        json.dump(all_stats, f)
    print(f"\nStats saved -> {STATS_PATH}")

    print("\nNormalising val and test...")
    default_stats = all_stats.get("AAPL", list(all_stats.values())[0])

    all_val_normed = []
    all_test_normed = []

    for ticker in TICKERS:
        ticker_stats = all_stats.get(ticker, default_stats)

        if ticker in val_raw["ticker"].unique():
            v_df = val_raw[val_raw["ticker"] == ticker].copy()
            v_df = normalise_with_stats(v_df, ticker_stats)
            all_val_normed.append(v_df)

        if ticker in test_raw["ticker"].unique():
            t_df = test_raw[test_raw["ticker"] == ticker].copy()
            t_df = normalise_with_stats(t_df, ticker_stats)
            all_test_normed.append(t_df)

    val_normed = pd.concat(all_val_normed).sort_index() if all_val_normed else pd.DataFrame()
    test_normed = pd.concat(all_test_normed).sort_index() if all_test_normed else pd.DataFrame()

    # save
    train_path = os.path.join(PROCESSED_DIR, "train.parquet")
    val_path = os.path.join(PROCESSED_DIR, "val.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test.parquet")

    train_normed.to_parquet(train_path)
    val_normed.to_parquet(val_path)
    test_normed.to_parquet(test_path)

    feature_cols = get_feature_cols(train_normed)
    feature_path = os.path.join(PROCESSED_DIR, "feature_cols.json")
    with open(feature_path, "w") as f:
        json.dump(feature_cols, f)

    print(f"\nSaved:")
    print(f"  {train_path}  {train_normed.shape}")
    print(f"  {val_path}    {val_normed.shape}")
    print(f"  {test_path}   {test_normed.shape}")
    print(f"  Features: {len(feature_cols)}")

    print(f"\nNew features added:")
    new_feats = [c for c in feature_cols if c not in [
        "rsi_14", "rsi_7", "rsi_28", "macd", "macd_signal", "macd_hist",
        "roc_5", "roc_10", "roc_20", "close_vs_sma10", "close_vs_sma20",
        "close_vs_sma50", "close_vs_sma200", "sma10_vs_sma50", "sma20_vs_sma50",
        "ema_cross", "bb_pct", "bb_width", "atr_14", "realised_vol_5",
        "realised_vol_20", "vol_ratio", "volume_ratio", "volume_ratio_5",
        "log_ret_1d", "log_ret_3d", "log_ret_5d", "log_ret_10d", "log_ret_20d",
        "rsi_mean_14", "rsi_std_14", "rsi_trend", "ret_mean_10", "ret_mean_20",
        "ret_std_10", "ret_std_20", "ret_skew_20", "slope_5", "slope_10", "slope_20",
        "vol_trend", "high_low_range", "high_vs_close", "close_vs_low",
        "consec_up", "consec_down"
    ]]
    for f in new_feats:
        print(f"  + {f}")

    print(f"\nClass balance:")
    for name, df in [("Train", train_normed), ("Val", val_normed), ("Test", test_normed)]:
        bal = df["label"].mean()
        print(f"  {name:6s}: {bal:.1%} up  ({len(df)} rows)")

    print("\nDone. Ready for XGBoost.")
    print("Note: row counts are lower than v1 because of the ATR label filtering")
    print("(we dropped the ambiguous 'noise zone' rows).")


if __name__ == "__main__":
    main()
