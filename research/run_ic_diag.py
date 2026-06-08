"""
Diagnostic: does cross-sectional signal exist in these features, and at what horizon?

Computes the cross-sectional rank Information Coefficient (IC) of each technical
feature vs forward returns at several horizons. Rank IC here = correlation between
per-day cross-sectional feature ranks and per-day forward-return ranks (pooled),
a fast proxy for the mean daily Spearman IC.

Rough reading:
  |IC| ~ 0.00-0.01  -> noise, no usable cross-sectional signal
  |IC| ~ 0.02-0.05  -> weak but real signal (typical for equity factors)
  |IC| > 0.05       -> strong (rare on liquid large caps)

Downloads once and caches the panel to diag_panel.parquet for reuse.
"""

import numpy as np
import pandas as pd
import data   # reuse download_prices / compute_features / TICKERS / TECH_FEATURES

HORIZONS = [5, 10, 21, 63]
TEST_START = "2024-06-01"
PANEL_PATH = "data/processed_xgb/diag_panel.parquet"


def build_panel():
    frames = []
    for tk in data.TICKERS:
        if tk == "SPY":
            continue
        df = data.download_prices(tk)
        if df.empty or len(df) < 300:
            continue
        df = data.compute_features(df)
        for H in HORIZONS:
            df[f"fwd_{H}"] = np.log(df["close"].shift(-H) / df["close"])
        df["ticker"] = tk
        df = df.dropna(subset=["rsi_14", "macd", "slope_20", "rsi_mean_14"])
        frames.append(df)
    panel = pd.concat(frames).sort_index()
    panel.to_parquet(PANEL_PATH)
    return panel


def rank_ic(panel, feats, horizons):
    franks = {f: panel.groupby(level=0)[f].rank(pct=True) for f in feats}
    out = {}
    for H in horizons:
        fwd_rank = panel.groupby(level=0)[f"fwd_{H}"].rank(pct=True)
        ics = {}
        for f in feats:
            a, b = franks[f], fwd_rank
            m = a.notna() & b.notna()
            ics[f] = np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 100 else np.nan
        out[H] = pd.Series(ics)
    return pd.DataFrame(out)


def main():
    print("Building panel (downloading)...")
    panel = build_panel()
    print(f"Panel: {panel.shape[0]:,} rows, {panel['ticker'].nunique()} tickers")
    test = panel[panel.index >= TEST_START]
    print(f"Test slice (>= {TEST_START}): {test.shape[0]:,} rows")

    for name, p in [("FULL SAMPLE", panel), ("TEST PERIOD", test)]:
        ic = rank_ic(p, data.TECH_FEATURES, HORIZONS)
        print(f"\n===== {name}: cross-sectional rank IC =====")
        print("mean |IC| by horizon:")
        print(ic.abs().mean().round(4).to_string())
        print("best |IC| by horizon:")
        print(ic.abs().max().round(4).to_string())
        for H in HORIZONS:
            top = ic[H].reindex(ic[H].abs().sort_values(ascending=False).index).head(6)
            print(f"  H={H:>2} top: " + ", ".join(f"{f}={v:+.3f}" for f, v in top.items()))


if __name__ == "__main__":
    main()
