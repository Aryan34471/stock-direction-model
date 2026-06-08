"""Checks whether macro features predict the market's forward return, full-sample vs out-of-sample."""

import numpy as np
import pandas as pd
import data

H_LIST = [5, 21, 63]
TEST_START = "2024-06-01"

panel = pd.read_parquet("data/processed_xgb/diag_panel.parquet")
mkt = pd.DataFrame({H: panel.groupby(level=0)[f"fwd_{H}"].mean() for H in H_LIST})

print("Downloading SPY + VIX for macro features...")
spy = data.download_prices("SPY")
try:
    vix = data.download_prices("^VIX")
except Exception:
    vix = None
macro = data.compute_macro_features(spy, vix)
macro_cols = list(macro.columns)

df = macro.join(mkt, how="inner")

for name, sl in [("FULL SAMPLE", df), (f"TEST (>={TEST_START})", df[df.index >= TEST_START])]:
    print(f"\n== {name} ==  n={len(sl)}")
    for H in H_LIST:
        y = sl[H]
        cors = {c: sl[c].corr(y, method="spearman") for c in macro_cols}
        cors = pd.Series(cors).dropna()
        cors = cors.reindex(cors.abs().sort_values(ascending=False).index)
        print(f"  fwd {H:>2}d market return vs macro: "
              + ", ".join(f"{c}={v:+.3f}" for c, v in cors.head(5).items()))
