"""
Diagnostic for recommendation #1: does the macro/vol feature set predict the MARKET
(time-series), and does it persist out-of-sample? This is the "market timing"
hypothesis - the only signal that survived OOS in the cross-sectional work.

Market proxy = equal-weight universe forward return (mean of fwd_H across tickers
per day, from the cached panel). Macro features come from SPY/VIX. We report the
time-series Spearman correlation of each macro feature with the forward market
return, full-sample vs test period.

CAVEAT: daily obs with overlapping H-day forward windows are highly autocorrelated,
so these correlations have wide confidence intervals - read them as directional
evidence, not precise estimates. Market timing genuinely needs more independent
data (decades, or higher frequency) to validate; this is a first look only.
"""

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
