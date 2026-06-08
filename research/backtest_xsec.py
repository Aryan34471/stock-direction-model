"""
Market-neutral long/short factor backtest for the cross-sectional model.

Reuses diag_panel.parquet (raw features + fwd_63), so no re-download. Each day the
ensemble scores the whole universe; we long the top quintile and short the bottom
quintile (equal weight) and measure the realised H-day forward spread. This is the
standard way to judge a cross-sectional signal - it isolates stock selection from
market direction.

NOTE: overlapping H-day windows, so the annualised figure is a rough point estimate
(autocorrelated daily obs -> wide confidence interval). Read the spread + decile
monotonicity, not the headline annualised number.
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

H = 63
PANEL = "data/processed_xgb/diag_panel.parquet"
fwd = f"fwd_{H}"

panel = pd.read_parquet(PANEL).dropna(subset=[fwd]).copy()
feature_cols = json.load(open("data/processed_xgb/feature_cols.json"))
top = json.load(open("top_features.json"))
idx = [feature_cols.index(f) for f in top]

xgb = pickle.load(open("xgb_model.pkl", "rb"))
try:
    xgb.set_params(device="cpu")
except Exception:
    pass
lr = pickle.load(open("lr_model.pkl", "rb"))
imp = SimpleImputer(strategy="constant", fill_value=0.5)
imp.fit(np.full((1, len(top)), 0.5))

# cross-sectional rank features per day, then score the universe
ranked = panel.groupby(level=0)[feature_cols].rank(pct=True)
X = np.nan_to_num(ranked.values.astype(np.float32), nan=0.5)
xp = xgb.predict_proba(X)[:, 1]
lp = lr.predict_proba(imp.transform(X[:, idx]))[:, 1]
panel["score"] = 0.55 * xp + 0.45 * lp


def ls_spread(df, q=0.2):
    def day(g):
        if len(g) < 10:
            return np.nan
        n = max(1, int(len(g) * q))
        s = g.sort_values("score")
        return s.tail(n)[fwd].mean() - s.head(n)[fwd].mean()
    return df.groupby(level=0).apply(day).dropna()


def quintile_profile(df):
    d = df.copy()
    d["q"] = df.groupby(level=0)["score"].transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop")
    )
    return (d.groupby("q")[fwd].mean() * 100).round(2)


for name, sl in [("FULL SAMPLE", panel), ("TEST (>=2024-06)", panel[panel.index >= "2024-06-01"])]:
    sp = ls_spread(sl)
    ann = sp.mean() / H * 252 * 100
    print(f"\n== {name} ==  days={len(sp)}")
    print(f"  mean {H}d long/short spread: {sp.mean()*100:+.2f}%  (positive on {(sp > 0).mean()*100:.0f}% of days)")
    print(f"  approx annualised:          {ann:+.1f}%  (overlapping, rough)")
    print(f"  mean {H}d fwd return by score quintile (0=lowest .. 4=highest):")
    print(f"    {quintile_profile(sl).to_dict()}")
