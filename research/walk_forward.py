"""Walk-forward test - retrains monthly on a rolling window to check if rolling retraining recovers OOS signal."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import data

H = 63
PANEL = "data/processed_xgb/diag_panel.parquet"
TEST_START = pd.Timestamp("2024-06-01")
TRAIN_WINDOW = 756   # ~3 years of trading days
PURGE = H            # gap between trainable labels and the prediction date
STEP = 21            # rebalance ~monthly
feats = data.TECH_FEATURES

panel = pd.read_parquet(PANEL)
# per-day cross-sectional feature ranks (each day independent -> no leakage)
panel[feats] = panel.groupby(level=0)[feats].rank(pct=True).values

dates = np.array(sorted(panel.index.unique()))
date_pos = {d: i for i, d in enumerate(dates)}
test_dates = [d for d in dates if d >= TEST_START]

params = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, tree_method="hist", device="cuda",
              eval_metric="auc", random_state=42, verbosity=0)


def ls_spread_day(g, q=0.2):
    if len(g) < 10:
        return np.nan
    n = max(1, int(len(g) * q))
    s = g.sort_values("score")
    return s.tail(n)[f"fwd_{H}"].mean() - s.head(n)[f"fwd_{H}"].mean()


spreads = []
n_steps = 0
i = 0
while i < len(test_dates):
    D = test_dates[i]
    pos = date_pos[D]
    train_end = pos - PURGE
    if train_end < TRAIN_WINDOW:
        i += STEP
        continue

    train_dates = set(dates[train_end - TRAIN_WINDOW:train_end])
    tr = panel[panel.index.isin(train_dates)].dropna(subset=[f"fwd_{H}"]).copy()
    r = tr.groupby(level=0)[f"fwd_{H}"].rank(pct=True).values
    y = np.full(len(tr), np.nan)
    y[r >= 0.66] = 1.0
    y[r <= 0.34] = 0.0
    m = ~np.isnan(y)
    if m.sum() < 2000:
        i += STEP
        continue
    Xtr = np.nan_to_num(tr.loc[m, feats].values.astype(np.float32), nan=0.5)
    ytr = y[m].astype(int)

    model = XGBClassifier(**params)
    model.fit(Xtr, ytr)

    pred_dates = set(test_dates[i:i + STEP])
    pr = panel[panel.index.isin(pred_dates)].dropna(subset=[f"fwd_{H}"]).copy()
    if len(pr):
        Xpr = np.nan_to_num(pr[feats].values.astype(np.float32), nan=0.5)
        pr["score"] = model.predict_proba(Xpr)[:, 1]
        sp = pr.groupby(level=0).apply(ls_spread_day).dropna()
        spreads.extend(sp.tolist())
        n_steps += 1
    i += STEP

spreads = np.array(spreads)
print(f"Walk-forward OOS: {n_steps} retrains, {len(spreads)} prediction-days")
if len(spreads):
    print(f"  mean {H}d long/short spread: {spreads.mean()*100:+.2f}%  "
          f"(positive on {(spreads > 0).mean()*100:.0f}% of days)")
    print(f"  approx annualised:          {spreads.mean()/H*252*100:+.1f}%  (overlapping, rough)")
print(f"  (single-train OOS spread was -0.04% -> compare)")
