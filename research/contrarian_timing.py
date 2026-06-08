"""Tests whether the contrarian market-timing signal beats buy-and-hold on a risk-adjusted basis."""

import numpy as np
import pandas as pd
import data

W = 252
SIGNS = {"vix_level": +1, "spy_vol_20": +1, "spy_vs_sma200": -1, "spy_rsi_14": -1}

spy = data.download_prices("SPY")
try:
    vix = data.download_prices("^VIX")
except Exception:
    vix = None

df = data.compute_macro_features(spy, vix).copy()
df["spy_ret"] = spy["close"].pct_change()
df = df.dropna(subset=["spy_ret"])

comp = pd.DataFrame(index=df.index)
for c, s in SIGNS.items():
    x = s * df[c]
    comp[c] = (x - x.rolling(W).mean()) / (x.rolling(W).std() + 1e-9)
df["composite"] = comp.mean(axis=1)
df = df.dropna(subset=["composite"])

df["exposure"] = (1 + df["composite"]).clip(0, 2)
df["strat_ret"] = df["exposure"].shift(1) * df["spy_ret"]   # lag -> no look-ahead
df = df.dropna(subset=["strat_ret"])


def stats(r):
    ann = r.mean() * 252 * 100
    vol = r.std() * np.sqrt(252) * 100
    sharpe = (r.mean() / (r.std() + 1e-12)) * np.sqrt(252)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min() * 100
    return ann, vol, sharpe, dd


for name, sl in [("FULL SAMPLE", df), ("TEST (>=2024-06)", df[df.index >= "2024-06-01"])]:
    print(f"\n== {name} ==  n={len(sl)}  avg exposure={sl['exposure'].mean():.2f}")
    print(f"  {'strategy':18s} {'annRet':>8s} {'annVol':>8s} {'Sharpe':>8s} {'maxDD':>8s}")
    for label, r in [("Buy & Hold SPY", sl["spy_ret"]), ("Contrarian timing", sl["strat_ret"])]:
        ann, vol, sh, dd = stats(r)
        print(f"  {label:18s} {ann:+7.1f}% {vol:7.1f}% {sh:+8.2f} {dd:7.1f}%")
