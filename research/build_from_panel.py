"""
Rebuild the cross-sectional train/val/test parquet at a chosen horizon, reusing the
cached diag_panel.parquet (raw features + multi-horizon forward returns). Avoids
re-downloading. Usage:  python build_from_panel.py [HORIZON]   (default 21)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import data   # TECH_FEATURES, TRAIN_END, PURGE_END, VAL_END, PROCESSED_DIR

H = int(sys.argv[1]) if len(sys.argv) > 1 else 21
PANEL_PATH = "data/processed_xgb/diag_panel.parquet"
fwd = f"fwd_{H}"

print(f"Building cross-sectional dataset at horizon H={H} from {PANEL_PATH}")
panel = pd.read_parquet(PANEL_PATH)
if fwd not in panel.columns:
    raise SystemExit(f"{fwd} not in panel (has {[c for c in panel.columns if c.startswith('fwd_')]})")

panel = panel.dropna(subset=[fwd]).copy()

# cross-sectional feature ranking per day (positional assign to dodge dup-index)
ranked = panel.groupby(level=0)[data.TECH_FEATURES].rank(pct=True)
panel[data.TECH_FEATURES] = ranked.values

# cross-sectional label: top tercile = outperform (1), bottom = underperform (0)
r = panel.groupby(level=0)[fwd].rank(pct=True).values
label = np.full(len(panel), np.nan)
label[r >= 0.66] = 1.0
label[r <= 0.34] = 0.0
panel["label"] = label
panel = panel.dropna(subset=["label"])

train = panel[panel.index < data.TRAIN_END].copy()
val = panel[(panel.index >= data.PURGE_END) & (panel.index < data.VAL_END)].copy()
test = panel[panel.index >= data.VAL_END].copy()

os.makedirs(data.PROCESSED_DIR, exist_ok=True)
train.to_parquet(os.path.join(data.PROCESSED_DIR, "train.parquet"))
val.to_parquet(os.path.join(data.PROCESSED_DIR, "val.parquet"))
test.to_parquet(os.path.join(data.PROCESSED_DIR, "test.parquet"))
with open(os.path.join(data.PROCESSED_DIR, "feature_cols.json"), "w") as f:
    json.dump(data.TECH_FEATURES, f, indent=2)

print(f"Labeled rows: {len(panel):,}  balance={panel['label'].mean()*100:.1f}% outperform")
for name, d in [("Train", train), ("Val", val), ("Test", test)]:
    print(f"  {name:6s}: {len(d):,} rows  {d['label'].mean()*100:.1f}% outperform")
print("Saved parquet + feature_cols.json")
