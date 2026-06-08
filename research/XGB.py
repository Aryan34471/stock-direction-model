"""XGBoost + LR ensemble training."""

import json
import pickle
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# paths
DATA_DIR = Path("data/processed_xgb")
OUT_DIR = Path(".")   # models saved next to this script
OUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.parquet"
VAL_PATH = DATA_DIR / "val.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
FEAT_PATH = DATA_DIR / "feature_cols.json"

XGB_MODEL_PATH = OUT_DIR / "xgb_model.pkl"
LR_MODEL_PATH = OUT_DIR / "lr_model.pkl"
TOP_FEATURES_PATH = OUT_DIR / "top_features.json"

# config
TOP_N_FEATURES = 20          # how many features to pass to the LR
ENSEMBLE_XGB_WEIGHT = 0.55   # soft-vote weight for XGBoost
ENSEMBLE_LR_WEIGHT = 0.45


print("=" * 60)
print("Loading data...")
train = pd.read_parquet(TRAIN_PATH)
val = pd.read_parquet(VAL_PATH)
test = pd.read_parquet(TEST_PATH)

with open(FEAT_PATH) as f:
    feature_cols = json.load(f)

print(f"  Train: {len(train):,} rows")
print(f"  Val:   {len(val):,} rows")
print(f"  Test:  {len(test):,} rows")
print(f"  Features: {len(feature_cols)}")

X_train, y_train = train[feature_cols].values, train["label"].values
X_val, y_val = val[feature_cols].values, val["label"].values
X_test, y_test = test[feature_cols].values, test["label"].values

# class imbalance ratio for scale_pos_weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = neg / pos
print(f"\nClass balance - pos: {pos:,}  neg: {neg:,}  scale_pos_weight: {spw:.3f}")

# XGBoost hyperparameter grid
PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
}


def grid_keys_vals(grid):
    keys = list(grid.keys())
    vals = list(grid.values())
    for combo in product(*vals):
        yield dict(zip(keys, combo))


n_combos = 1
for v in PARAM_GRID.values():
    n_combos *= len(v)
print(f"\nXGBoost grid search: {n_combos} combinations...")

best_score = -np.inf
best_params = None
best_xgb = None
start = time.time()

for i, params in enumerate(grid_keys_vals(PARAM_GRID), 1):
    model = XGBClassifier(
        **params,
        scale_pos_weight=spw,
        eval_metric="auc",
        early_stopping_rounds=30,
        tree_method="hist",
        device="cuda",   # RTX 4060 Ti
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    preds = model.predict(X_val)
    score = balanced_accuracy_score(y_val, preds)

    if score > best_score:
        best_score = score
        best_params = params
        best_xgb = model
        marker = "  <- best"
    else:
        marker = ""

    elapsed = time.time() - start
    print(f"  [{i:3d}/{n_combos}]  bal_acc={score:.4f}  {params}{marker}  ({elapsed:.0f}s)")

print(f"\nBest val balanced_accuracy: {best_score:.4f}")
print(f"Best params: {best_params}")

# feature importances
importances = best_xgb.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

print("\nTop 20 features by XGBoost importance:")
print(feat_imp.head(20).to_string())

top_features = feat_imp.head(TOP_N_FEATURES).index.tolist()

with open(TOP_FEATURES_PATH, "w") as f:
    json.dump(top_features, f, indent=2)
print(f"\nSaved top_features.json ({TOP_N_FEATURES} features)")

# logistic regression on the top features
print("\nTraining Logistic Regression on top features...")

feat_idx = [feature_cols.index(f) for f in top_features]
X_train_lr = X_train[:, feat_idx]
X_val_lr = X_val[:, feat_idx]
X_test_lr = X_test[:, feat_idx]

imputer = SimpleImputer(strategy="median")
X_train_lr = imputer.fit_transform(X_train_lr)
X_val_lr = imputer.transform(X_val_lr)
X_test_lr = imputer.transform(X_test_lr)

best_lr = None
best_lr_score = -np.inf

for C in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    lr = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=C,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    lr.fit(X_train_lr, y_train)
    preds = lr.predict(X_val_lr)
    score = balanced_accuracy_score(y_val, preds)
    print(f"  C={C:.3f}  val bal_acc={score:.4f}")
    if score > best_lr_score:
        best_lr_score = score
        best_lr = lr

print(f"\nBest LR val balanced_accuracy: {best_lr_score:.4f}  (C={best_lr.C})")

# evaluate the ensemble on the test set
print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)


def evaluate(name, y_true, y_pred, y_prob):
    print(f"\n-- {name} --")
    print(f"  Accuracy:          {(y_pred == y_true).mean():.4f}")
    print(f"  Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"  Macro F1:          {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"  ROC AUC:           {roc_auc_score(y_true, y_prob):.4f}")
    print(classification_report(y_true, y_pred, target_names=["Underperform", "Outperform"]))


# XGBoost alone
xgb_prob_test = best_xgb.predict_proba(X_test)[:, 1]
xgb_pred_test = (xgb_prob_test >= 0.5).astype(int)
evaluate("XGBoost (standalone)", y_test, xgb_pred_test, xgb_prob_test)

# LR alone
lr_prob_test = best_lr.predict_proba(X_test_lr)[:, 1]
lr_pred_test = (lr_prob_test >= 0.5).astype(int)
evaluate("Logistic Regression (standalone)", y_test, lr_pred_test, lr_prob_test)

# soft-vote ensemble
ens_prob_test = ENSEMBLE_XGB_WEIGHT * xgb_prob_test + ENSEMBLE_LR_WEIGHT * lr_prob_test
ens_pred_test = (ens_prob_test >= 0.5).astype(int)
evaluate("Ensemble (soft vote)", y_test, ens_pred_test, ens_prob_test)

# how accuracy changes as we raise the confidence threshold
for thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
    mask = ens_prob_test >= thresh
    n = mask.sum()
    if n == 0:
        continue
    acc = (ens_pred_test[mask] == y_test[mask]).mean()
    print(f"  P >= {thresh:.2f}  ->  {n:5d} samples  accuracy={acc:.4f}")

# save models
with open(XGB_MODEL_PATH, "wb") as f:
    pickle.dump(best_xgb, f)
print(f"\nSaved {XGB_MODEL_PATH}")

with open(LR_MODEL_PATH, "wb") as f:
    pickle.dump(best_lr, f)
print(f"Saved {LR_MODEL_PATH}")

print("\nDone.")
