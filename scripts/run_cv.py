"""
run_cv.py
=========
Rolling-origin (walk-forward) cross-validation across all five models
and four market segments.

CV design
---------
  4 folds, each with an expanding training window and a fixed 3-month
  validation window:

    Fold 1: train ≤ 2024-01-31 | val 2024-02-01 – 2024-04-30
    Fold 2: train ≤ 2024-04-30 | val 2024-05-01 – 2024-07-31
    Fold 3: train ≤ 2024-07-31 | val 2024-08-01 – 2024-10-31
    Fold 4: train ≤ 2024-10-31 | val 2024-11-01 – 2025-01-31

Output
------
  - Per-fold metrics table (console + CSV).
  - Mean ± std summary across folds.
  - Fold-stability pivot (MAPE std).
  - Ranking by mean MAPE across all segments.
  - Results saved to  scripts/../cv_results.csv.

Usage
-----
  cd thesis_package
  python scripts/run_cv.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — make sure thesis_package/src/ is importable
# ---------------------------------------------------------------------------
_HERE   = os.path.dirname(os.path.abspath(__file__))          # scripts/
_PKG    = os.path.dirname(_HERE)                               # thesis_package/
_REPO   = os.path.dirname(_PKG)                               # repo root
_SRCDIR = os.path.join(_PKG, "src")

for _p in (_SRCDIR, _PKG, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import from the canonical source modules located at <repo_root>/src/
# (not thesis_package/src/ — the pipeline, cleaning, geo, macro modules
#  live in the original src/ directory).
sys.path.insert(0, _REPO)
from src.pipeline import step1_load, step2_clean, step3_features, step4_macro_geo
from src.preprocessing import preprocess_features

# Config and feature list from thesis_package
from config import RANDOM_STATE
from preprocessing import FEATURE_COLS_BASE

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — XGB will be skipped.")

try:
    from lightgbm import LGBMRegressor
    import lightgbm as lgb_lib
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: lightgbm not installed — LGBM will be skipped.")


# ---------------------------------------------------------------------------
# CV fold definitions
# ---------------------------------------------------------------------------

FOLDS = [
    (pd.Timestamp("2024-01-31"), pd.Timestamp("2024-04-30")),
    (pd.Timestamp("2024-04-30"), pd.Timestamp("2024-07-31")),
    (pd.Timestamp("2024-07-31"), pd.Timestamp("2024-10-31")),
    (pd.Timestamp("2024-10-31"), pd.Timestamp("2025-01-31")),
]

SEG_LABELS = {
    ("Львів", "Оренда"): "Lviv-Rent",
    ("Київ",  "Оренда"): "Kyiv-Rent",
    ("Львів", "Продаж"): "Lviv-Sale",
    ("Київ",  "Продаж"): "Kyiv-Sale",
}

SEG_TARGETS = {
    ("Львів", "Оренда"): "price_uah",
    ("Київ",  "Оренда"): "price_uah",
    ("Львів", "Продаж"): "price_m2_usd",
    ("Київ",  "Продаж"): "price_m2_usd",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill(X, med):
    return X.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)


def _metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100)
    return mae, rmse, mape


# ---------------------------------------------------------------------------
# Model factories (one per fold — trained from scratch each time)
# ---------------------------------------------------------------------------

def _model_factories(X_tr, y_tr_log, X_va, y_va_log):
    """
    Return a dict of {model_name: fit_fn}.
    Each fit_fn() trains the model and returns (predict_fn, extra_info).
    """
    factories = {}

    # Ridge — alpha selected on val
    def fit_ridge():
        best_a, best_mse = 1.0, np.inf
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            p = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=alpha, random_state=RANDOM_STATE))])
            p.fit(X_tr, y_tr_log)
            mse = mean_squared_error(y_va_log, p.predict(X_va))
            if mse < best_mse:
                best_mse, best_a = mse, alpha
        pipe = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=best_a, random_state=RANDOM_STATE))])
        pipe.fit(X_tr, y_tr_log)
        return (lambda X: pipe.predict(X)), best_a

    factories["Ridge"] = fit_ridge

    # Random Forest
    def fit_rf():
        rf = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, max_features="sqrt",
            n_jobs=-1, random_state=RANDOM_STATE,
        )
        rf.fit(X_tr, y_tr_log)
        return (lambda X: rf.predict(X)), None

    factories["RF"] = fit_rf

    # XGBoost
    if HAS_XGB:
        def fit_xgb():
            probe = XGBRegressor(
                n_estimators=2000, learning_rate=0.02, max_depth=6,
                subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.5, n_jobs=-1, random_state=RANDOM_STATE,
                verbosity=0, tree_method="hist", early_stopping_rounds=80,
            )
            probe.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
            n_best = int((probe.best_iteration or 0) + 1)
            final = XGBRegressor(
                n_estimators=max(100, n_best), learning_rate=0.02, max_depth=6,
                subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.5, n_jobs=-1, random_state=RANDOM_STATE,
                verbosity=0, tree_method="hist",
            )
            final.fit(X_tr, y_tr_log)
            return (lambda X: final.predict(X)), n_best

        factories["XGB"] = fit_xgb

    # LightGBM
    if HAS_LGBM:
        def fit_lgbm():
            cbs = [lgb_lib.early_stopping(60, verbose=False), lgb_lib.log_evaluation(period=-1)]
            probe = LGBMRegressor(
                n_estimators=2000, learning_rate=0.02, num_leaves=255,
                min_child_samples=20, feature_fraction=0.85, bagging_fraction=0.85,
                bagging_freq=5, reg_alpha=0.1, reg_lambda=1.5,
                n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
            )
            probe.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], callbacks=cbs)
            n_best = int(probe.best_iteration_ or probe.n_estimators)
            final = LGBMRegressor(
                n_estimators=max(100, n_best), learning_rate=0.02, num_leaves=255,
                min_child_samples=20, feature_fraction=0.85, bagging_fraction=0.85,
                bagging_freq=5, reg_alpha=0.1, reg_lambda=1.5,
                n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
            )
            final.fit(X_tr, y_tr_log)
            return (lambda X: final.predict(X)), n_best

        factories["LGBM"] = fit_lgbm

    # MLP
    def fit_mlp():
        x_sc = StandardScaler().fit(X_tr)
        y_sc = StandardScaler().fit(y_tr_log.reshape(-1, 1))
        Xts  = x_sc.transform(X_tr)
        Xvs  = x_sc.transform(X_va)
        yts  = y_sc.transform(y_tr_log.reshape(-1, 1)).ravel()
        yvs  = y_sc.transform(y_va_log.reshape(-1, 1)).ravel()
        mlp  = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            solver="adam", learning_rate_init=5e-4,
            max_iter=1, warm_start=True, random_state=RANDOM_STATE,
        )
        best_val, patience, best_state = np.inf, 0, None
        for _ in range(150):
            mlp.fit(Xts, yts)
            vl = mean_squared_error(yvs, mlp.predict(Xvs))
            if vl < best_val - 1e-6:
                best_val, best_state, patience = vl, (mlp.coefs_[:], mlp.intercepts_[:]), 0
            else:
                patience += 1
            if patience >= 15:
                break
        if best_state:
            mlp.coefs_, mlp.intercepts_ = best_state

        def pred_fn(X):
            ys = mlp.predict(x_sc.transform(X))
            return y_sc.inverse_transform(ys.reshape(-1, 1)).ravel()

        return pred_fn, None

    factories["MLP"] = fit_mlp

    return factories


# ---------------------------------------------------------------------------
# Per-segment CV runner
# ---------------------------------------------------------------------------

def run_cv_for_segment(df_seg: pd.DataFrame, target: str, label: str) -> pd.DataFrame:
    """Run 4-fold rolling-origin CV for all models on one segment."""
    df = df_seg.copy()
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df[target]  = pd.to_numeric(df[target], errors="coerce")
    df          = df.dropna(subset=["date", target])
    feats       = [c for c in FEATURE_COLS_BASE if c in df.columns]

    rows = []
    print(f"\n{'─' * 60}")
    print(f"  Segment: {label}  |  features={len(feats)}  |  N={len(df):,}")
    print(f"{'─' * 60}")

    for i, (tr_end, va_end) in enumerate(FOLDS, 1):
        tr = df[df["date"] <= tr_end]
        va = df[(df["date"] > tr_end) & (df["date"] <= va_end)]

        if len(tr) < 500 or len(va) < 100:
            print(f"  Fold {i}: SKIP  (train={len(tr)}, val={len(va)} — too small)")
            continue

        med       = tr[feats].median()
        X_tr      = _fill(tr[feats], med).values
        X_va      = _fill(va[feats], med).values
        y_tr_log  = np.log1p(tr[target].values)
        y_va_log  = np.log1p(va[target].values)
        y_va      = va[target].values

        print(f"\n  Fold {i}  [{tr_end.date()} -> {va_end.date()}]  "
              f"train={len(tr):,}  val={len(va):,}")

        factories = _model_factories(X_tr, y_tr_log, X_va, y_va_log)

        for name, fit_fn in factories.items():
            try:
                pred_fn, extra = fit_fn()
                y_pred = np.expm1(pred_fn(X_va))
                mae, rmse, mape_val = _metrics(y_va, y_pred)
                wape_val = 100.0 * np.sum(np.abs(y_va - y_pred)) / (np.sum(np.abs(y_va)) + 1e-12)
                print(f"    {name:<6}  MAE={mae:>10.2f}  RMSE={rmse:>10.2f}  "
                      f"MAPE={mape_val:>5.2f}%  WAPE={wape_val:>5.2f}%")
                rows.append({
                    "Segment": label, "Model": name, "Fold": i,
                    "Train_End": str(tr_end.date()), "Val_End": str(va_end.date()),
                    "N_Train": len(tr), "N_Val": len(va),
                    "MAE": round(mae, 2), "RMSE": round(rmse, 2),
                    "MAPE%": round(mape_val, 2), "WAPE%": round(wape_val, 2),
                    "Extra": extra,
                })
            except Exception as exc:
                print(f"    {name:<6}  ERROR: {exc}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  ROLLING-ORIGIN CV  |  5 models x 4 segments x 4 folds")
    print("=" * 70)

    print("\n[1/4] Loading & cleaning...")
    _, segments = step1_load()
    clean       = step2_clean(segments)

    print("\n[2/4] Feature engineering...")
    feat = step3_features(clean)

    print("\n[3/4] Macro + alerts (no Overpass geo — fast mode)...")
    enriched = step4_macro_geo(feat, load_geo=False)

    print("\n[4/4] Preprocessing (encoding, no leakage)...")
    processed = {k: preprocess_features(v) for k, v in enriched.items()}

    # --- Run CV ---
    print("\n" + "=" * 70)
    print("  RUNNING CV")
    print("=" * 70)

    all_frames = []
    for (city, ctype), df in processed.items():
        target = SEG_TARGETS[(city, ctype)]
        label  = SEG_LABELS[(city, ctype)]
        frame  = run_cv_for_segment(df, target, label)
        all_frames.append(frame)

    df_all = pd.concat(all_frames, ignore_index=True)

    # --- Per-fold table ---
    print("\n\n" + "=" * 70)
    print("  CV RESULTS — PER-FOLD TABLE")
    print("=" * 70)
    cols = ["Segment", "Model", "Fold", "Train_End", "Val_End",
            "N_Train", "N_Val", "MAE", "RMSE", "MAPE%", "WAPE%"]
    print(df_all[cols].to_string(index=False))

    # --- Summary ---
    print("\n\n" + "=" * 70)
    print("  CV SUMMARY — MEAN +/- STD ACROSS FOLDS")
    print("=" * 70)
    summary = (
        df_all.groupby(["Segment", "Model"])
        .agg(
            MAE_mean=("MAE",   "mean"), MAE_std=("MAE",   "std"),
            RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
            MAPE_mean=("MAPE%","mean"), MAPE_std=("MAPE%","std"),
            WAPE_mean=("WAPE%","mean"), WAPE_std=("WAPE%","std"),
            n_folds=("Fold",   "count"),
        )
        .round(2)
        .reset_index()
    )
    print(summary.to_string(index=False))

    # --- Stability pivot ---
    print("\n\n" + "=" * 70)
    print("  STABILITY: MAPE CV-Std (lower = more stable)")
    print("=" * 70)
    pivot = summary.pivot(index="Model", columns="Segment", values="MAPE_std").round(2)
    print(pivot.to_string())

    # --- Ranking ---
    print("\n\n" + "=" * 70)
    print("  RANKING: Mean MAPE% across all segments (lower is better)")
    print("=" * 70)
    rank = (
        summary.groupby("Model")
        .agg(
            Overall_MAPE_mean=("MAPE_mean", "mean"),
            Overall_MAPE_std= ("MAPE_std",  "mean"),
            Overall_WAPE_mean=("WAPE_mean", "mean"),
        )
        .round(2)
        .sort_values("Overall_MAPE_mean")
    )
    print(rank.to_string())

    # --- Fold drift ---
    print("\n\n" + "=" * 70)
    print("  FOLD DRIFT: MAPE by fold (temporal degradation check)")
    print("=" * 70)
    fold_drift = df_all.groupby(["Model", "Fold"])["MAPE%"].mean().unstack("Fold").round(2)
    print(fold_drift.to_string())

    # --- Save ---
    out_csv = os.path.join(_PKG, "cv_results.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"\n\nResults saved -> {out_csv}")

    return df_all, summary


if __name__ == "__main__":
    df_all, summary = main()
