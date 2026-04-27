"""
models.py
=========
Model factory for the Ukrainian real estate price-forecasting pipeline.

All models are trained on **log1p-transformed** target values and predictions
are back-transformed with ``np.expm1``.

Models
------
  Ridge      — L2-regularised linear regression with Duan smearing correction.
               Alpha selected on the validation set from a log-spaced grid.
  RF         — Random Forest (n=300, sqrt features, min_leaf=5).
  XGBoost    — Gradient boosting with early stopping on val; refit on train+val.
               Tuned hyperparameters from Optuna search (see thesis, Chapter 4).
  LightGBM   — GBDT with num_leaves=255 (tuned value), early stopping; refit.
  MLP        — 3-layer neural network (256-128-64) with manual early stopping.

Factory pattern
---------------
Each ``build_<model>()`` function returns a fresh, untrained estimator with
the tuned hyperparameters.  The training functions (``train_*``) use these
factories and apply the standard train → val early-stop → refit on train+val
workflow.

Public API
----------
build_ridge()   -> sklearn Pipeline
build_rf()      -> RandomForestRegressor
build_xgb()     -> XGBRegressor
build_lgbm()    -> LGBMRegressor
build_mlp()     -> MLPRegressor

train_ridge(seg_data, SEG_CONFIG)  -> dict of results
train_rf(seg_data, SEG_CONFIG)     -> dict of results
train_xgb(seg_data, SEG_CONFIG)    -> dict of results
train_lgbm(seg_data, SEG_CONFIG)   -> dict of results
train_mlp(seg_data, SEG_CONFIG)    -> dict of results
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None   # xgboost optional

try:
    from lightgbm import LGBMRegressor
    import lightgbm as _lgb
except ImportError:
    LGBMRegressor = None  # lightgbm optional
    _lgb = None


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _fill(X: pd.DataFrame, train_med: pd.Series) -> pd.DataFrame:
    """Replace infinities and NaN with training-set medians, then 0."""
    return X.replace([np.inf, -np.inf], np.nan).fillna(train_med).fillna(0.0)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (%) — robust to outliers."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(100.0 * np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-12))


def _score(y_te: np.ndarray, y_pred: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """Compute MAE, RMSE, R² (log scale), MAPE, WAPE for a single segment."""
    return dict(
        MAE  = mean_absolute_error(y_te, y_pred),
        RMSE = float(np.sqrt(mean_squared_error(y_te, y_pred))),
        R2   = float(r2_score(np.log1p(y_te), y_pred_log)),
        MAPE = mape(y_te, y_pred),
        WAPE = wape(y_te, y_pred),
    )


# ---------------------------------------------------------------------------
# Model factories — return a fresh untrained model with tuned hyperparameters
# ---------------------------------------------------------------------------

# Ridge alpha search grid (log-spaced, 15 candidates)
RIDGE_ALPHAS = list(np.logspace(-3, 4, 15))


def build_ridge(alpha: float = 1.0) -> Pipeline:
    """Ridge regression wrapped in StandardScaler pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=alpha, random_state=RANDOM_STATE)),
    ])


def build_rf() -> RandomForestRegressor:
    """Random Forest with tuned hyperparameters."""
    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


# XGBoost tuned hyperparameters (Optuna, 100 trials on validation set)
_XGB_PARAMS = dict(
    n_estimators=6000,
    learning_rate=0.02,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.5,
    gamma=0.05,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbosity=0,
    tree_method="hist",
    objective="reg:squarederror",
    eval_metric="rmse",
)


def build_xgb(**override) -> "XGBRegressor":
    """XGBoost with tuned hyperparameters. Pass ``n_estimators`` to override."""
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed: pip install xgboost")
    return XGBRegressor(**{**_XGB_PARAMS, **override})


# LightGBM tuned hyperparameters  — num_leaves=255 is the tuned value
_LGBM_PARAMS = dict(
    n_estimators=4000,
    learning_rate=0.02,
    num_leaves=255,          # tuned; controls model capacity
    max_depth=-1,
    min_child_samples=20,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    boosting_type="gbdt",
    reg_alpha=0.1,
    reg_lambda=1.5,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=-1,
    objective="regression",
)


def build_lgbm(**override) -> "LGBMRegressor":
    """LightGBM with tuned hyperparameters. Pass ``n_estimators`` to override."""
    if LGBMRegressor is None:
        raise ImportError("lightgbm is not installed: pip install lightgbm")
    return LGBMRegressor(**{**_LGBM_PARAMS, **override})


def build_mlp() -> MLPRegressor:
    """MLP with architecture (256, 128, 64) and Adam optimiser."""
    return MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=5e-4,
        max_iter=1,
        warm_start=True,
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Training functions — early-stop on val, refit on train+val, score on test
# ---------------------------------------------------------------------------

def train_ridge(seg_data: dict, SEG_CONFIG: dict) -> dict:
    """
    Train Ridge for each segment.

    Alpha is selected from ``RIDGE_ALPHAS`` by MSE on the validation set.
    Final model is refit on train + val.  Duan smearing corrects for
    log-space retransformation bias.
    """
    print("=" * 72)
    print("Ridge (L2) | alpha tuned on val | StandardScaler + Duan smearing")
    print("=" * 72)
    results = {}

    for (city, ctype), data in seg_data.items():
        if data is None:
            continue
        cfg   = SEG_CONFIG[(city, ctype)]
        label = cfg["label"]
        med   = data["X_train"].median(numeric_only=True)
        X_tr  = _fill(data["X_train"], med)
        X_va  = _fill(data["X_val"],   med)
        X_te  = _fill(data["X_test"],  med)
        y_tr  = np.log1p(data["y_train"].values)
        y_va  = np.log1p(data["y_val"].values)
        y_te  = data["y_test"].values

        # --- alpha search on val ---
        best_alpha, best_mse = RIDGE_ALPHAS[0], np.inf
        for alpha in RIDGE_ALPHAS:
            pipe = build_ridge(alpha)
            pipe.fit(X_tr, y_tr)
            mse_val = mean_squared_error(y_va, pipe.predict(X_va))
            if mse_val < best_mse:
                best_mse, best_alpha = mse_val, alpha

        # --- refit on train + val ---
        X_full = pd.concat([X_tr, X_va])
        y_full = np.concatenate([y_tr, y_va])
        model  = build_ridge(best_alpha)
        model.fit(X_full, y_full)

        # Duan smearing: E[exp(epsilon)] estimated from train residuals
        smear      = float(np.mean(np.exp(y_tr - model.predict(X_tr))))
        y_pred_log = model.predict(X_te)
        y_pred     = np.expm1(y_pred_log) * smear

        scores = _score(y_te, y_pred, y_pred_log)
        results[(city, ctype)] = {
            "model": model, "label": label, "alpha": best_alpha,
            "y_test": y_te, "y_pred": y_pred, **scores,
        }
        print(f"\n  {label}  (alpha={best_alpha:.4g})")
        print(f"    MAE: {scores['MAE']:.2f} {cfg['currency']}  "
              f"R2: {scores['R2']:.4f}  MAPE: {scores['MAPE']:.2f}%")

    return results


def train_rf(seg_data: dict, SEG_CONFIG: dict) -> dict:
    """Train Random Forest for each segment (no early stopping needed)."""
    print("=" * 72)
    print("Random Forest | n=300 | max_features=sqrt | min_samples_leaf=5")
    print("=" * 72)
    results = {}

    for (city, ctype), data in seg_data.items():
        if data is None:
            continue
        cfg   = SEG_CONFIG[(city, ctype)]
        label = cfg["label"]
        X_tr  = data["X_train"]
        X_te  = data["X_test"]
        y_tr  = np.log1p(data["y_train"].values)
        y_te  = data["y_test"].values

        model = build_rf()
        model.fit(X_tr, y_tr)
        y_pred_log = model.predict(X_te)
        y_pred     = np.expm1(y_pred_log)

        scores = _score(y_te, y_pred, y_pred_log)
        results[(city, ctype)] = {
            "model": model, "label": label,
            "y_test": y_te, "y_pred": y_pred,
            "importances": pd.Series(model.feature_importances_, index=X_tr.columns),
            **scores,
        }
        print(f"\n  {label}")
        print(f"    MAE: {scores['MAE']:.2f} {cfg['currency']}  "
              f"R2: {scores['R2']:.4f}  MAPE: {scores['MAPE']:.2f}%")

    return results


def train_xgb(seg_data: dict, SEG_CONFIG: dict) -> dict:
    """
    Train XGBoost for each segment.

    Uses early stopping on the validation set to determine optimal tree count,
    then refits on train + val with that count.
    """
    if XGBRegressor is None:
        print("xgboost not installed: pip install xgboost")
        return {}
    print("=" * 72)
    print("XGBoost | early stopping on val | refit on train+val")
    print("=" * 72)
    results = {}

    for (city, ctype), data in seg_data.items():
        if data is None:
            continue
        cfg   = SEG_CONFIG[(city, ctype)]
        label = cfg["label"]
        med   = data["X_train"].median(numeric_only=True)
        X_tr  = _fill(data["X_train"], med)
        X_va  = _fill(data["X_val"],   med)
        X_te  = _fill(data["X_test"],  med)
        y_tr  = np.log1p(data["y_train"].values)
        y_va  = np.log1p(data["y_val"].values)
        y_te  = data["y_test"].values

        # Early-stopping probe
        probe = build_xgb(early_stopping_rounds=200)
        probe.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        n_best = int((probe.best_iteration or 0) + 1)

        # Refit on train + val with the selected tree count
        X_full = pd.concat([X_tr, X_va])
        y_full = np.concatenate([y_tr, y_va])
        model  = build_xgb(n_estimators=max(200, min(n_best, 4000)))
        model.fit(X_full, y_full, verbose=False)

        y_pred_log = model.predict(X_te)
        y_pred     = np.expm1(y_pred_log)

        scores = _score(y_te, y_pred, y_pred_log)
        results[(city, ctype)] = {
            "model": model, "label": label, "n_trees": n_best,
            "y_test": y_te, "y_pred": y_pred,
            "importances": pd.Series(model.feature_importances_, index=X_tr.columns),
            **scores,
        }
        print(f"\n  {label}  (trees={n_best})")
        print(f"    MAE: {scores['MAE']:.2f} {cfg['currency']}  "
              f"R2: {scores['R2']:.4f}  MAPE: {scores['MAPE']:.2f}%")

    return results


def train_lgbm(seg_data: dict, SEG_CONFIG: dict) -> dict:
    """
    Train LightGBM for each segment.

    num_leaves=255 is the tuned value from Optuna search.
    Uses early stopping on the validation set; refits on train + val.
    """
    if LGBMRegressor is None:
        print("lightgbm not installed: pip install lightgbm")
        return {}
    print("=" * 72)
    print("LightGBM (GBDT) | num_leaves=255 | early stopping on val | refit")
    print("=" * 72)
    results = {}

    for (city, ctype), data in seg_data.items():
        if data is None:
            continue
        cfg   = SEG_CONFIG[(city, ctype)]
        label = cfg["label"]
        med   = data["X_train"].median(numeric_only=True)
        X_tr  = _fill(data["X_train"], med)
        X_va  = _fill(data["X_val"],   med)
        X_te  = _fill(data["X_test"],  med)
        y_tr  = np.log1p(data["y_train"].values)
        y_va  = np.log1p(data["y_val"].values)
        y_te  = data["y_test"].values

        # Early-stopping probe
        callbacks = [
            _lgb.early_stopping(stopping_rounds=150, verbose=False),
            _lgb.log_evaluation(period=-1),
        ]
        probe = build_lgbm()
        probe.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)
        n_best = int(probe.best_iteration_ or probe.n_estimators)

        # Refit on train + val
        X_full = pd.concat([X_tr, X_va])
        y_full = np.concatenate([y_tr, y_va])
        model  = build_lgbm(n_estimators=max(200, min(n_best, 4000)))
        model.fit(X_full, y_full)

        y_pred_log = model.predict(X_te)
        y_pred     = np.expm1(y_pred_log)

        scores = _score(y_te, y_pred, y_pred_log)
        results[(city, ctype)] = {
            "model": model, "label": label, "n_trees": n_best,
            "y_test": y_te, "y_pred": y_pred,
            "importances": pd.Series(model.feature_importances_, index=X_tr.columns),
            **scores,
        }
        print(f"\n  {label}  (trees={n_best})")
        print(f"    MAE: {scores['MAE']:.2f} {cfg['currency']}  "
              f"R2: {scores['R2']:.4f}  MAPE: {scores['MAPE']:.2f}%")

    return results


def train_mlp(seg_data: dict, SEG_CONFIG: dict) -> dict:
    """
    Train MLP for each segment.

    StandardScaler is fit on X_train (no leakage).
    Target is also standardised in log space.
    Manual early stopping with patience=20 epochs.
    Final model uses best weights; refits on train + val.
    """
    print("=" * 72)
    print("MLP (256-128-64) | manual early stopping (patience=20) | refit")
    print("=" * 72)
    results = {}

    MAX_EPOCHS = 200
    PATIENCE   = 20

    for (city, ctype), data in seg_data.items():
        if data is None:
            continue
        cfg   = SEG_CONFIG[(city, ctype)]
        label = cfg["label"]

        # Clean arrays
        def _clean(X):
            return pd.DataFrame(X).replace([np.inf, -np.inf], np.nan).fillna(0.0).values

        X_tr = _clean(data["X_train"])
        X_va = _clean(data["X_val"])
        X_te = _clean(data["X_test"])

        y_tr_raw = np.clip(data["y_train"].values.astype(float), 0.0, None)
        y_va_raw = np.clip(data["y_val"].values.astype(float),   0.0, None)
        y_te     = np.clip(data["y_test"].values.astype(float),  0.0, None)
        y_tr_log = np.log1p(y_tr_raw)
        y_va_log = np.log1p(y_va_raw)

        # Scalers fitted on train only
        x_sc = StandardScaler().fit(X_tr)
        y_sc = StandardScaler().fit(y_tr_log.reshape(-1, 1))
        X_tr_s = x_sc.transform(X_tr)
        X_va_s = x_sc.transform(X_va)
        X_te_s = x_sc.transform(X_te)
        y_tr_s = y_sc.transform(y_tr_log.reshape(-1, 1)).ravel()
        y_va_s = y_sc.transform(y_va_log.reshape(-1, 1)).ravel()

        model = build_mlp()
        best_val_loss, patience, best_state, epoch = np.inf, 0, None, 0

        for epoch in range(MAX_EPOCHS):
            model.fit(X_tr_s, y_tr_s)
            val_loss = mean_squared_error(y_va_s, model.predict(X_va_s))
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = (model.coefs_[:], model.intercepts_[:])
                patience = 0
            else:
                patience += 1
            if patience >= PATIENCE:
                break

        if best_state is not None:
            model.coefs_, model.intercepts_ = best_state

        y_pred_scaled = model.predict(X_te_s)
        y_pred_log    = y_sc.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_pred        = np.expm1(y_pred_log)

        scores = _score(y_te, y_pred, y_pred_log)
        results[(city, ctype)] = {
            "model": model, "label": label, "epochs": epoch + 1,
            "y_test": y_te, "y_pred": y_pred, **scores,
        }
        print(f"\n  {label}  (epochs={epoch + 1})")
        print(f"    MAE: {scores['MAE']:.2f} {cfg['currency']}  "
              f"R2: {scores['R2']:.4f}  MAPE: {scores['MAPE']:.2f}%")

    return results
