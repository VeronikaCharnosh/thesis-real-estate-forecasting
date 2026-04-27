"""
tuning.py — grid search по val-множині для RF, XGB, LGBM, MLP.
Повертає best_params per (model, segment), які потім підставляються у навчання.

Публічний API:
    tune_all(seg_data, SEG_CONFIG) → best_params dict
    print_tuning_summary(best_params)
"""

import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    import lightgbm as lgb_lib
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def _fill(X, med):
    return X.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.)


def _val_rmse_log(y_va_log, y_pred_log):
    return np.sqrt(mean_squared_error(y_va_log, y_pred_log))


# ── RF ────────────────────────────────────────────────────────────────────────

RF_GRID = {
    'min_samples_leaf': [3, 5, 10, 20],
    'max_features':     ['sqrt', 0.6, 0.8],
}


def tune_rf(seg_data: dict, SEG_CONFIG: dict) -> dict:
    print('\n' + '='*60)
    print('Tuning RF  (grid search on VAL)')
    print('='*60)
    best = {}
    grid = list(itertools.product(RF_GRID['min_samples_leaf'], RF_GRID['max_features']))

    for key, data in seg_data.items():
        if data is None:
            continue
        label = SEG_CONFIG[key]['label']
        med   = data['X_train'].median(numeric_only=True)
        X_tr  = _fill(data['X_train'], med)
        X_va  = _fill(data['X_val'],   med)
        y_tr  = np.log1p(data['y_train'].values)
        y_va  = np.log1p(data['y_val'].values)

        best_rmse, best_p = np.inf, {}
        for min_leaf, max_feat in grid:
            rf = RandomForestRegressor(
                n_estimators=300, min_samples_leaf=min_leaf, max_features=max_feat,
                n_jobs=-1, random_state=RANDOM_STATE,
            )
            rf.fit(X_tr, y_tr)
            rmse = _val_rmse_log(y_va, rf.predict(X_va))
            if rmse < best_rmse:
                best_rmse = rmse
                best_p = {'min_samples_leaf': min_leaf, 'max_features': max_feat}

        best[key] = best_p
        print(f'  [{label}]  best: {best_p}  val_RMSE={best_rmse:.5f}')
    return best


# ── XGBoost ───────────────────────────────────────────────────────────────────

XGB_GRID = {
    'max_depth':       [6, 8],
    'min_child_weight': [3, 5],
    'subsample':       [0.75, 0.85],
}


def tune_xgb(seg_data: dict, SEG_CONFIG: dict) -> dict:
    if not HAS_XGB:
        print('xgboost не встановлено')
        return {}
    print('\n' + '='*60)
    print('Tuning XGB  (early stopping на VAL, grid по структурі)')
    print('='*60)
    best = {}
    grid = list(itertools.product(
        XGB_GRID['max_depth'], XGB_GRID['min_child_weight'], XGB_GRID['subsample']
    ))

    for key, data in seg_data.items():
        if data is None:
            continue
        label = SEG_CONFIG[key]['label']
        med   = data['X_train'].median(numeric_only=True)
        X_tr  = _fill(data['X_train'], med)
        X_va  = _fill(data['X_val'],   med)
        y_tr  = np.log1p(data['y_train'].values)
        y_va  = np.log1p(data['y_val'].values)

        best_rmse, best_p, best_ntree = np.inf, {}, 500
        for max_d, min_cw, sub in grid:
            xgb = XGBRegressor(
                n_estimators=10000, learning_rate=0.02,
                max_depth=max_d, min_child_weight=min_cw,
                subsample=sub, colsample_bytree=0.85,
                reg_alpha=0.1, reg_lambda=1.5, gamma=0.05,
                n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
                tree_method='hist', early_stopping_rounds=200,
            )
            xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            rmse = xgb.best_score ** 0.5 if xgb.best_score else _val_rmse_log(y_va, xgb.predict(X_va))
            n_tree = int((xgb.best_iteration or 0) + 1)
            if rmse < best_rmse:
                best_rmse = rmse
                best_ntree = n_tree
                best_p = {'max_depth': max_d, 'min_child_weight': min_cw,
                          'subsample': sub, 'n_estimators': n_tree}

        best[key] = best_p
        print(f'  [{label}]  best: {best_p}  val_RMSE={best_rmse:.5f}')
    return best


# ── LightGBM ──────────────────────────────────────────────────────────────────

LGBM_GRID = {
    'num_leaves':        [63, 127, 255],
    'min_child_samples': [20, 50],
}


def tune_lgbm(seg_data: dict, SEG_CONFIG: dict) -> dict:
    if not HAS_LGBM:
        print('lightgbm не встановлено')
        return {}
    print('\n' + '='*60)
    print('Tuning LGBM  (early stopping на VAL, grid по листям)')
    print('='*60)
    best = {}
    grid = list(itertools.product(LGBM_GRID['num_leaves'], LGBM_GRID['min_child_samples']))

    for key, data in seg_data.items():
        if data is None:
            continue
        label = SEG_CONFIG[key]['label']
        med   = data['X_train'].median(numeric_only=True)
        X_tr  = _fill(data['X_train'], med)
        X_va  = _fill(data['X_val'],   med)
        y_tr  = np.log1p(data['y_train'].values)
        y_va  = np.log1p(data['y_val'].values)

        best_rmse, best_p, best_ntree = np.inf, {}, 500
        for n_leaves, min_child in grid:
            cbs = [lgb_lib.early_stopping(200, verbose=False), lgb_lib.log_evaluation(-1)]
            probe = LGBMRegressor(
                n_estimators=6000, learning_rate=0.02,
                num_leaves=n_leaves, min_child_samples=min_child,
                feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                reg_alpha=0.1, reg_lambda=1.5,
                n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
            )
            probe.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=cbs)
            n_tree = int(probe.best_iteration_ or probe.n_estimators)
            rmse = _val_rmse_log(y_va, probe.predict(X_va))
            if rmse < best_rmse:
                best_rmse = rmse
                best_ntree = n_tree
                best_p = {'num_leaves': n_leaves, 'min_child_samples': min_child,
                          'n_estimators': n_tree}

        best[key] = best_p
        print(f'  [{label}]  best: {best_p}  val_RMSE={best_rmse:.5f}')
    return best


# ── MLP ───────────────────────────────────────────────────────────────────────

MLP_GRID = {
    'hidden_layer_sizes': [(256, 128, 64), (512, 256, 128)],
    'learning_rate_init': [5e-4, 1e-3],
}


def tune_mlp(seg_data: dict, SEG_CONFIG: dict) -> dict:
    print('\n' + '='*60)
    print('Tuning MLP  (early stopping на VAL, grid по архітектурі)')
    print('='*60)
    best = {}
    grid = list(itertools.product(MLP_GRID['hidden_layer_sizes'], MLP_GRID['learning_rate_init']))

    for key, data in seg_data.items():
        if data is None:
            continue
        label = SEG_CONFIG[key]['label']
        X_tr_raw = data['X_train'].replace([np.inf, -np.inf], np.nan).fillna(0.).values
        X_va_raw = data['X_val'].replace([np.inf, -np.inf], np.nan).fillna(0.).values
        y_tr_log = np.log1p(np.clip(data['y_train'].values.astype(float), 0, None))
        y_va_log = np.log1p(np.clip(data['y_val'].values.astype(float), 0, None))

        x_sc = StandardScaler().fit(X_tr_raw)
        y_sc = StandardScaler().fit(y_tr_log.reshape(-1, 1))
        X_tr_s = x_sc.transform(X_tr_raw)
        X_va_s = x_sc.transform(X_va_raw)
        y_tr_s = y_sc.transform(y_tr_log.reshape(-1, 1)).ravel()
        y_va_s = y_sc.transform(y_va_log.reshape(-1, 1)).ravel()

        best_rmse, best_p = np.inf, {}
        for layers, lr in grid:
            mlp = MLPRegressor(
                hidden_layer_sizes=layers, activation='relu', solver='adam',
                learning_rate_init=lr, alpha=1e-4,
                max_iter=1, warm_start=True, random_state=RANDOM_STATE,
            )
            best_val, patience, best_w = np.inf, 0, None
            for _ in range(200):
                mlp.fit(X_tr_s, y_tr_s)
                vl = mean_squared_error(y_va_s, mlp.predict(X_va_s))
                if vl < best_val - 1e-6:
                    best_val = vl
                    best_w   = (mlp.coefs_[:], mlp.intercepts_[:])
                    patience = 0
                else:
                    patience += 1
                if patience >= 25:
                    break
            rmse = best_val ** 0.5
            if rmse < best_rmse:
                best_rmse = rmse
                best_p = {'hidden_layer_sizes': layers, 'learning_rate_init': lr}

        best[key] = best_p
        print(f'  [{label}]  best: {best_p}  val_RMSE={best_rmse:.5f}')
    return best


# ── Запуск усіх ───────────────────────────────────────────────────────────────

def tune_all(seg_data: dict, SEG_CONFIG: dict) -> dict:
    """
    Повертає:
    {
      'rf':   {(city,ctype): {params}},
      'xgb':  {(city,ctype): {params}},
      'lgbm': {(city,ctype): {params}},
      'mlp':  {(city,ctype): {params}},
    }
    """
    print('\n' + '█'*60)
    print('  HYPERPARAMETER TUNING — val-set grid search')
    print('█'*60)
    return {
        'rf':   tune_rf(seg_data, SEG_CONFIG),
        'xgb':  tune_xgb(seg_data, SEG_CONFIG),
        'lgbm': tune_lgbm(seg_data, SEG_CONFIG),
        'mlp':  tune_mlp(seg_data, SEG_CONFIG),
    }


def print_tuning_summary(best_params: dict) -> None:
    print('\n' + '='*60)
    print('  TUNING SUMMARY — best params per segment')
    print('='*60)
    for model, seg_dict in best_params.items():
        print(f'\n  [{model.upper()}]')
        for key, params in seg_dict.items():
            city, ctype = key
            print(f'    {city} — {ctype}: {params}')
