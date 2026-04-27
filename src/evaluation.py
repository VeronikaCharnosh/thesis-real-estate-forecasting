"""
evaluation.py — оцінка якості моделей, порівняльна таблиця, SHAP, ablation (H1).

Публічний API:
    evaluate_model(y_true, y_pred, model_name, use_log) → dict
    baseline_median_segment(df_train, df_test, target)  → (preds, metrics)
    compare_models(model_collections, SEG_CONFIG)       → pd.DataFrame
    plot_shap(model, X_test, title)                     → None
    run_ablation(df_feat, SEG_CONFIG, seg_datasets)     → pd.DataFrame
    plot_loss_curves(seg_data, SEG_CONFIG)              → None
    diagnose_worst_residuals(results, SEG_CONFIG, n)    → None
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import RANDOM_STATE, TRAIN_END, TEST_START


# ── Базові метрики ─────────────────────────────────────────────────────────────

def evaluate_model(
    y_true, y_pred,
    model_name: str = 'Model',
    use_log: bool = True,
) -> dict:
    """Повертає dict з MAE, RMSE, R², MAPE та друкує зведений рядок."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    if use_log:
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-9))) * 100

    print(f"\n{'─'*40}")
    print(f"  {model_name}")
    print(f"{'─'*40}")
    print(f"  MAE  (log):  {mae:.4f}")
    print(f"  RMSE (log):  {rmse:.4f}")
    print(f"  R²:          {r2:.4f}")
    print(f"  MAPE (orig): {mape:.2f}%")
    print(f"{'─'*40}")
    return {'model': model_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE_%': mape}


def _wape_mape_trim(y_true, y_pred, trim_q: float = 0.02):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) == 0:
        return float('nan'), float('nan')
    wape = 100. * np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-12)
    thr  = np.quantile(y_true, trim_q)
    sub  = y_true >= max(thr, 1.)
    if not np.any(sub):
        return wape, float('nan')
    mape_t = 100. * np.mean(np.abs((y_true[sub] - y_pred[sub]) / y_true[sub]))
    return wape, mape_t


# ── Baseline: медіана по сегменту ─────────────────────────────────────────────

def baseline_median_segment(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    segment_col: str = 'segment',
) -> tuple[pd.Series, dict]:
    """Базова лінія: медіана таргету по сегменту з train."""
    medians = df_train.groupby(segment_col)[target].median()
    global_med = df_train[target].median()
    preds = df_test[segment_col].map(medians).fillna(global_med)
    y_te  = df_test[target].values
    mae   = mean_absolute_error(y_te, preds)
    rmse  = np.sqrt(mean_squared_error(y_te, preds))
    mape  = np.mean(np.abs((y_te - preds.values) / (y_te + 1e-9))) * 100
    print(f'Baseline (median)  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%')
    return preds, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


# ── Порівняльна таблиця моделей ───────────────────────────────────────────────

def compare_models(
    model_collections: list[tuple[str, dict]],
    SEG_CONFIG: dict,
) -> pd.DataFrame:
    """
    model_collections: [('Ridge', ridge_results), ('XGB', xgb_results), ...]
    Повертає DataFrame з метриками всіх моделей і сегментів.
    """
    rows = []
    for model_name, res_dict in model_collections:
        for (city, ctype), r in res_dict.items():
            if r is None:
                continue
            cfg  = SEG_CONFIG[(city, ctype)]
            wape, mape_t = _wape_mape_trim(r['y_test'], r['y_pred'])
            rows.append({
                'Модель':       model_name,
                'Сегмент':      cfg['label'],
                'MAE':          round(r['MAE'],  2),
                'RMSE':         round(r['RMSE'], 2),
                'R²':           round(r['R2'],   4),
                'MAPE%':        round(r['MAPE'], 2),
                'WAPE%':        round(wape, 2),
                'MAPE_trim%':   round(mape_t, 2),
            })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ── SHAP ─────────────────────────────────────────────────────────────────────

def plot_shap(model, X_test: pd.DataFrame, title: str = 'SHAP Summary') -> None:
    try:
        import shap
        sample = X_test.sample(min(1500, len(X_test)), random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(sample)
        shap.summary_plot(sv, sample, show=False, max_display=20)
        plt.title(title)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print('shap не встановлено: pip install shap')
    except Exception as e:
        print(f'SHAP: {repr(e)}')


# ── Ablation H1 (A / B / C) ──────────────────────────────────────────────────

INTERNAL_FEATS = [
    'room_count', 'area_total', 'area_living', 'area_kitchen',
    'floor', 'floor_count', 'floor_ratio', 'is_first_floor', 'is_last_floor',
    'ceiling_height', 'building_age', 'is_new_building', 'built_year',
    'autonomy_score', 'has_furniture', 'has_balcony', 'has_parking',
    'has_gas', 'amenity_score', 'is_without_renovation', 'is_babushka_renovation',
    'year', 'month', 'quarter', 'dist_to_center_km',
]
MACRO_FEATS = [
    'usd_uah', 'nbu_rate', 'construction_idx_residential',
    'construction_idx_total', 'food_price_idx', 'days_since_war',
]
SECURITY_FEATS = [
    'alert_count_month', 'alert_duration_h_month', 'alert_days_month',
    'alert_count_cumulative', 'alert_ratio',
    'dist_to_shelter_km', 'dist_to_subway_km', 'subway_count_1km',
    'is_wartime',
    'autonomy_x_alerts', 'autonomy_x_wartime', 'newbuild_x_usd',
    'shelter_x_alerts', 'log_days_x_log_usd', 'era_ord_x_center',
]

ABLATION_CONFIGS = {
    'A: internal':        INTERNAL_FEATS,
    'B: +macro':          INTERNAL_FEATS + MACRO_FEATS,
    'C: +macro+security': INTERNAL_FEATS + MACRO_FEATS + SECURITY_FEATS,
}


def run_ablation(
    df_feat: pd.DataFrame,
    SEG_CONFIG: dict,
    model_factory=None,
) -> pd.DataFrame:
    """
    A/B/C ablation — доказ гіпотези H1.
    model_factory: callable() → sklearn-модель. За замовчуванням — XGBoost.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print('xgboost потрібен для ablation')
        return pd.DataFrame()

    if model_factory is None:
        def model_factory():
            return XGBRegressor(
                n_estimators=800, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
                tree_method='hist',
            )

    from .data import split_mask

    rows = []
    df = df_feat.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    for (city, ctype), cfg in SEG_CONFIG.items():
        target = cfg['target']
        seg = df[
            (df.get('city', pd.Series([''] * len(df))) == city) &
            (df['contract_type'] == ctype) &
            df[target].notna()
        ].copy() if 'city' in df.columns else df[df['contract_type'] == ctype].copy()

        if len(seg) < 200:
            continue

        m_tr = split_mask(seg, 'train')
        m_te = split_mask(seg, 'test')

        for config_name, feat_list in ABLATION_CONFIGS.items():
            feats = [f for f in feat_list if f in seg.columns]
            if not feats:
                continue

            X_tr = seg.loc[m_tr, feats].replace([np.inf, -np.inf], np.nan).fillna(0.)
            X_te = seg.loc[m_te, feats].replace([np.inf, -np.inf], np.nan).fillna(0.)
            y_tr = np.log1p(seg.loc[m_tr, target].values)
            y_te = seg.loc[m_te, target].values

            if len(X_tr) < 100 or len(X_te) < 10:
                continue

            m = model_factory()
            m.fit(X_tr, y_tr)
            y_pred = np.expm1(m.predict(X_te))

            mae  = mean_absolute_error(y_te, y_pred)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            r2   = r2_score(np.log1p(y_te), np.log1p(y_pred.clip(0)))
            mape = np.mean(np.abs((y_te - y_pred) / (y_te + 1e-9))) * 100
            rows.append({
                'Сегмент': cfg['label'], 'Config': config_name,
                'n_feats': len(feats), 'MAE': round(mae, 2),
                'RMSE': round(rmse, 2), 'R²': round(r2, 4), 'MAPE%': round(mape, 2),
            })
            print(f'  [{cfg["label"]}]  {config_name}  MAE={mae:.2f}  R²={r2:.4f}  MAPE={mape:.2f}%')

    result = pd.DataFrame(rows)
    print('\n', result.to_string(index=False))
    return result


# ── Діагностика найгірших залишків ────────────────────────────────────────────

def diagnose_worst_residuals(results: dict, SEG_CONFIG: dict, n: int = 20) -> None:
    print('=' * 78)
    print(f'TOP-{n} worst residuals per segment')
    print('=' * 78)
    for (city, ctype), r in results.items():
        if r is None:
            continue
        cfg  = SEG_CONFIG[(city, ctype)]
        y_te = r['y_test']
        y_pr = r['y_pred']
        ape  = np.abs((y_te - y_pr) / (y_te + 1e-9)) * 100
        order = np.argsort(ape)[::-1][:n]
        print(f'\n  [{cfg["label"]}]  median APE={np.median(ape):.2f}%  '
              f'p95={np.percentile(ape, 95):.2f}%  p99={np.percentile(ape, 99):.2f}%')
        print(f'    {"y_actual":>14}  {"y_pred":>14}  {"APE%":>8}')
        for k in order[:8]:
            print(f'    {y_te[k]:>14,.0f}  {y_pr[k]:>14,.0f}  {ape[k]:>8.1f}%')

    print('\nRMSE/MAE ratio (>3 → outlier-dominated RMSE)')
    for (city, ctype), r in results.items():
        if r is None:
            continue
        cfg   = SEG_CONFIG[(city, ctype)]
        ratio = r['RMSE'] / max(r['MAE'], 1e-9)
        flag  = 'outliers' if ratio > 3 else ('OK' if ratio < 2.2 else 'marginal')
        print(f'  {cfg["label"]:<20}  RMSE/MAE = {ratio:>5.2f}  {flag}')


# ── Loss curves (train vs val) ────────────────────────────────────────────────

def plot_loss_curves(seg_data: dict, SEG_CONFIG: dict) -> None:
    """Малює train/val RMSE по ітераціях для XGBoost (per segment)."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print('xgboost потрібен')
        return

    seg_keys = [k for k, v in seg_data.items() if v is not None]
    fig, axes = plt.subplots(1, len(seg_keys), figsize=(5 * len(seg_keys), 4), sharey=False)
    if len(seg_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, seg_keys):
        d   = seg_data[key]
        cfg = SEG_CONFIG[key]
        X_tr = d['X_train'].replace([np.inf, -np.inf], np.nan)
        X_va = d['X_val'].replace([np.inf, -np.inf], np.nan)
        med  = X_tr.median(numeric_only=True)
        X_tr = X_tr.fillna(med).fillna(0.)
        X_va = X_va.fillna(med).fillna(0.)
        y_tr = np.log1p(d['y_train'].values)
        y_va = np.log1p(d['y_val'].values)

        m = XGBRegressor(
            n_estimators=2000, learning_rate=0.02, max_depth=8,
            subsample=0.85, colsample_bytree=0.85,
            n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
            tree_method='hist', objective='reg:squarederror', eval_metric='rmse',
            early_stopping_rounds=150,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], verbose=False)
        hist = m.evals_result()
        tr_rmse = hist['validation_0']['rmse']
        va_rmse = hist['validation_1']['rmse']
        best_iter = int(m.best_iteration or (len(tr_rmse) - 1))

        iters = np.arange(1, len(tr_rmse) + 1)
        ax.plot(iters, tr_rmse, label='train', color='#4e79a7', lw=1.8)
        ax.plot(iters, va_rmse, label='val',   color='#e15759', lw=1.8)
        ax.axvline(best_iter + 1, color='#c33', ls='--', lw=1.2, alpha=0.7,
                   label=f'best={best_iter+1}')
        ax.set_title(cfg['label'], fontsize=10, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMSE (log)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('XGBoost: Train / Val RMSE per Iteration', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
