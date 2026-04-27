"""
pipeline.py — головний пайплайн: запускає весь ланцюг від сирих даних до навчених моделей.

Використання:
    from src.pipeline import run_full_pipeline
    seg_data, SEG_CONFIG, results = run_full_pipeline()

Або покроково:
    from src.pipeline import (
        step1_load, step2_clean, step3_features,
        step4_macro_geo, step5_merge, step6_preprocess,
        step7_train
    )
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from .config import SEGMENT_KEYS
from .data import load_raw, split_segments, drop_gaps
from .cleaning import clean_segment
from .features import run_feature_engineering
from .macro import build_macro_df, load_monthly_alerts, add_macro_features, add_alert_features
from .geo import add_geo_features, add_knn_price_feature, add_district_month_median, add_post_external_interactions
from .preprocessing import preprocess_features, build_seg_data_from_frames
from .models import train_ridge, train_rf, train_xgb, train_lgbm, train_mlp


# ── Крок 1: Завантаження ──────────────────────────────────────────────────────

def step1_load(path: str = None) -> tuple[pd.DataFrame, dict]:
    """Завантажує CSV і розбиває на 4 сегменти."""
    from .config import PATH_LISTINGS
    data = load_raw(path or PATH_LISTINGS)
    segments = split_segments(data)
    return data, segments


# ── Крок 2: Очищення ──────────────────────────────────────────────────────────

def step2_clean(segments: dict) -> dict:
    """Дедуплікація + фізичні аномалії + цінові outlier-и для кожного сегменту."""
    print('\n' + '=' * 55)
    print('ОЧИЩЕННЯ ДАНИХ')
    print('=' * 55)
    clean = {}
    for (city, ctype), df in segments.items():
        label = f'{city} — {ctype}'
        clean[(city, ctype)] = clean_segment(df, ctype, label)
    return clean


# ── Крок 3: Feature Engineering ───────────────────────────────────────────────

def step3_features(clean_segments: dict) -> dict:
    """Часові, структурні, будівельні фічі для кожного сегменту."""
    print('\n' + '=' * 55)
    print('FEATURE ENGINEERING')
    print('=' * 55)
    feat = {}
    for (city, ctype), df in clean_segments.items():
        label = f'{city} — {ctype}'
        feat[(city, ctype)] = run_feature_engineering(df, ctype, label)
    return feat


# ── Крок 4: Макро + Гео + Тривоги ────────────────────────────────────────────

def step4_macro_geo(feat_segments: dict, load_geo: bool = True) -> dict:
    """
    Приєднує макро-дані (НБУ, курс, будівництво, продукти),
    дані тривог і геопросторові фічі.
    load_geo=False — пропускає Overpass API (швидко, без інтернету).
    """
    print('\n' + '=' * 55)
    print('МАКРО / ГЕО / ТРИВОГИ')
    print('=' * 55)

    macro = build_macro_df()

    try:
        alerts = load_monthly_alerts()
    except Exception as e:
        print(f'  Тривоги не завантажено: {e}')
        alerts = None

    enriched = {}
    for (city, ctype), df in feat_segments.items():
        label = f'{city} — {ctype}'
        df = add_macro_features(df, macro)
        if alerts is not None:
            df = add_alert_features(df, city, alerts)
        if load_geo:
            df = add_geo_features(df, city)
            df = add_knn_price_feature(df, ctype, label=label)
            df = add_district_month_median(df, ctype)
        df = add_post_external_interactions(df)
        enriched[(city, ctype)] = df
        print(f'  [{label}] → {df.shape[1]} колонок')
    return enriched


# ── Крок 5: Об'єднання + drop gaps ───────────────────────────────────────────

def step5_merge(enriched_segments: dict) -> pd.DataFrame:
    """Збирає 4 сегменти в єдиний df_feat і видаляє GAP-місяці."""
    parts = []
    for (city, ctype), df in enriched_segments.items():
        t = df.copy()
        t['segment'] = f'{city} — {ctype}'
        parts.append(t)
    df_feat = pd.concat(parts, ignore_index=True)
    df_feat['date'] = pd.to_datetime(df_feat['date'], errors='coerce')
    n_before = len(df_feat)
    df_feat = drop_gaps(df_feat)
    print(f'\ndf_feat: {n_before:,} → {len(df_feat):,} рядків (GAP видалено)')
    return df_feat


# ── Крок 6: Препроцесинг + побудова матриць ───────────────────────────────────

def step6_preprocess(enriched_segments: dict) -> tuple[dict, dict]:
    """
    Запускає preprocess_features для кожного сегменту і будує
    train/val/test матриці через build_seg_data_from_frames.
    """
    print('\n' + '=' * 55)
    print('PREPROCESSING + TEMPORAL SPLIT')
    print('=' * 55)
    processed = {k: preprocess_features(v) for k, v in enriched_segments.items()}
    seg_data, SEG_CONFIG = build_seg_data_from_frames(processed)
    return seg_data, SEG_CONFIG


# ── Крок 7: Навчання моделей ──────────────────────────────────────────────────

def step7_train(
    seg_data: dict,
    SEG_CONFIG: dict,
    models: list = None,
) -> dict:
    """
    Навчає вибрані моделі.
    models: список з ['ridge', 'rf', 'xgb', 'lgbm', 'mlp'].
            За замовчуванням — всі.
    Повертає dict {'ridge': results, 'xgb': results, ...}
    """
    if models is None:
        models = ['ridge', 'rf', 'xgb', 'lgbm', 'mlp']

    trainers = {
        'ridge': train_ridge,
        'rf':    train_rf,
        'xgb':  train_xgb,
        'lgbm': train_lgbm,
        'mlp':  train_mlp,
    }

    all_results = {}
    for name in models:
        if name not in trainers:
            print(f'Невідома модель: {name!r}. Доступні: {list(trainers)}')
            continue
        print(f'\n{"="*72}')
        all_results[name] = trainers[name](seg_data, SEG_CONFIG)

    return all_results


# ── Повний пайплайн ───────────────────────────────────────────────────────────

def run_full_pipeline(
    path: str = None,
    load_geo: bool = False,
    models: list = None,
) -> tuple[dict, dict, dict]:
    """
    Запускає всі кроки від завантаження до навчених моделей.

    Args:
        path:     шлях до CSV (за замовчуванням із config.py)
        load_geo: True — завантажувати POI з Overpass API (повільно)
        models:   список моделей для навчання, наприклад ['xgb', 'lgbm']

    Returns:
        (seg_data, SEG_CONFIG, all_results)
    """
    _, segments       = step1_load(path)
    clean             = step2_clean(segments)
    feat              = step3_features(clean)
    enriched          = step4_macro_geo(feat, load_geo=load_geo)
    seg_data, cfg     = step6_preprocess(enriched)
    all_results       = step7_train(seg_data, cfg, models)
    return seg_data, cfg, all_results
