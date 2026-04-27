"""
cleaning.py — очищення даних: дедуплікація, фізичні аномалії, цінові outlier-и.

Публічний API:
    diagnose_duplicates(data)                      → (time_spread_df, group_sizes_df)
    compare_gap_strategies(data)                   → results_df
    deduplicate_monthly(df, label)                 → df_clean
    filter_anomalies(df, contract_type, label)     → df_clean
    filter_price_outliers(df, contract_type, label)→ df_clean
    clean_segment(df, contract_type, label)        → df повністю очищений
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import TRAIN_END

# ── Параметри фільтрації ───────────────────────────────────────────────────────
FILTERS_COMMON = {
    'area_total':  (10, 400),
    'floor':       (1,  50),
    'floor_count': (1,  50),
    'room_count':  (1,  10),
}

PRICE_BOUNDS = {
    'Оренда': (0.005, 0.995),
    'Продаж': (0.005, 0.995),
}

HARD_BOUNDS = {
    'price_uah':    (3_000, 500_000),   # грн/міс
    'price_m2_usd': (150,   7_500),     # USD/м²
}

_ID_GEO  = ['latitude', 'longitude', 'area_total', 'floor', 'contract_type']
_ID_ADDR = ['city', 'district', 'street', 'house_number', 'area_total', 'floor', 'contract_type']


# ── Діагностика ────────────────────────────────────────────────────────────────

def diagnose_duplicates(data: pd.DataFrame):
    dup_mask = data.duplicated(subset=_ID_GEO, keep=False)
    dups = data[dup_mask].copy()
    print(f"Записів з потенційними дублікатами: {len(dups):,} ({100*len(dups)/len(data):.1f}%)")

    cross = (dups.groupby(_ID_GEO)['site'].nunique()
             .reset_index(name='n_sites')
             .query('n_sites > 1'))
    print(f"Груп на кількох сайтах (справжні дублікати): {len(cross):,}")

    time_spread = (
        dups.groupby(_ID_GEO)['date']
        .agg(min_date='min', max_date='max', n_dates='nunique')
        .reset_index()
    )
    time_spread['days_span'] = (time_spread['max_date'] - time_spread['min_date']).dt.days
    print(f"Груп з різницею > 180 днів: {(time_spread['days_span'] > 180).sum():,}")
    print(f"Груп з різницею ≤ 30 днів:  {(time_spread['days_span'] <= 30).sum():,}")

    group_sizes = data.groupby(_ID_GEO).size().reset_index(name='count')
    print("\nРозподіл кількості копій на групу:")
    print(group_sizes['count'].value_counts().head(10))
    return time_spread, group_sizes


def compare_gap_strategies(data: pd.DataFrame):
    """Порівнює різні пороги gap_days та їх вплив на розмір датасету."""
    df = data.sort_values('date').reset_index(drop=True)
    df['_prev_date'] = df.groupby(_ID_GEO)['date'].shift(1)
    df['_days_since'] = (df['date'] - df['_prev_date']).dt.days

    rows = []
    for gap in [14, 30, 60, 90, 120, 180, 365]:
        n_removed = (df['_prev_date'].notna() & (df['_days_since'] <= gap)).sum()
        rows.append({
            'gap_days': gap,
            'removed': n_removed,
            'remaining': len(df) - n_removed,
            'pct_removed': 100 * n_removed / len(df),
        })
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.bar(result['gap_days'].astype(str), result['remaining'],
            color='steelblue', alpha=0.7, label='Залишилось записів')
    ax2.plot(result['gap_days'].astype(str), result['pct_removed'],
             color='red', marker='o', linewidth=2, label='% видалено')
    ax1.set_xlabel('Поріг gap_days')
    ax1.set_ylabel('Кількість записів', color='steelblue')
    ax2.set_ylabel('% видалено', color='red')
    ax1.set_title('Вплив порогу gap_days на розмір датасету')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout()
    plt.savefig('dedup_strategy_comparison.png', dpi=150)
    plt.show()
    return result


# ── Дедуплікація ───────────────────────────────────────────────────────────────

def deduplicate_monthly(df: pd.DataFrame, label: str = '') -> pd.DataFrame:
    """
    Для кожного об'єкта і кожного місяця залишає лише ОСТАННІЙ запис.
    Рядки без достатніх ідентифікаторів не змінюються.
    """
    n_before = len(df)
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['_year_month'] = df['date'].dt.to_period('M')

    has_geo = df['latitude'].notna() & df['longitude'].notna()
    df_geo  = df[has_geo].copy()
    df_addr = df[~has_geo].copy()

    def keep_last(d, id_cols):
        available = [c for c in id_cols if c in d.columns]
        if len(available) < len(id_cols) or len(d) == 0:
            return d
        has_id    = d[available].notna().all(axis=1)
        d_with_id = d[has_id].copy()
        d_no_id   = d[~has_id].copy()
        group_cols = available + ['_year_month']
        d_deduped = (d_with_id
                     .sort_values('date')
                     .groupby(group_cols, observed=True)
                     .last()
                     .reset_index())
        return pd.concat([d_deduped, d_no_id], ignore_index=True)

    df_geo_dedup  = keep_last(df_geo,  _ID_GEO)
    df_addr_dedup = keep_last(df_addr, _ID_ADDR)
    result = pd.concat([df_geo_dedup, df_addr_dedup], ignore_index=True)
    result = result.drop(columns=['_year_month'], errors='ignore')

    n_removed = n_before - len(result)
    print(f"  [{label}] дедуплікація: {n_before:,} → {len(result):,} "
          f"(видалено {n_removed:,}, {100*n_removed/n_before:.1f}%)")
    return result


# ── Фізичні аномалії ───────────────────────────────────────────────────────────

def filter_anomalies(df: pd.DataFrame, contract_type: str, label: str = '') -> pd.DataFrame:
    """Видаляє рядки з фізично неможливими значеннями (площа, поверх, кімнати)."""
    mask = pd.Series(True, index=df.index)
    n_before = len(df)
    removed_by_col = {}

    for col, (lo, hi) in FILTERS_COMMON.items():
        if col not in df.columns:
            continue
        col_num  = pd.to_numeric(df[col], errors='coerce')
        col_mask = col_num.between(lo, hi) | col_num.isna()
        n_rm = (~col_mask).sum()
        if n_rm > 0:
            removed_by_col[col] = n_rm
        mask &= col_mask

    df_out = df[mask].copy().reset_index(drop=True)
    n_removed = n_before - len(df_out)

    print(f"\n  ┌─ {label}")
    for col, n in removed_by_col.items():
        lo, hi = FILTERS_COMMON[col]
        print(f"  │  {col:<18} поза [{lo:>3}, {hi:>3}] → видалено {n:,}")
    print(f"  │  Всього видалено: {n_removed:,}  ({100*n_removed/n_before:.1f}%)")
    print(f"  └─ Залишилось:     {len(df_out):,}")
    return df_out


# ── Цінові outlier-и ───────────────────────────────────────────────────────────

def filter_price_outliers(df: pd.DataFrame, contract_type: str, label: str = '') -> pd.DataFrame:
    """
    Квантильне обрізання [0.5%, 99.5%] — квантилі розраховуються ТІЛЬКИ на train.
    Межі потім застосовуються до всього df (без витоку з test).
    """
    df = df.copy()
    n0 = len(df)
    price_col = 'price_m2_usd' if contract_type == 'Продаж' else 'price_uah'
    if price_col not in df.columns:
        print(f'  [{label}] пропущено: немає {price_col}')
        return df

    p = pd.to_numeric(df[price_col], errors='coerce')
    train_mask = pd.to_datetime(df['date'], errors='coerce') <= TRAIN_END
    p_train = p[train_mask].dropna()

    lo_q, hi_q = PRICE_BOUNDS[contract_type]
    if len(p_train) < 100:
        print(f'  [{label}] WARNING: train має лише {len(p_train)} рядків — фолбек на весь df')
        q_lo, q_hi = p.quantile([lo_q, hi_q]).values
    else:
        q_lo, q_hi = p_train.quantile([lo_q, hi_q]).values

    hb_lo, hb_hi = HARD_BOUNDS[price_col]
    lo = max(q_lo, hb_lo)
    hi = min(q_hi, hb_hi)

    mask  = p.between(lo, hi)
    df_out = df[mask].reset_index(drop=True)
    print(f'  [{label}] {price_col}: [{lo:,.0f}, {hi:,.0f}]  '
          f'→ видалено {n0-len(df_out):,} ({100*(n0-len(df_out))/n0:.1f}%)  '
          f'залишилось {len(df_out):,}')
    return df_out


# ── Комплексне очищення сегменту ───────────────────────────────────────────────

def clean_segment(df: pd.DataFrame, contract_type: str, label: str = '') -> pd.DataFrame:
    """Виконує повний цикл очищення: dedup → фізичні аномалії → цінові outlier-и."""
    df = deduplicate_monthly(df, label)
    df = filter_anomalies(df, contract_type, label)
    df = filter_price_outliers(df, contract_type, label)
    return df
