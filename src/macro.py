"""
macro.py — макроекономічні дані: курс USD/UAH, ставка НБУ, індекс будівництва,
           індекс споживчих цін на продукти, дані про повітряні тривоги.

Публічний API:
    load_nbu_rate()                    → pd.DataFrame (year_month, nbu_rate)
    load_exchange_rates(path)          → pd.DataFrame (year_month, usd_uah)
    load_construction_index(path)      → pd.DataFrame (year_month, construction_idx_*)
    load_food_price_index(path)        → pd.DataFrame (year_month, food_price_idx)
    build_macro_df()                   → об'єднаний DataFrame з усіма макро-фічами
    load_monthly_alerts()              → pd.DataFrame (city, year_month, alert_*)
    add_macro_features(df, macro)      → df з приєднаними макро-колонками
    add_alert_features(df, city, alerts) → df з приєднаними даними тривог
"""

import io
import re
import warnings

import numpy as np
import pandas as pd
import requests

from .config import (
    NBU_RATE_CHANGES, ALERTS_URL,
    PATH_EXCHANGE, PATH_CONSTR, PATH_CONSUMER,
)


# ── НБУ — облікова ставка ─────────────────────────────────────────────────────

def load_nbu_rate(
    changes=None,
    freq_start: str = '2023-01-01',
    freq_end:   str = '2026-12-01',
) -> pd.DataFrame:
    """Будує місячний ряд облікової ставки НБУ (step-function між змінами)."""
    if changes is None:
        changes = NBU_RATE_CHANGES

    df_ch = pd.DataFrame(changes, columns=['date', 'nbu_rate'])
    df_ch['date'] = pd.to_datetime(df_ch['date'])
    df_ch = df_ch.sort_values('date').reset_index(drop=True)

    months = pd.date_range(start=freq_start, end=freq_end, freq='MS')
    df_m = pd.DataFrame({'year_month': months})
    df_m['_month_end'] = df_m['year_month'] + pd.offsets.MonthEnd(0)

    rates = []
    for _, row in df_m.iterrows():
        applicable = df_ch[df_ch['date'] <= row['_month_end']]
        rates.append(applicable['nbu_rate'].iloc[-1] if len(applicable) else np.nan)

    df_m['nbu_rate'] = rates
    return df_m[['year_month', 'nbu_rate']]


# ── Курс USD/UAH ──────────────────────────────────────────────────────────────

def load_exchange_rates(path: str = PATH_EXCHANGE) -> pd.DataFrame:
    """Парсить файл НБУ (.xls) і повертає місячний ряд курсу USD/UAH."""
    xl = pd.ExcelFile(path)
    records = []
    for sheet in xl.sheet_names:
        df_xl = pd.read_excel(path, sheet_name=sheet, header=None)
        yr_row_idx = None
        for i in range(6):
            yr_vals = [v for v in df_xl.iloc[i]
                       if isinstance(v, (int, float)) and 1990 <= v <= 2030]
            if yr_vals:
                yr_row_idx = i
                break
        if yr_row_idx is None:
            continue
        usd_row = None
        for i in range(len(df_xl)):
            if 'долар США' in str(df_xl.iloc[i, 0]):
                usd_row = df_xl.iloc[i]
                break
        if usd_row is None:
            continue
        unit_m = re.search(r'(\d+)', str(usd_row.iloc[0]))
        unit = int(unit_m.group(1)) if unit_m else 1
        header_row = df_xl.iloc[yr_row_idx]
        year_positions = [
            (int(v), j) for j, v in enumerate(header_row)
            if isinstance(v, (int, float)) and 1990 <= v <= 2030 and j > 0
        ]
        seen, year_starts = set(), {}
        for yr, col in year_positions:
            if yr not in seen:
                seen.add(yr)
                year_starts[yr] = col
        for yr, start_col in sorted(year_starts.items()):
            for m_offset in range(1, 13):
                col = start_col + m_offset - 1
                if col >= len(usd_row):
                    break
                val = usd_row.iloc[col]
                if pd.notna(val) and str(val).strip() not in ['-', '–', '…', '']:
                    try:
                        rate = float(str(val).replace(',', '.')) / unit
                        records.append({'year': yr, 'month': m_offset, 'usd_uah': rate})
                    except ValueError:
                        pass
    if not records:
        return pd.DataFrame(columns=['year_month', 'usd_uah'])
    df = pd.DataFrame(records)
    df['year_month'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    return df[['year_month', 'usd_uah']].sort_values('year_month').reset_index(drop=True)


# ── Індекс будівельних матеріалів ─────────────────────────────────────────────

def load_construction_index(path: str = PATH_CONSTR) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
    except Exception:
        return pd.DataFrame(columns=['year_month', 'construction_idx_residential', 'construction_idx_total'])

    df.columns = df.columns.astype(str).str.strip()
    date_col = next((c for c in df.columns if 'дата' in c.lower() or 'date' in c.lower()), None)
    if date_col is None:
        return pd.DataFrame(columns=['year_month', 'construction_idx_residential', 'construction_idx_total'])

    df['year_month'] = pd.to_datetime(df[date_col], errors='coerce').dt.to_period('M').dt.to_timestamp()

    res_col = next((c for c in df.columns if 'жит' in c.lower() or 'resid' in c.lower()), None)
    tot_col = next((c for c in df.columns if 'загал' in c.lower() or 'total' in c.lower()), None)

    out = pd.DataFrame({'year_month': df['year_month']})
    if res_col:
        out['construction_idx_residential'] = pd.to_numeric(df[res_col], errors='coerce')
    if tot_col:
        out['construction_idx_total'] = pd.to_numeric(df[tot_col], errors='coerce')
    return out.dropna(subset=['year_month']).sort_values('year_month').reset_index(drop=True)


# ── Індекс споживчих цін на продукти ─────────────────────────────────────────

def load_food_price_index(path: str = PATH_CONSUMER) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
    except Exception:
        return pd.DataFrame(columns=['year_month', 'food_price_idx'])

    df.columns = df.columns.astype(str).str.strip()
    date_col = next((c for c in df.columns if 'дата' in c.lower() or 'date' in c.lower()), None)
    val_col  = next((c for c in df.columns if 'індекс' in c.lower() or 'index' in c.lower()), None)

    if date_col is None or val_col is None:
        return pd.DataFrame(columns=['year_month', 'food_price_idx'])

    out = pd.DataFrame({
        'year_month':    pd.to_datetime(df[date_col], errors='coerce').dt.to_period('M').dt.to_timestamp(),
        'food_price_idx': pd.to_numeric(df[val_col], errors='coerce'),
    })
    return out.dropna(subset=['year_month']).sort_values('year_month').reset_index(drop=True)


# ── Зведена макро-таблиця ─────────────────────────────────────────────────────

def build_macro_df() -> pd.DataFrame:
    """Збирає всі макро-джерела в єдиний місячний DataFrame."""
    nbu      = load_nbu_rate()
    exchange = load_exchange_rates()
    constr   = load_construction_index()
    food     = load_food_price_index()

    macro = nbu.copy()
    for df in [exchange, constr, food]:
        if len(df) > 0:
            macro = macro.merge(df, on='year_month', how='left')

    print(f"macro_df: {len(macro)} місяців × {macro.shape[1]} колонок")
    print(f"Колонки: {list(macro.columns)}")
    return macro


# ── Дані про повітряні тривоги ────────────────────────────────────────────────

def load_monthly_alerts(url: str = ALERTS_URL) -> pd.DataFrame:
    """Завантажує дані тривог і агрегує по місяцях і містах."""
    r = requests.get(url, timeout=60)
    raw = pd.read_csv(io.StringIO(r.text))

    raw['started_at']  = pd.to_datetime(raw['started_at'],  utc=True)
    raw['finished_at'] = pd.to_datetime(raw['finished_at'], utc=True)
    raw['duration_h']  = (raw['finished_at'] - raw['started_at']).dt.total_seconds() / 3600

    oblast_map = {'Kyiv City': 'Київ', 'Lvivska oblast': 'Львів'}
    df = raw[
        (raw['level'] == 'oblast') & raw['oblast'].isin(oblast_map)
    ].copy()
    df['city'] = df['oblast'].map(oblast_map)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df['year_month'] = df['started_at'].dt.to_period('M').dt.to_timestamp()

    monthly = (
        df.groupby(['city', 'year_month'])
        .agg(
            alert_count_month      = ('started_at', 'count'),
            alert_duration_h_month = ('duration_h',  'sum'),
            alert_days_month       = ('started_at',  lambda x: x.dt.date.nunique()),
        )
        .reset_index()
        .sort_values(['city', 'year_month'])
    )
    monthly['alert_count_cumulative'] = (
        monthly.groupby('city')['alert_count_month'].cumsum()
    )
    monthly['alert_ratio'] = (
        monthly['alert_days_month'] / monthly['year_month'].dt.days_in_month
    )
    print(f"Тривог завантажено: Київ={len(monthly[monthly.city=='Київ'])} міс, "
          f"Львів={len(monthly[monthly.city=='Львів'])} міс")
    return monthly


# ── Приєднання макро-фіч до датасету ─────────────────────────────────────────

def add_macro_features(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['year_month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    macro_cols = ['year_month', 'usd_uah',
                  'construction_idx_residential', 'construction_idx_total',
                  'food_price_idx', 'nbu_rate']
    available = [c for c in macro_cols if c in macro.columns]
    df = df.merge(macro[available], on='year_month', how='left')
    df = df.drop(columns=['year_month'], errors='ignore')
    return df


def add_alert_features(df: pd.DataFrame, city: str, monthly_alerts: pd.DataFrame) -> pd.DataFrame:
    """Приєднує місячні дані тривог (по місту) до датасету."""
    df = df.copy()
    df['year_month'] = pd.to_datetime(df['date'], errors='coerce').dt.to_period('M').dt.to_timestamp()
    city_alerts = monthly_alerts[monthly_alerts['city'] == city][
        ['year_month', 'alert_count_month', 'alert_duration_h_month',
         'alert_days_month', 'alert_count_cumulative', 'alert_ratio']
    ]
    df = df.merge(city_alerts, on='year_month', how='left')
    df = df.drop(columns=['year_month'], errors='ignore')
    return df
