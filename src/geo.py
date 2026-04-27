"""
geo.py — геопросторові фічі: відстані до POI, KNN-ціна, просторово-часові лаги,
         взаємодії (H1-інтеракції) після зовнішніх даних.

Публічний API:
    haversine_vec(lat1, lon1, lat2_arr, lon2_arr) → np.ndarray (км)
    dist_to_nearest(lat, lon, poi_lats, poi_lons) → float (км)
    count_within_radius(lat, lon, poi_lats, poi_lons, radius_km) → int
    overpass_query(query)                          → list[dict]
    add_geo_features(df, city)                     → df з dist_to_center_km та POI
    add_knn_price_feature(df, contract_type, train_cutoff, k, label) → df
    add_district_month_median(df, contract_type, train_cutoff)       → df
    add_post_external_interactions(df)             → df з H1-взаємодіями
"""

import time

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

from .config import CITY_CENTERS, CITY_BBOX, TRAIN_END


# ── Гаверсинова відстань ──────────────────────────────────────────────────────

def haversine_vec(lat1: float, lon1: float,
                  lat2_arr, lon2_arr) -> np.ndarray:
    R = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2 = np.radians(np.asarray(lat2_arr, dtype=float))
    lon2 = np.radians(np.asarray(lon2_arr, dtype=float))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def dist_to_nearest(lat, lon, poi_lats, poi_lons) -> float:
    if pd.isna(lat) or pd.isna(lon) or len(poi_lats) == 0:
        return np.nan
    return haversine_vec(lat, lon, poi_lats, poi_lons).min()


def count_within_radius(lat, lon, poi_lats, poi_lons, radius_km: float = 1.0) -> int:
    if pd.isna(lat) or pd.isna(lon) or len(poi_lats) == 0:
        return np.nan
    return int((haversine_vec(lat, lon, poi_lats, poi_lons) <= radius_km).sum())


# ── Overpass API ──────────────────────────────────────────────────────────────

def overpass_query(query: str, retries: int = 3) -> list:
    for attempt in range(retries):
        try:
            time.sleep(2)
            r = requests.post(
                'https://overpass-api.de/api/interpreter',
                data={'data': query}, timeout=60,
            )
            if r.status_code == 200 and r.text.strip():
                return r.json().get('elements', [])
        except Exception as e:
            print(f'  Overpass attempt {attempt+1}: {e}')
    return []


def _extract_coords(elements: list) -> tuple[np.ndarray, np.ndarray]:
    lats, lons = [], []
    for e in elements:
        lat = e.get('lat') or (e.get('center') or {}).get('lat')
        lon = e.get('lon') or (e.get('center') or {}).get('lon')
        if lat and lon:
            lats.append(lat)
            lons.append(lon)
    return np.array(lats), np.array(lons)


# ── Геопросторові фічі ────────────────────────────────────────────────────────

def add_geo_features(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Додає відстань до центру міста і, якщо є координати POI (метро, укриття),
    відстань до них.  POI завантажуються через Overpass API.
    """
    df = df.copy()
    if city not in CITY_CENTERS:
        return df

    center_lat, center_lon = CITY_CENTERS[city]
    lats = pd.to_numeric(df.get('latitude'),  errors='coerce').values
    lons = pd.to_numeric(df.get('longitude'), errors='coerce').values

    df['dist_to_center_km'] = np.where(
        ~np.isnan(lats) & ~np.isnan(lons),
        haversine_vec(center_lat, center_lon, lats, lons),
        np.nan,
    )

    s, w, n, e = CITY_BBOX[city]
    bbox = f'{s},{w},{n},{e}'

    # Метро (тільки Київ)
    if city == 'Київ':
        subway_q = f'[out:json][timeout:60];node["railway"="station"]["station"="subway"]({bbox});out;'
        elements = overpass_query(subway_q)
        if elements:
            sub_lats, sub_lons = _extract_coords(elements)
            df['dist_to_subway_km'] = [
                dist_to_nearest(la, lo, sub_lats, sub_lons) for la, lo in zip(lats, lons)
            ]
            df['subway_count_1km'] = [
                count_within_radius(la, lo, sub_lats, sub_lons, 1.0) for la, lo in zip(lats, lons)
            ]

    # Укриття
    shelter_q = (
        f'[out:json][timeout:60];'
        f'(node["amenity"="shelter"]({bbox});'
        f'way["amenity"="shelter"]({bbox}););out center;'
    )
    elements = overpass_query(shelter_q)
    if elements:
        sh_lats, sh_lons = _extract_coords(elements)
        df['dist_to_shelter_km'] = [
            dist_to_nearest(la, lo, sh_lats, sh_lons) for la, lo in zip(lats, lons)
        ]

    return df


# ── KNN-ціна сусідів ──────────────────────────────────────────────────────────

def add_knn_price_feature(
    df: pd.DataFrame,
    contract_type: str,
    train_cutoff=None,
    k: int = 10,
    label: str = '',
) -> pd.DataFrame:
    """
    Медіана price/m² k найближчих TRAIN-лістингів у просторі
    (lat, lon, area_total, room_count) з z-score нормалізацією.
    Для train-рядків самого об'єкта виключають з пошуку (no self-match).
    """
    if train_cutoff is None:
        train_cutoff = TRAIN_END

    df = df.copy()
    price_col = 'price_m2_usd' if contract_type == 'Продаж' else 'price_m2_uah'
    knn_feats = ['latitude', 'longitude', 'area_total', 'room_count']

    if price_col not in df.columns or 'latitude' not in df.columns:
        print(f'  [{label}] SKIP knn: немає {price_col} або координат')
        return df

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    cutoff = pd.Timestamp(train_cutoff)

    for c in knn_feats + [price_col]:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    valid_all   = df[knn_feats + [price_col]].notna().all(axis=1)
    valid_train = valid_all & (df['date'] <= cutoff)

    if valid_train.sum() < k * 2:
        print(f'  [{label}] SKIP knn: замало train-рядків ({valid_train.sum()} valid)')
        return df

    mu = df.loc[valid_train, knn_feats].mean()
    sd = df.loc[valid_train, knn_feats].std().replace(0, 1)

    X_all   = ((df.loc[valid_all,   knn_feats] - mu) / sd).values
    X_train = ((df.loc[valid_train, knn_feats] - mu) / sd).values
    y_train = np.log1p(df.loc[valid_train, price_col].values)

    tree = BallTree(X_train)
    all_idx  = df.index[valid_all]
    train_idx_set = set(df.index[valid_train])

    knn_prices = np.full(len(df), np.nan)
    dists, inds = tree.query(X_all, k=k + 1)

    for i, (row_idx, nn_inds) in enumerate(zip(all_idx, inds)):
        is_train_row = row_idx in train_idx_set
        valid_nn = [j for j in nn_inds if j < len(y_train)]
        if is_train_row and valid_nn:
            valid_nn = valid_nn[1:]  # exclude self
        if valid_nn:
            knn_prices[df.index.get_loc(row_idx)] = np.median(y_train[valid_nn])

    df['knn_price_m2'] = knn_prices
    print(f'  [{label}] knn_price_m2: {(~np.isnan(knn_prices)).sum():,} заповнено')
    return df


# ── Просторово-часовий лаг ────────────────────────────────────────────────────

def add_district_month_median(
    df: pd.DataFrame,
    contract_type: str,
    train_cutoff=None,
) -> pd.DataFrame:
    """
    TRAIN-only медіана price/m² по (district, month), зсунута на -1 місяць.
    Test-рядки отримують останню train-медіану (forward-fill по district).
    """
    if train_cutoff is None:
        train_cutoff = TRAIN_END

    df = df.copy()
    price_col = 'price_m2_usd' if contract_type == 'Продаж' else 'price_m2_uah'
    if price_col not in df.columns or 'district' not in df.columns:
        return df

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['_ym'] = df['date'].dt.to_period('M').dt.to_timestamp()
    cutoff = pd.Timestamp(train_cutoff)
    train_df = df[df['date'] <= cutoff].copy()

    monthly_med = (
        train_df.groupby(['district', '_ym'])[price_col]
        .median()
        .reset_index()
        .rename(columns={price_col: 'district_month_median', '_ym': '_ym_lag'})
    )
    monthly_med['_ym_lag'] = monthly_med['_ym_lag'] + pd.DateOffset(months=1)

    df = df.merge(monthly_med, left_on=['district', '_ym'],
                  right_on=['district', '_ym_lag'], how='left')
    df = df.drop(columns=['_ym', '_ym_lag'], errors='ignore')
    return df


# ── H1-інтеракції ─────────────────────────────────────────────────────────────

_ERA_ORDER = {
    'Сталінка (до 1960)':      4,
    'Хрущовка (1960-1975)':    2,
    'Брежнєвка (1976-1991)':   3,
    'Пострадянська (1992-2010)': 1,
    'Новобудова (після 2010)': 0,
    'Невідомо':                 2,
}


def add_post_external_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Додає взаємодії між зовнішніми факторами (макро / geo / alerts).
    Викликати ПІСЛЯ того, як макро- і alert-фічі вже приєднані.
    """
    from .config import WAR_START
    df = df.copy()

    if 'date' in df.columns:
        df['is_wartime'] = (
            pd.to_datetime(df['date'], errors='coerce') >= WAR_START
        ).astype(int)

    def _num(col):
        return pd.to_numeric(df.get(col), errors='coerce').fillna(0)

    if 'autonomy_score' in df.columns and 'alert_count_month' in df.columns:
        df['autonomy_x_alerts'] = _num('autonomy_score') * _num('alert_count_month')

    if 'autonomy_score' in df.columns and 'is_wartime' in df.columns:
        df['autonomy_x_wartime'] = _num('autonomy_score') * df['is_wartime']

    if 'is_new_building' in df.columns and 'usd_uah' in df.columns:
        df['newbuild_x_usd'] = _num('is_new_building') * _num('usd_uah')

    if 'dist_to_shelter_km' in df.columns and 'alert_count_month' in df.columns:
        df['shelter_x_alerts'] = _num('dist_to_shelter_km') * _num('alert_count_month')

    if 'days_since_war' in df.columns and 'usd_uah' in df.columns:
        dsw = pd.to_numeric(df['days_since_war'], errors='coerce').clip(lower=1)
        usd = pd.to_numeric(df['usd_uah'],        errors='coerce').clip(lower=1)
        df['log_days_x_log_usd'] = np.log1p(dsw) * np.log1p(usd)

    if 'building_era' in df.columns and 'dist_to_center_km' in df.columns:
        era_ord = df['building_era'].map(_ERA_ORDER).fillna(2)
        df['era_ord_x_center'] = era_ord * _num('dist_to_center_km')

    return df
