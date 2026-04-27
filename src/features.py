"""
features.py
===========
Feature engineering for the Ukrainian real estate price-forecasting pipeline.

All transformations are deterministic (no statistical fitting) and therefore
carry zero risk of data leakage between train, validation, and test sets.

Feature groups
--------------
  Time            year, month, quarter, cyclical month encoding, wartime flag,
                  days elapsed since Russia's full-scale invasion.
  Floor           floor_ratio, is_first_floor, is_last_floor.
  Area            living_ratio, kitchen_ratio, area_per_room.
  Building        building_age, building_era (ordinal), is_new_building.
  Autonomy        autonomy_heat/power/water/net, autonomy_score, amenity_score.
                  These are wartime-specific hedonic features capturing
                  self-sufficiency (generator, Starlink, borehole, etc.).

Public API
----------
add_time_features(df)     -> df
add_floor_features(df)    -> df
add_area_features(df)     -> df
add_building_features(df) -> df
add_autonomy_features(df) -> df
add_binary_amenities(df)  -> df
run_feature_engineering(df, contract_type, label='') -> df
    Convenience wrapper that applies all feature groups in order.
"""

import numpy as np
import pandas as pd

from .config import WAR_START


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar and war-context temporal features.

    New columns
    -----------
    year, month, quarter  — standard calendar components.
    month_sin, month_cos  — cyclical sine/cosine encoding of month
                            (preserves the Dec–Jan continuity).
    is_wartime            — 1 if the listing date falls on or after
                            Russia's full-scale invasion (2022-02-24).
    days_since_war        — integer days elapsed since the invasion;
                            clipped to 0 for pre-war dates.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["is_wartime"]     = (df["date"] >= WAR_START).astype(int)
    df["year"]           = df["date"].dt.year
    df["month"]          = df["date"].dt.month
    df["quarter"]        = df["date"].dt.quarter
    df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)
    df["days_since_war"] = (df["date"] - WAR_START).dt.days.clip(lower=0)
    return df


# ---------------------------------------------------------------------------
# Floor features
# ---------------------------------------------------------------------------

def add_floor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add floor-relative features.

    New columns
    -----------
    floor_ratio     — floor / floor_count (position within the building).
    is_first_floor  — 1 if the listing is on the ground floor.
    is_last_floor   — 1 if the listing is on the top floor.
    """
    df = df.copy()
    df["floor"]       = pd.to_numeric(df.get("floor"),       errors="coerce")
    df["floor_count"] = pd.to_numeric(df.get("floor_count"), errors="coerce")
    df["floor_ratio"]    = df["floor"] / (df["floor_count"] + 1e-9)
    df["is_first_floor"] = (df["floor"] == 1).astype(int)
    df["is_last_floor"]  = (
        (df["floor"] == df["floor_count"]) & df["floor"].notna()
    ).astype(int)
    return df


# ---------------------------------------------------------------------------
# Area features
# ---------------------------------------------------------------------------

def add_area_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add area-ratio and per-room features.

    New columns
    -----------
    living_ratio    — area_living  / area_total.
    kitchen_ratio   — area_kitchen / area_total.
    area_per_room   — area_total   / room_count.
    """
    df = df.copy()
    df["area_total"]   = pd.to_numeric(df.get("area_total"),   errors="coerce")
    df["area_living"]  = pd.to_numeric(df.get("area_living"),  errors="coerce")
    df["area_kitchen"] = pd.to_numeric(df.get("area_kitchen"), errors="coerce")
    df["room_count"]   = pd.to_numeric(df.get("room_count"),   errors="coerce")
    df["living_ratio"]  = df["area_living"]  / (df["area_total"] + 1e-9)
    df["kitchen_ratio"] = df["area_kitchen"] / (df["area_total"] + 1e-9)
    df["area_per_room"] = df["area_total"]   / (df["room_count"] + 1e-9)
    return df


# ---------------------------------------------------------------------------
# Building-era features
# ---------------------------------------------------------------------------

# Ordinal era thresholds: (upper_year_exclusive, era_label)
_ERA_MAP = [
    (1960, "Stalinka (pre-1960)"),
    (1976, "Khrushchevka (1960-1975)"),
    (1992, "Brezhnevka (1976-1991)"),
    (2011, "Post-Soviet (1992-2010)"),
    (9999, "New build (post-2010)"),
]

# Ordinal integer for era (used as a numeric feature)
_ERA_ORD = {label: i for i, (_, label) in enumerate(_ERA_MAP)}


def _classify_era(year) -> str:
    """Map a construction year to its era label."""
    if pd.isna(year):
        return "Unknown"
    for threshold, label in _ERA_MAP:
        if year < threshold:
            return label
    return "New build (post-2010)"


def add_building_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add building age and era classification features.

    New columns
    -----------
    building_age    — listing year minus built_year, clipped to [0, 120].
    building_era    — string era label (Stalinka, Khrushchevka, …).
    era_ord         — ordinal integer encoding of building_era.
    is_new_building — 1 if built_year >= 2010.
    """
    df = df.copy()
    df["built_year"] = pd.to_numeric(df.get("built_year"), errors="coerce")
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    df["building_age"]    = (df["year"] - df["built_year"]).clip(0, 120)
    df["building_era"]    = df["built_year"].apply(_classify_era)
    df["era_ord"]         = df["building_era"].map(_ERA_ORD).fillna(0).astype(int)
    df["is_new_building"] = (df["built_year"] >= 2010).astype(int)
    return df


# ---------------------------------------------------------------------------
# Autonomy features (wartime-specific hedonic block)
# ---------------------------------------------------------------------------

# Maps each autonomy feature to the raw source columns that may contribute to it
_AUTONOMY_COLS = {
    "autonomy_heat":  ["has_individual_heating", "has_autonomous_heating"],
    "autonomy_power": ["has_generator", "has_solar_panels"],
    "autonomy_water": ["has_borehole", "has_water_tank"],
    "autonomy_net":   ["has_starlink", "has_fiber_internet"],
}


def _binary_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a {0, 1, NaN} Series for a potentially string-typed boolean column."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    s = df[col].copy()
    if s.dtype == object or s.dtype.name == "category":
        s = s.astype(str).str.lower().str.strip()
        s = s.map({
            "true": 1, "false": 0,
            "yes":  1, "no":    0,
            "1":    1, "0":     0,
            "1.0":  1, "0.0":   0,
            "nan": np.nan,
        })
    return pd.to_numeric(s, errors="coerce")


def add_autonomy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add wartime self-sufficiency (autonomy) scores.

    Each autonomy sub-feature is the logical OR of two raw source columns
    (e.g. ``autonomy_power`` = 1 if ``has_generator`` OR ``has_solar_panels``).

    New columns
    -----------
    autonomy_heat, autonomy_power, autonomy_water, autonomy_net
                    — binary indicators per autonomy dimension.
    autonomy_score  — sum of the four autonomy sub-features (0–4).
    amenity_score   — sum of has_furniture, has_balcony, has_parking, has_gas.
    """
    df = df.copy()
    for feat, source_cols in _AUTONOMY_COLS.items():
        vals = [_binary_col(df, c) for c in source_cols if c in df.columns]
        df[feat] = pd.concat(vals, axis=1).max(axis=1).fillna(0) if vals else 0

    amenity_cols = ["has_furniture", "has_balcony", "has_parking", "has_gas"]
    amenity_vals = [_binary_col(df, c).fillna(0) for c in amenity_cols if c in df.columns]
    df["amenity_score"] = sum(amenity_vals) if amenity_vals else 0

    sub = [df[f].fillna(0) for f in _AUTONOMY_COLS if f in df.columns]
    df["autonomy_score"] = sum(sub) if sub else 0
    return df


def add_binary_amenities(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce boolean amenity columns to {0, 1, NaN}."""
    df = df.copy()
    bool_cols = [
        "has_furniture", "has_balcony", "has_parking", "has_gas",
        "is_owner", "is_without_renovation", "is_babushka_renovation",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = _binary_col(df, col)
    return df


# ---------------------------------------------------------------------------
# Full feature-engineering pipeline
# ---------------------------------------------------------------------------

def run_feature_engineering(
    df: pd.DataFrame,
    contract_type: str,
    label: str = "",
) -> pd.DataFrame:
    """
    Apply the complete feature-engineering chain to a single segment.

    Steps applied in order:
      1. Temporal features (add_time_features)
      2. Floor features   (add_floor_features)
      3. Area features    (add_area_features)
      4. Building features (add_building_features)
      5. Autonomy + amenity features (add_autonomy_features, add_binary_amenities)

    Parameters
    ----------
    df : pd.DataFrame
        Raw segment DataFrame (already filtered to city + contract_type).
    contract_type : str
        "Оренда" (rental) or "Продаж" (sale).  Currently used for logging only;
        segment-specific transformations can be added here in future.
    label : str
        Human-readable segment label for progress logging.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with all engineered features appended.
    """
    if label:
        print(f"  [{label}] feature engineering...", end=" ")

    df = add_time_features(df)
    df = add_floor_features(df)
    df = add_area_features(df)
    df = add_building_features(df)
    df = add_autonomy_features(df)
    df = add_binary_amenities(df)

    if label:
        print(f"done  ({len(df):,} rows, {df.shape[1]} columns)")

    return df
