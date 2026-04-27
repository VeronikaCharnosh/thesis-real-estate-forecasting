"""
preprocessing.py
================
Final preprocessing step: categorical encoding, binary coercion, and
construction of per-segment train / val / test matrices.

DATA LEAKAGE PREVENTION
-----------------------
A key invariant of this module: **nothing is fit on validation or test data**.

Concretely:
  - ``house_type`` group mapping  — deterministic lookup table, no fitting.
  - Binary columns               — deterministic coercion, no fitting.
  - Label encoding (non-district) — category universe drawn from train rows only;
                                    unseen categories in val/test become NaN.
  - ``district`` target encoding  — ``TargetEncoder`` (or smoothed group-mean
                                    fallback) is fit exclusively on train rows
                                    using log1p-transformed target values.
  - ``StandardScaler``            — fitted in ``build_seg_data_from_frames``
                                    on ``X_train`` and applied to val/test.

The train boundary is read from ``config.TRAIN_END``. The split helper
``split_mask(df, 'train')`` returns a boolean Series covering all rows with
``date <= TRAIN_END`` (excluding gap months between train and val).

Public API
----------
preprocess_features(df) -> pd.DataFrame
    Apply encoding to a single segment DataFrame.

build_seg_data_from_frames(seg_datasets) -> (seg_data, SEG_CONFIG)
    Build X/y matrices for all four segments ready for model training.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import TRAIN_END, TEST_START, VAL_START, RANDOM_STATE

try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None  # graceful fallback to smoothed group-mean


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Binary columns that may arrive as strings ("true"/"false", "1"/"0", etc.)
BINARY_COLS = [
    "autonomy_heat", "autonomy_power", "autonomy_water", "autonomy_net",
    "has_furniture", "has_balcony", "has_parking", "has_gas",
    "is_without_renovation", "is_babushka_renovation", "is_owner",
    "is_new_building", "is_first_floor", "is_last_floor", "is_wartime",
]

# Canonical feature list used by the CV and evaluation scripts.
# Order is stable — do not reorder without updating downstream consumers.
FEATURE_COLS_BASE = [
    # Structural
    "room_count", "area_total", "area_living", "area_kitchen",
    "floor", "floor_count", "floor_ratio", "is_first_floor", "is_last_floor",
    "ceiling_height",
    # Building
    "building_age", "is_new_building", "built_year",
    # Autonomy / amenities (wartime-specific hedonic features)
    "autonomy_score", "autonomy_heat", "autonomy_power", "autonomy_water", "autonomy_net",
    "has_furniture", "has_balcony", "has_parking", "has_gas", "amenity_score",
    "is_without_renovation", "is_babushka_renovation",
    # Temporal
    "year", "month", "quarter", "month_sin", "month_cos",
    "is_wartime", "days_since_war",
    # Macro-economic
    "usd_uah", "nbu_rate", "food_price_idx",
    "construction_idx_residential", "construction_idx_total",
    # Security / conflict
    "alert_count_month", "alert_duration_h_month", "alert_days_month",
    "alert_count_cumulative", "alert_ratio",
    # Geographic
    "dist_to_center_km", "dist_to_shelter_km", "dist_to_subway_km", "subway_count_1km",
    # Neighbourhood price signals
    "knn_price_m2", "district_month_median",
    # Interaction terms
    "autonomy_x_alerts", "autonomy_x_wartime", "newbuild_x_usd",
    "shelter_x_alerts", "log_days_x_log_usd", "era_ord_x_center",
    # Encoded categoricals
    "district_te", "city_enc", "house_type_enc",
    # Derived ratios
    "living_ratio", "kitchen_ratio", "area_per_room",
]

# Mapping: raw house_type strings → normalised group labels
_HOUSE_MAP = {
    "хрущовка":             "хрущовка",
    "сталінка":             "до_1960",
    "дореволюційний":       "до_1960",
    "австрійський будинок": "до_1960",
    "польський люкс":       "до_1960",
    "польський будинок":    "до_1960",
    "радмін":               "до_1960",
    "гостинка":             "радянська_панель",
    "чеський проект":       "радянська_панель",
    "малосімейка":          "радянська_панель",
}

_SERIES_TYPES = ("АППС", "АППС-люкс", "БПС", "серія КТ", "серія КС", "серія КП")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _map_house_type(h) -> str:
    """Deterministic house-type normalisation (no fitting, no leakage)."""
    if pd.isna(h):
        return "невідомо"
    if h in _HOUSE_MAP:
        return _HOUSE_MAP[h]
    hs = str(h).lower()
    if "серія" in hs or h in _SERIES_TYPES:
        return "радянська_панель"
    if "спец" in hs:
        return "нова_забудова"
    return "інше"


def _to_binary(s: pd.Series) -> pd.Series:
    """Coerce a string/mixed column to {0, 1, NaN}."""
    if s.dtype == object or s.dtype.name == "category":
        s = s.astype(str).str.lower().str.strip()
        s = s.map({
            "true": 1, "false": 0, "yes": 1, "no": 0,
            "1": 1, "0": 0, "1.0": 1, "0.0": 0, "nan": np.nan,
        })
    return pd.to_numeric(s, errors="coerce")


def _split_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting training rows (date <= TRAIN_END)."""
    return pd.to_datetime(df["date"], errors="coerce") <= TRAIN_END


def _label_encode_train_only(
    df: pd.DataFrame,
    train_mask: pd.Series,
    raw_col: str,
    enc_col: str,
) -> pd.DataFrame:
    """
    Integer-encode a categorical column using the category universe observed
    in the training set only.  Unseen values in val/test become NaN.
    """
    if raw_col not in df.columns:
        return df
    train_cats = sorted(df.loc[train_mask, raw_col].dropna().unique())
    cat_to_int = {c: i for i, c in enumerate(train_cats)}
    df[enc_col] = df[raw_col].map(cat_to_int)
    return df


def _district_target_encode(
    df: pd.DataFrame,
    train_mask: pd.Series,
    target_col: str,
    smoothing: float = 10.0,
) -> pd.DataFrame:
    """
    Smooth target encoding for the 'district' categorical feature.
    Fitted on training rows only; val/test districts are transformed
    using train-derived statistics (unseen districts receive the global mean).
    """
    if "district" not in df.columns:
        return df

    y_log = np.log1p(pd.to_numeric(df[target_col], errors="coerce"))
    fit_mask = train_mask & y_log.notna() & df["district"].notna()

    if fit_mask.sum() < 50:
        # Not enough training data — skip encoding
        return df

    if TargetEncoder is not None:
        # category_encoders implementation (preferred)
        te = TargetEncoder(cols=["district"], smoothing=smoothing)
        te.fit(df.loc[fit_mask, ["district"]].astype(str), y_log.loc[fit_mask])
        df["district_te"] = te.transform(df[["district"]].astype(str))["district"]
    else:
        # Fallback: manually compute smoothed group means
        global_mean = y_log.loc[fit_mask].mean()
        grp    = y_log.loc[fit_mask].groupby(df.loc[fit_mask, "district"])
        counts = grp.count()
        means  = grp.mean()
        smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)
        df["district_te"] = df["district"].map(smoothed).fillna(global_mean)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all encoding transformations to a single segment DataFrame.

    Operations (in order, all leakage-free):
      1. Parse ``date`` to datetime.
      2. Map ``house_type`` → normalised group string (deterministic).
      3. Coerce binary columns to {0, 1, NaN}.
      4. Label-encode ``house_type_group`` and ``city`` using train categories.
      5. Target-encode ``district`` fitted on train target (log1p price).

    Parameters
    ----------
    df : pd.DataFrame
        A segment DataFrame that has already passed feature engineering
        (``features.run_feature_engineering``).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with encoding columns added.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    train_mask = _split_mask(df)

    # --- house type grouping (deterministic mapping, no leakage) ---
    if "house_type" in df.columns:
        df["house_type_group"] = df["house_type"].apply(_map_house_type)
    else:
        df["house_type_group"] = "невідомо"

    # --- binary coercion ---
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = _to_binary(df[col])

    # --- label encoding (train universe only) ---
    target_col = "price_m2_usd" if "price_m2_usd" in df.columns else "price_uah"
    for raw_col, enc_col in [("house_type_group", "house_type_enc"), ("city", "city_enc")]:
        df = _label_encode_train_only(df, train_mask, raw_col, enc_col)

    # --- district target encoding (fit on train only) ---
    df = _district_target_encode(df, train_mask, target_col)

    return df


def _coerce_numeric_frame(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Cast all listed columns to float64.
    String representations of yes/no/true/false are mapped to 1/0/NaN.
    """
    yn_map = {
        "yes": 1., "true": 1., "1": 1., "1.0": 1.,
        "no":  0., "false": 0., "0": 0., "0.0": 0.,
        "unknown": np.nan, "nan": np.nan, "": np.nan, "none": np.nan,
    }
    out = {}
    for c in cols:
        s = df[c]
        if s.dtype == object or str(s.dtype).startswith("category"):
            s2 = s.astype(str).str.strip().str.lower().map(yn_map)
            out[c] = pd.to_numeric(s2, errors="coerce")
        else:
            out[c] = pd.to_numeric(s, errors="coerce")
    return pd.DataFrame(out, index=df.index)


def build_seg_data_from_frames(seg_datasets: dict) -> tuple:
    """
    Build train / val / test matrices for all four market segments.

    Parameters
    ----------
    seg_datasets : dict
        Mapping ``{(city, ctype): df_feat}`` where each DataFrame has already
        been processed by ``preprocess_features``.

    Returns
    -------
    seg_data : dict
        ``{(city, ctype): {'X_train', 'X_val', 'X_test',
                           'y_train', 'y_val', 'y_test',
                           'features', 'target'}}``
    SEG_CONFIG : dict
        ``{(city, ctype): {'label', 'target', 'currency'}}``

    Notes
    -----
    StandardScaler is fit on X_train and applied to X_val / X_test — no leakage.
    The gap months (May 2025, Sep 2025) between train/val and val/test are
    dropped from all splits.
    """
    SEG_CONFIG = {
        ("Львів", "Оренда"): {"label": "Lviv — Rental",  "target": "price_uah",    "currency": "UAH/mo"},
        ("Київ",  "Оренда"): {"label": "Kyiv  — Rental",  "target": "price_uah",    "currency": "UAH/mo"},
        ("Львів", "Продаж"): {"label": "Lviv — Sale",    "target": "price_m2_usd", "currency": "USD/m²"},
        ("Київ",  "Продаж"): {"label": "Kyiv  — Sale",    "target": "price_m2_usd", "currency": "USD/m²"},
    }

    seg_data = {}

    for key, df_feat in seg_datasets.items():
        cfg = SEG_CONFIG.get(key)
        if cfg is None:
            continue

        target = cfg["target"]
        df = df_feat.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Drop gap months
        gap_mask = (
            ((df["date"] >= pd.Timestamp("2025-05-01")) & (df["date"] <= pd.Timestamp("2025-05-31")))
            | ((df["date"] >= pd.Timestamp("2025-09-01")) & (df["date"] <= pd.Timestamp("2025-09-30")))
        )
        df = df[~gap_mask]

        # Split masks
        tr_mask  = df["date"] <= TRAIN_END
        va_mask  = (df["date"] >= VAL_START) & (df["date"] <= pd.Timestamp("2025-08-31"))
        te_mask  = df["date"] >= TEST_START

        # Keep only columns present in FEATURE_COLS_BASE
        feats = [c for c in FEATURE_COLS_BASE if c in df.columns]

        # Filter valid target rows per split
        tr = df[tr_mask].dropna(subset=[target])
        va = df[va_mask].dropna(subset=[target])
        te = df[te_mask].dropna(subset=[target])

        X_train_raw = _coerce_numeric_frame(tr, feats)
        X_val_raw   = _coerce_numeric_frame(va, feats)
        X_test_raw  = _coerce_numeric_frame(te, feats)

        # Impute with train medians (no leakage from val/test)
        train_med = X_train_raw.median()
        X_train = X_train_raw.fillna(train_med).fillna(0.0)
        X_val   = X_val_raw.fillna(train_med).fillna(0.0)
        X_test  = X_test_raw.fillna(train_med).fillna(0.0)

        # Replace infinities
        for X in (X_train, X_val, X_test):
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(0.0, inplace=True)

        seg_data[key] = {
            "X_train":  X_train,
            "X_val":    X_val,
            "X_test":   X_test,
            "y_train":  tr[target].astype(float),
            "y_val":    va[target].astype(float),
            "y_test":   te[target].astype(float),
            "features": feats,
            "target":   target,
        }

        print(
            f"  {cfg['label']:<20}  "
            f"train={len(tr):>6,}  val={len(va):>5,}  test={len(te):>5,}  "
            f"features={len(feats)}"
        )

    return seg_data, SEG_CONFIG
