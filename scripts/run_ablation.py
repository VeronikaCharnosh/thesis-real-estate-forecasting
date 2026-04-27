"""
run_ablation.py
===============
3-condition ablation study for the Ukrainian real estate price-forecasting pipeline.

Purpose
-------
Quantify the marginal contribution of each feature block to predictive accuracy
by systematically removing feature groups and measuring the resulting change in
MAPE on the held-out test set.

Ablation conditions
-------------------
  Condition A — Structural only
      room_count, area_total, area_living, area_kitchen,
      floor, floor_count, floor_ratio, is_first_floor, is_last_floor,
      ceiling_height, building_age, is_new_building, built_year,
      autonomy_score, autonomy_heat, autonomy_power, autonomy_water, autonomy_net,
      has_furniture, has_balcony, has_parking, has_gas, amenity_score,
      is_without_renovation, is_babushka_renovation,
      year, month, quarter, month_sin, month_cos,
      dist_to_center_km, dist_to_shelter_km, dist_to_subway_km, subway_count_1km,
      knn_price_m2, district_month_median,
      district_te, city_enc, house_type_enc,
      living_ratio, kitchen_ratio, area_per_room

  Condition B — Structural + Macro
      Condition A + usd_uah, food_price_idx, nbu_rate,
                    days_since_war, is_wartime

  Condition C — Structural + Macro + Security
      Condition B + alert_count_month, alert_count_cumulative

Model: XGBoost (same tuned hyperparameters as main pipeline).
Metric: MAPE on test split (2025-10-01 – 2026-01-25).

Known results (from full-pipeline runs, reported in thesis Table 5.3)
----------------------------------------------------------------------
See KNOWN_ABLATION_RESULTS below.

Usage
-----
  cd thesis_package
  python scripts/run_ablation.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE   = os.path.dirname(os.path.abspath(__file__))
_PKG    = os.path.dirname(_HERE)
_REPO   = os.path.dirname(_PKG)
_SRCDIR = os.path.join(_PKG, "src")

for _p in (_SRCDIR, _PKG, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

sys.path.insert(0, _REPO)
from src.pipeline import step1_load, step2_clean, step3_features, step4_macro_geo
from src.preprocessing import preprocess_features

from config import RANDOM_STATE, TEST_START, TRAIN_END, VAL_START
from preprocessing import FEATURE_COLS_BASE

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("ERROR: xgboost not installed. Run:  pip install xgboost")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Previously measured ablation results (thesis Table 5.3)
# Keys: (city, contract_type)  Values: {condition: MAPE%}
# ---------------------------------------------------------------------------

KNOWN_ABLATION_RESULTS = {
    ("Київ",  "Оренда"): {"A": 18.67, "B": 18.56, "C": 18.49},
    ("Київ",  "Продаж"): {"A": 16.87, "B": 16.55, "C": 16.70},
    ("Львів", "Оренда"): {"A": 18.29, "B": 18.29, "C": 19.33},
    ("Львів", "Продаж"): {"A": 17.50, "B": 17.27, "C": 17.78},
}


# ---------------------------------------------------------------------------
# Feature sets for each ablation condition
# ---------------------------------------------------------------------------

# Condition A: structural + hedonic + geographic + encoded categoricals
FEATURES_A = [
    # Structural
    "room_count", "area_total", "area_living", "area_kitchen",
    "floor", "floor_count", "floor_ratio", "is_first_floor", "is_last_floor",
    "ceiling_height",
    # Building
    "building_age", "is_new_building", "built_year",
    # Autonomy / amenities
    "autonomy_score", "autonomy_heat", "autonomy_power", "autonomy_water", "autonomy_net",
    "has_furniture", "has_balcony", "has_parking", "has_gas", "amenity_score",
    "is_without_renovation", "is_babushka_renovation",
    # Calendar
    "year", "month", "quarter", "month_sin", "month_cos",
    # Geographic
    "dist_to_center_km", "dist_to_shelter_km", "dist_to_subway_km", "subway_count_1km",
    # Neighbourhood price signals
    "knn_price_m2", "district_month_median",
    # Encoded categoricals
    "district_te", "city_enc", "house_type_enc",
    # Derived ratios
    "living_ratio", "kitchen_ratio", "area_per_room",
]

# Condition B: A + macro-economic block
_MACRO_FEATURES = ["usd_uah", "food_price_idx", "nbu_rate", "days_since_war", "is_wartime"]
FEATURES_B = FEATURES_A + _MACRO_FEATURES

# Condition C: B + security / conflict block
_SECURITY_FEATURES = ["alert_count_month", "alert_count_cumulative"]
FEATURES_C = FEATURES_B + _SECURITY_FEATURES

CONDITION_FEATURES = {"A": FEATURES_A, "B": FEATURES_B, "C": FEATURES_C}

CONDITION_LABELS = {
    "A": "Structural only",
    "B": "Structural + Macro",
    "C": "Structural + Macro + Security",
}

SEG_TARGETS = {
    ("Львів", "Оренда"): "price_uah",
    ("Київ",  "Оренда"): "price_uah",
    ("Львів", "Продаж"): "price_m2_usd",
    ("Київ",  "Продаж"): "price_m2_usd",
}

SEG_LABELS = {
    ("Львів", "Оренда"): "Lviv-Rent",
    ("Київ",  "Оренда"): "Kyiv-Rent",
    ("Львів", "Продаж"): "Lviv-Sale",
    ("Київ",  "Продаж"): "Kyiv-Sale",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill(X: pd.DataFrame, med: pd.Series) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _train_xgb_one_condition(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
) -> float:
    """
    Train XGBoost with early stopping on val; evaluate MAPE on test.
    Returns MAPE% on the test set.
    """
    y_tr_log = np.log1p(y_tr)
    y_va_log = np.log1p(y_va)

    med  = X_tr.median()
    Xtr  = _fill(X_tr, med)
    Xva  = _fill(X_va, med)
    Xte  = _fill(X_te, med)

    probe = XGBRegressor(
        n_estimators=2000, learning_rate=0.02, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.5, n_jobs=-1,
        random_state=RANDOM_STATE, verbosity=0, tree_method="hist",
        early_stopping_rounds=80,
    )
    probe.fit(Xtr, y_tr_log, eval_set=[(Xva, y_va_log)], verbose=False)
    n_best = int((probe.best_iteration or 0) + 1)

    final = XGBRegressor(
        n_estimators=max(100, n_best), learning_rate=0.02, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.5, n_jobs=-1,
        random_state=RANDOM_STATE, verbosity=0, tree_method="hist",
    )
    X_full = pd.concat([Xtr, Xva])
    y_full = np.concatenate([y_tr_log, y_va_log])
    final.fit(X_full, y_full, verbose=False)

    y_pred = np.expm1(final.predict(Xte))
    return _mape(y_te, y_pred)


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablation(processed: dict) -> pd.DataFrame:
    """
    Run all three ablation conditions across all four segments.

    Parameters
    ----------
    processed : dict
        {(city, ctype): df_preprocessed}

    Returns
    -------
    pd.DataFrame
        Long-form table with columns [Segment, Condition, FeatureCount, MAPE%].
    """
    rows = []

    for (city, ctype), df in processed.items():
        target = SEG_TARGETS[(city, ctype)]
        label  = SEG_LABELS[(city, ctype)]
        df     = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df[target] = pd.to_numeric(df[target], errors="coerce")

        # Gap months are excluded from all splits
        gap = (
            ((df["date"] >= pd.Timestamp("2025-05-01")) & (df["date"] <= pd.Timestamp("2025-05-31")))
            | ((df["date"] >= pd.Timestamp("2025-09-01")) & (df["date"] <= pd.Timestamp("2025-09-30")))
        )
        df = df[~gap].dropna(subset=["date", target])

        tr = df[df["date"] <= TRAIN_END]
        va = df[(df["date"] >= VAL_START) & (df["date"] <= pd.Timestamp("2025-08-31"))]
        te = df[df["date"] >= TEST_START]

        if len(tr) < 200 or len(va) < 50 or len(te) < 50:
            print(f"  {label}: SKIP (too few rows)")
            continue

        y_tr = tr[target].values.astype(float)
        y_va = va[target].values.astype(float)
        y_te = te[target].values.astype(float)

        print(f"\n  {label}  (train={len(tr):,}  val={len(va):,}  test={len(te):,})")

        for cond, feat_list in CONDITION_FEATURES.items():
            # Use only features that actually exist in this segment's DataFrame
            avail = [f for f in feat_list if f in df.columns]

            X_tr = tr[avail].copy()
            X_va = va[avail].copy()
            X_te = te[avail].copy()

            mape_val = _train_xgb_one_condition(X_tr, y_tr, X_va, y_va, X_te, y_te)

            known = KNOWN_ABLATION_RESULTS.get((city, ctype), {}).get(cond)
            delta = f"  (known={known:.2f}%,  diff={mape_val - known:+.2f}%)" if known else ""
            print(f"    Condition {cond} [{CONDITION_LABELS[cond]}]: "
                  f"MAPE={mape_val:.2f}%  n_features={len(avail)}{delta}")

            rows.append({
                "Segment":      label,
                "City":         city,
                "ContractType": ctype,
                "Condition":    cond,
                "ConditionLabel": CONDITION_LABELS[cond],
                "N_Features":   len(avail),
                "MAPE%":        round(mape_val, 2),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  ABLATION STUDY  |  3 conditions x 4 segments  |  XGBoost")
    print("=" * 70)

    print("\n[1/4] Loading & cleaning...")
    _, segments = step1_load()
    clean = step2_clean(segments)

    print("\n[2/4] Feature engineering...")
    feat = step3_features(clean)

    print("\n[3/4] Macro + alerts (no geo — fast mode)...")
    enriched = step4_macro_geo(feat, load_geo=False)

    print("\n[4/4] Preprocessing...")
    processed = {k: preprocess_features(v) for k, v in enriched.items()}

    print("\n" + "=" * 70)
    print("  RUNNING ABLATION")
    print("=" * 70)
    results = run_ablation(processed)

    # --- Pivot table ---
    print("\n\n" + "=" * 70)
    print("  RESULTS TABLE (MAPE%)")
    print("=" * 70)
    if not results.empty:
        pivot = results.pivot(index="Segment", columns="Condition", values="MAPE%")
        pivot["B-A"] = (pivot["B"] - pivot["A"]).round(2)
        pivot["C-B"] = (pivot["C"] - pivot["B"]).round(2)
        print(pivot.to_string())

    # --- Compare with known results ---
    print("\n\n" + "=" * 70)
    print("  COMPARISON WITH KNOWN RESULTS (thesis Table 5.3)")
    print("=" * 70)
    print(f"  {'Segment':<14} {'Cond':<5} {'Measured':>10} {'Known':>10} {'Delta':>10}")
    print("  " + "-" * 52)
    for _, row in results.iterrows():
        city, ctype, cond = row["City"], row["ContractType"], row["Condition"]
        known = KNOWN_ABLATION_RESULTS.get((city, ctype), {}).get(cond)
        if known is not None:
            delta = row["MAPE%"] - known
            print(f"  {row['Segment']:<14} {cond:<5} {row['MAPE%']:>10.2f} "
                  f"{known:>10.2f} {delta:>+10.2f}")

    # --- Save ---
    out_csv = os.path.join(_PKG, "ablation_results.csv")
    results.to_csv(out_csv, index=False)
    print(f"\n\nResults saved -> {out_csv}")

    return results


if __name__ == "__main__":
    results = main()
