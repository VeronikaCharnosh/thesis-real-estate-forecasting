"""
compute_vif.py
==============
Compute Variance Inflation Factor (VIF) for the macro and security feature
block to diagnose multicollinearity.

Background
----------
VIF measures how much the variance of a regression coefficient is inflated
due to linear dependence with other predictors.  As a rule of thumb:

    VIF < 5   — low multicollinearity (acceptable)
    VIF < 10  — moderate (monitor)
    VIF >= 10 — high (potential problem)

The macro block in this dataset contains highly correlated time-series:
USD/UAH exchange rate, food-price index, NBU policy rate, and days-since-war
all trend together over the 2022-2026 period.  Expected VIF range: 122–1979
(confirmed in the main notebook).  These high values are expected and do NOT
invalidate the gradient-boosting models (which are not sensitive to
multicollinearity), but they rule out interpreting raw regression coefficients.

Feature block examined
----------------------
  usd_uah               — UAH/USD exchange rate (monthly average)
  food_price_idx        — Food price index (State Statistics Service)
  nbu_rate              — NBU key policy rate (%)
  days_since_war        — Days elapsed since 2022-02-24
  alert_count_month     — Air-raid alerts in city-month
  alert_count_cumulative— Cumulative alerts since war start

VIF is computed on the **training-set rows** of each segment to avoid
data leakage from test into the collinearity diagnostics.

Usage
-----
  cd thesis_package
  python scripts/compute_vif.py
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

from config import TRAIN_END

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:
    print("ERROR: statsmodels not installed.  Run:  pip install statsmodels")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Feature block to examine
# ---------------------------------------------------------------------------

MACRO_FEATURES = [
    "usd_uah",
    "food_price_idx",
    "nbu_rate",
    "days_since_war",
    "alert_count_month",
    "alert_count_cumulative",
]

# Expected VIF ranges from the main notebook (thesis Table 4.2)
EXPECTED_VIF = {
    "usd_uah":               (1000, 2000),   # ~1979
    "food_price_idx":        (1000, 2000),   # ~1892
    "nbu_rate":              (100,  500),    # ~122
    "days_since_war":        (1000, 2000),   # ~1742
    "alert_count_month":     (1,    50),     # ~3.1
    "alert_count_cumulative":(1000, 2000),   # ~1641
}

SEG_LABELS = {
    ("Львів", "Оренда"): "Lviv-Rent",
    ("Київ",  "Оренда"): "Kyiv-Rent",
    ("Львів", "Продаж"): "Lviv-Sale",
    ("Київ",  "Продаж"): "Kyiv-Sale",
}

SEG_TARGETS = {
    ("Львів", "Оренда"): "price_uah",
    ("Київ",  "Оренда"): "price_uah",
    ("Львів", "Продаж"): "price_m2_usd",
    ("Київ",  "Продаж"): "price_m2_usd",
}


# ---------------------------------------------------------------------------
# VIF computation
# ---------------------------------------------------------------------------

def compute_vif_for_block(df_train: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Compute VIF for each feature in ``features`` using the training subset.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training-set rows (date <= TRAIN_END).
    features : list[str]
        Feature names to include in the VIF computation.

    Returns
    -------
    pd.DataFrame
        Columns: feature, VIF.
    """
    avail = [f for f in features if f in df_train.columns]
    if len(avail) < 2:
        return pd.DataFrame({"feature": avail, "VIF": [np.nan] * len(avail)})

    # Drop rows with any NaN or inf in the feature block
    X = (
        df_train[avail]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .values.astype(float)
    )

    if len(X) < 10:
        return pd.DataFrame({"feature": avail, "VIF": [np.nan] * len(avail)})

    vif_vals = [
        variance_inflation_factor(X, i)
        for i in range(X.shape[1])
    ]

    return pd.DataFrame({"feature": avail, "VIF": vif_vals})


def _flag(vif_val: float, feature: str) -> str:
    """Return a flag string based on VIF severity."""
    if np.isnan(vif_val):
        return "N/A"
    if vif_val >= 10:
        return "HIGH (expected for time-series macro)"
    if vif_val >= 5:
        return "MODERATE"
    return "OK"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  VIF DIAGNOSTICS  |  Macro + Security feature block")
    print("=" * 70)
    print(f"\n  Features examined: {MACRO_FEATURES}")
    print(f"\n  NOTE: High VIF is expected for macro time-series variables.")
    print(f"  Gradient-boosting models (XGBoost, LightGBM) are not sensitive")
    print(f"  to multicollinearity — this analysis is for transparency only.")

    print("\n[1/4] Loading & cleaning...")
    _, segments = step1_load()
    clean = step2_clean(segments)

    print("\n[2/4] Feature engineering...")
    feat = step3_features(clean)

    print("\n[3/4] Macro + alerts (no geo — fast mode)...")
    enriched = step4_macro_geo(feat, load_geo=False)

    print("\n[4/4] Preprocessing...")
    processed = {k: preprocess_features(v) for k, v in enriched.items()}

    all_rows = []
    header_printed = False

    for (city, ctype), df in processed.items():
        label = SEG_LABELS.get((city, ctype), f"{city}-{ctype}")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # VIF computed on training rows only
        df_train = df[df["date"] <= TRAIN_END]

        vif_df = compute_vif_for_block(df_train, MACRO_FEATURES)
        vif_df.insert(0, "Segment", label)
        vif_df["Flag"] = vif_df.apply(lambda r: _flag(r["VIF"], r["feature"]), axis=1)

        # Print per-segment table
        print(f"\n{'─' * 60}")
        print(f"  Segment: {label}  (train rows = {len(df_train):,})")
        print(f"{'─' * 60}")
        header = f"  {'Feature':<30} {'VIF':>10}  {'Flag'}"
        if not header_printed:
            print(header)
            header_printed = True
        for _, row in vif_df.iterrows():
            vif_str = f"{row['VIF']:>10.1f}" if not np.isnan(row["VIF"]) else f"{'N/A':>10}"
            print(f"  {row['feature']:<30} {vif_str}  {row['Flag']}")

        all_rows.append(vif_df)

    # --- Summary across all segments ---
    df_all = pd.concat(all_rows, ignore_index=True)

    print("\n\n" + "=" * 70)
    print("  SUMMARY: Mean VIF across segments")
    print("=" * 70)
    mean_vif = df_all.groupby("feature")["VIF"].agg(["mean", "min", "max"]).round(1)
    mean_vif.columns = ["Mean_VIF", "Min_VIF", "Max_VIF"]
    mean_vif = mean_vif.reindex(MACRO_FEATURES)

    # Add expected range column
    mean_vif["Expected_range"] = mean_vif.index.map(
        lambda f: f"{EXPECTED_VIF[f][0]}–{EXPECTED_VIF[f][1]}" if f in EXPECTED_VIF else "—"
    )
    print(mean_vif.to_string())

    print("\n\n  Interpretation:")
    print("  - alert_count_month has low VIF (~3) — near-orthogonal to macro trend.")
    print("  - All other macro features have VIF > 100, consistent with")
    print("    strong temporal collinearity (all trend upward post-Feb 2022).")
    print("  - Tree-based models are unaffected; coefficient-based models")
    print("    (Ridge) should not have their weights interpreted causally.")

    # --- Save ---
    out_csv = os.path.join(_PKG, "vif_results.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"\n\nVIF table saved -> {out_csv}")

    return df_all


if __name__ == "__main__":
    vif_df = main()
