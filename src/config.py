"""
config.py
=========
Global constants, file paths, and temporal-split boundaries for the
Ukrainian real estate price-forecasting pipeline.

This is the single source of truth for all configuration values.
Edit here — nowhere else — to change any pipeline setting.

Temporal split (70 / 15 / 15 by listing-months)
------------------------------------------------
  Train : 2023-05-01  →  2025-04-30   (~73 %)
  GAP 1 : 2025-05-01  →  2025-05-31   (excluded — breaks autocorrelation)
  Val   : 2025-06-01  →  2025-08-31   (~9 %,  hyperparameter tuning)
  GAP 2 : 2025-09-01  →  2025-09-30   (excluded)
  Test  : 2025-10-01  →  2026-01-25   (~12 %, held-out; touch only once)
"""

import pathlib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Repository root is two levels above this file:  thesis_package/src/config.py
# → thesis_package/  → <repo root>
# Adjust ROOT if you relocate the package.
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent  # repo root

PATH_LISTINGS = str(ROOT / "unique_2023-05_2026-01_KL.csv")
PATH_EXCHANGE = str(ROOT / "Exchange_r (4).xls")
PATH_CONSTR   = str(
    ROOT
    / "dataset_2026-03-23T18_00_25.794972804Z_DEFAULT_INTEGRATION_SSSU_DF_PRICE_CHANGE_CONSTRUCTION_LATEST.xlsx"
)
PATH_CONSUMER = str(
    ROOT
    / "dataset_2026-03-23T17_18_31.080960851Z_DEFAULT_INTEGRATION_SSSU_DF_PRICE_CHANGE_CONSUMER_GOODS_SERVICE_LATEST.xlsx"
)

# ---------------------------------------------------------------------------
# Temporal split boundaries
# ---------------------------------------------------------------------------

TRAIN_END  = pd.Timestamp("2025-04-30")
GAP1_START = pd.Timestamp("2025-05-01")
GAP1_END   = pd.Timestamp("2025-05-31")
VAL_START  = pd.Timestamp("2025-06-01")
VAL_END    = pd.Timestamp("2025-08-31")
GAP2_START = pd.Timestamp("2025-09-01")
GAP2_END   = pd.Timestamp("2025-09-30")
TEST_START = pd.Timestamp("2025-10-01")

# Legacy alias kept for backward compatibility
TRAIN_CUTOFF = TRAIN_END

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

WAR_START    = pd.Timestamp("2022-02-24")   # Russia's full-scale invasion
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Market segments
# ---------------------------------------------------------------------------

SEGMENT_KEYS = [
    ("Львів", "Оренда"),   # Lviv  — Rental
    ("Київ",  "Оренда"),   # Kyiv  — Rental
    ("Львів", "Продаж"),   # Lviv  — Sale
    ("Київ",  "Продаж"),   # Kyiv  — Sale
]

# Colour palette for plotting (Tableau-style, colourblind-safe)
SEG_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]

# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------

# City centre coordinates (lat, lon) — used for Haversine distance feature
CITY_CENTERS = {
    "Київ":  (50.4501, 30.5234),
    "Львів": (49.8397, 24.0297),
}

# Bounding boxes for Overpass API queries (south, west, north, east)
CITY_BBOX = {
    "Київ":  (50.30, 30.35, 50.58, 30.65),
    "Львів": (49.77, 23.95, 49.90, 24.15),
}

# ---------------------------------------------------------------------------
# NBU key policy rate — chronological change log
# (date of decision, rate in percent per annum)
# ---------------------------------------------------------------------------

NBU_RATE_CHANGES = [
    ("2023-01-27", 25.0), ("2023-03-17", 25.0), ("2023-04-28", 25.0),
    ("2023-06-16", 25.0), ("2023-07-28", 22.0), ("2023-09-15", 20.0),
    ("2023-10-27", 16.0), ("2023-12-15", 15.0),
    ("2024-01-26", 15.0), ("2024-03-15", 14.5), ("2024-04-26", 13.5),
    ("2024-06-14", 13.0), ("2024-07-26", 13.0), ("2024-09-20", 13.0),
    ("2024-11-01", 13.0), ("2024-12-13", 13.5),
    ("2025-01-24", 14.5), ("2025-03-07", 15.5), ("2025-04-18", 15.5),
    ("2025-06-06", 15.5), ("2025-07-25", 15.5), ("2025-09-12", 15.5),
    ("2025-10-24", 15.5), ("2025-12-12", 15.5),
    ("2026-01-30", 15.0), ("2026-03-20", 15.0),
]

# ---------------------------------------------------------------------------
# External data URLs
# ---------------------------------------------------------------------------

ALERTS_URL = (
    "https://raw.githubusercontent.com/Vadimkin/"
    "ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv"
)
