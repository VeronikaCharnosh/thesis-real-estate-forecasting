# Real Estate Price Forecasting in Wartime Ukraine

> **Bachelor's thesis** — Ukrainian Catholic University (UCU), Faculty of Applied Sciences, 2025–2026
> Author: **Veronika Charnosh**

A reproducible machine-learning pipeline that forecasts residential real estate
prices in Kyiv and Lviv across four market segments (rental and sale).
The system integrates **structural hedonic features** with **macroeconomic
indicators** (UAH/USD rate, NBU policy rate, food-price index) and
**wartime security covariates** (Ukrainian air-raid alert statistics) to
quantify how external shocks shape Ukrainian housing markets during the
full-scale Russian invasion.

---

## Research Questions

This thesis addresses three questions:

1. **RQ1**. Does a non-linear, gradient-boosted ensemble enriched with
   external macroeconomic and security covariates outperform the classical
   hedonic baseline (Ridge + structural features) for Ukrainian wartime
   real estate?

2. **RQ2**. Do the dominant price drivers shift between contract types
   (rental vs. sale) and between cities (Kyiv vs. Lviv)?

3. **RQ3 / H1**. Does adding macroeconomic and security features yield a
   measurable reduction in forecast error relative to a structural-only
   baseline, or are these channels already absorbed by location and
   property attributes?

---

## Headline Results (LightGBM, held-out test set Q4 2025 – Jan 2026)

| Segment       | n (test) |   R²  | MAPE  | Log-RMSE |
|---------------|---------:|:-----:|:-----:|:--------:|
| Kyiv — Rent   |   33,127 | 0.862 | 17.5% |  0.248   |
| Kyiv — Sale   |   52,239 | 0.741 | 14.8% |  0.217   |
| Lviv — Rent   |    8,597 | 0.680 | 16.8% |  0.233   |
| Lviv — Sale   |   15,817 | 0.503 | 15.2% |  0.219   |

**Findings:**
- **LightGBM** is the best model across all four segments, beating the
  Ridge baseline by 4–9 percentage points of MAPE.
- **Macro block helps universally** (MAPE −0.1 to −0.3 pp across all
  segments) — confirms part of H1.
- **Security block helps only in high-exposure markets** (Kyiv-Rent);
  for low-exposure Lviv segments it adds noise — partial falsification
  of H1, an interesting empirical finding.
- **Per-segment modelling beats pooled** — Kyiv and Lviv have distinct
  price dynamics that justify separate models.

---

## Dataset

| Property              | Value                                              |
|-----------------------|----------------------------------------------------|
| Source                | LUN.ua (Ukraine's largest real estate portal)      |
| Coverage              | Kyiv and Lviv, May 2023 – January 2026             |
| Raw rows (after dedup, physical & price filters) | 976,196 |
| Rows used in modelling (after gap-drop) | **916,697** |
| Observation unit      | Listing-month (one row per active listing per month) |
| Target (rental)       | Monthly price in UAH                               |
| Target (sale)         | Price per m² in USD                                |

### Segment sizes (after 5-part temporal split)

| Segment      |   Train  |  Val   |  Test  |  Total  |
|--------------|---------:|-------:|-------:|--------:|
| Kyiv — Rent  | 130,947  | 18,143 | 33,127 | 182,217 |
| Kyiv — Sale  | 380,682  | 44,404 | 52,239 | 477,325 |
| Lviv — Rent  | 103,846  |  9,883 |  8,597 | 122,326 |
| Lviv — Sale  | 104,562  | 14,450 | 15,817 | 134,829 |
| **Total**    | **720,037** | **86,880** | **109,780** | **916,697** |

---

## Methodology

### Temporal split (5-part, leakage-free)

To prevent data leakage and simulate realistic forecasting under
wartime conditions, the dataset is partitioned chronologically with
**buffer gap months** between splits:

```
Train       : 2023-05 → 2025-04   (~79 %, 23 months)
🚫 Gap 1    : 2025-05               (dropped — autocorrelation buffer)
Validation  : 2025-06 → 2025-08   (~10 %, hyperparameter tuning, early stopping)
🚫 Gap 2    : 2025-09               (dropped)
Test        : 2025-10 → 2026-01   (~12 %, held-out; evaluated once)
```

Two one-month gaps eliminate temporal autocorrelation across split
boundaries (Bergmeir & Benítez, 2012).

### Models

Five regression models, each tuned via grid search on the validation set:

| Model         | Role                                  | Best for          |
|---------------|---------------------------------------|-------------------|
| **Ridge**     | regularised linear baseline           | interpretability  |
| **Random Forest** | bagged ensemble                  | robust default    |
| **XGBoost**   | level-wise boosting                   | conservative ablation |
| **LightGBM**  | leaf-wise boosting (GOSS)             | **headline accuracy** |
| **MLP**       | 3-layer feedforward NN                | non-linear baseline |

All tree-based models train on `log1p(price)` and back-transform via
`expm1`. Ridge applies the **Duan smearing estimator** to correct for
Jensen's-inequality bias on the back-transformation.

### Cross-validation

**Rolling-origin** (walk-forward) CV with 4 expanding folds inside the
training partition, providing temporally honest estimates of variance
across the wartime observation window.

### Metrics

- **MAE / RMSE** in original price units (UAH or USD/m²)
- **R²** computed in `log1p` space (consistent with training objective)
- **MAPE** trimmed at the lowest 2nd percentile (avoids near-zero
  denominator instability for rentals)
- **Log-RMSE** — RMSE in log1p space, scale-invariant and robust to
  back-transformation bias

### Feature blocks

| Block       | Examples                                              | n  |
|-------------|-------------------------------------------------------|----|
| Structural  | `area_total`, `room_count`, `floor_ratio`, `ceiling_height` | ~20 |
| Building    | `building_age`, `is_new_building`, `wall_type_enc`    | ~6 |
| Autonomy    | `autonomy_score`, `has_furniture`, `has_gas`          | ~7 |
| Location    | `dist_to_center_km`, `district_te`, `microdistrict_enc`, `knn_price_m2` | ~7 |
| Macro       | `usd_uah`, `nbu_rate`, `food_price_idx`, `days_since_war` | 4 |
| Security    | `alert_count_month`, `alert_count_cumulative`, `dist_to_shelter_km` | ~5 |
| Interactions | `autonomy_x_alerts`, `newbuild_x_usd`                | ~5 |

---

## Repository Structure

```
thesis_package/
├── src/                              # Core library (importable as `src.*`)
│   ├── config.py                     # Constants, paths, split boundaries
│   ├── data.py                       # split_mask, drop_gaps utilities
│   ├── cleaning.py                   # Raw-data cleaning + dedup
│   ├── features.py                   # Feature engineering (time, area, autonomy)
│   ├── macro.py                      # NBU rate, FX, food index, alerts loader
│   ├── geo.py                        # OSM-based geo enrichment + KNN
│   ├── preprocessing.py              # Encoding + leakage-free train/val/test
│   ├── models.py                     # Model factory: Ridge / RF / XGB / LGBM / MLP
│   ├── evaluation.py                 # Metric computation + reports
│   ├── pipeline.py                   # End-to-end orchestration
│   ├── eda.py                        # EDA helpers
│   └── tuning.py                     # Optuna hyperparameter search
│
├── notebooks/                        # 4-notebook reproducible pipeline
│   ├── 01_eda_data_quality.ipynb     # EDA, dedup, anomaly + price-outlier filters
│   ├── 02_feature_engineering.ipynb  # Time / macro / geo / alert features
│   ├── 03_preprocessing.ipynb        # Encoding + temporal splits + Spearman
│   └── 04_models_results.ipynb       # Training / SHAP / ablation / robustness
│
├── scripts/                          # Headless re-runs of headline analyses
│   ├── run_cv.py                     # 4-fold rolling-origin CV
│   ├── run_ablation.py               # 3-condition ablation (A / B / C)
│   └── compute_vif.py                # VIF for macro block
│
├── data/                             # Input data (CSV gitignored — see below)
│   ├── unique_2023-05_2026-01_KL.csv      ← 572 MB, NOT in git
│   ├── Exchange_r (4).xls                  ← NBU exchange rate
│   ├── *CONSTRUCTION*.xlsx                 ← construction price index
│   └── *CONSUMER_GOODS*.xlsx               ← food basket index
│
├── requirements.txt                  # Pinned Python dependencies
├── .gitignore                        # Excludes CSV, pickles, caches
└── README.md
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/VeronikaCharnosh/thesis-real-estate-forecasting.git
cd thesis-real-estate-forecasting

python3 -m venv .venv
source .venv/bin/activate              # macOS / Linux
# .venv\Scripts\activate                # Windows

pip install -r requirements.txt
```

### 2. Obtain the dataset

The main dataset (`unique_2023-05_2026-01_KL.csv`, 572 MB) is **not** stored
in this repository due to size constraints.

**Place the file at:** `data/unique_2023-05_2026-01_KL.csv`

> Contact the author for access:
> [veronikacharnosh@gmail.com](mailto:veronikacharnosh@gmail.com)

The macroeconomic source files (`Exchange_r (4).xls`, two `*.xlsx` from
the State Statistics Service of Ukraine) are tracked in git and require
no additional download.

### 3. Run the notebook pipeline (in order)

```bash
jupyter lab notebooks/
```

Run notebooks **in order** with **Restart & Run All**:

1. `01_eda_data_quality.ipynb` — loads CSV, EDA, dedup, anomaly filter,
   price outlier filter [0.5%, 99.5%]. Saves `notebooks_data/clean_segments.pkl`.
2. `02_feature_engineering.ipynb` — time, macro, geo, alerts, KNN,
   H1 interactions. Saves `notebooks_data/feat_segments.pkl`.
3. `03_preprocessing.ipynb` — encoding, temporal split, Spearman validation.
   Saves `notebooks_data/seg_data.pkl`.
4. `04_models_results.ipynb` — trains 5 models × 4 segments, summary
   metrics, SHAP, ablation, slice analysis, robustness check.

⏱ **Expected runtime** (M1/M2 Mac, end-to-end): ~45–60 minutes (model
training in notebook 04 dominates; ~25 min for MLP, ~15 min for boosted
trees).

### 4. Re-run headline analyses headlessly

```bash
python scripts/run_cv.py            # rolling-origin CV
python scripts/run_ablation.py      # 3-condition ablation (A/B/C)
python scripts/compute_vif.py       # VIF diagnostics for macro block
```

---

## Key Design Decisions

- **Log-target with leakage-free smearing.** All models are trained on
  `log1p(price)` to handle right-skewed prices; Ridge applies the
  Duan smearing estimator for unbiased back-transformation.
- **Train-only encodings.** `TargetEncoder` for `district`,
  `LabelEncoder` for categoricals, `StandardScaler` for MLP/Ridge —
  all fit on training rows only.
- **Price outlier filter on TRAIN quantiles.** Outlier cutoffs
  [0.5%, 99.5%] are computed on training-set prices, then applied
  to all rows. This eliminates outlier-driven leakage while excluding
  data-entry errors (e.g., 6M UAH/month rentals) that destabilise RMSE.
- **High macro VIF is expected and harmless.** The four macro time series
  share a common wartime trend (VIF 122–1979). Tree-based models are
  invariant to this; SHAP attribution is interpreted at the
  block level rather than per-feature.

---

## Limitations

1. **Self-reported listing prices ≠ transaction prices.** The dataset
   captures asking prices on LUN.ua, not closed deals.
2. **Selection bias.** Online listings under-represent off-market and
   peer-to-peer transactions, particularly in the rental market.
3. **No spatial autocorrelation test reported.** Moran's I on residuals
   would be the natural confirmatory check; visual inspection shows no
   residual clustering beyond Halytskyi (Lviv historic centre).
4. **Currency-shock variance.** The Lviv-Sale segment has lower R²
   (0.503) due to USD-denominated prices being heavily affected by
   FX volatility that structural features cannot capture.

---



## Acknowledgements

- **Data:** LUN.ua, National Bank of Ukraine, State Statistics Service
  of Ukraine, [`ukrainian-air-raid-sirens-dataset`](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
  (Vadym Klymenko)
- **Geospatial:** OpenStreetMap contributors via the Overpass API
- **Institution:** Ukrainian Catholic University, Faculty of Applied Sciences

---

## License

Code: MIT (see `LICENSE`).
Dataset: see contact above for access terms.

## Contact

**Veronika Charnosh** — [veronikacharnosh@gmail.com](mailto:veronikacharnosh@gmail.com)
