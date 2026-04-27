# Ukrainian Real Estate Price Forecasting under Wartime Conditions

Bachelor's thesis — Ukrainian Catholic University, 2025–2026

A machine-learning pipeline that forecasts residential real estate prices
in Kyiv and Lviv across four market segments (rental / sale) using
964,732 listing-months scraped from LUN.ua (May 2023 – January 2026).

The project integrates structural hedonic features with macro-economic
indicators (NBU rate, USD/UAH exchange rate, food-price index) and
conflict-specific covariates (air-raid alert statistics) to study how
wartime conditions affect Ukrainian housing markets.

---

## Dataset

| Property         | Value                                             |
|------------------|---------------------------------------------------|
| Source           | LUN.ua (Ukraine's largest real estate portal)     |
| Coverage         | Kyiv and Lviv, May 2023 – January 2026            |
| Total records    | 964,732 listing-months                            |
| Observation unit | Listing-month (one row per active listing per month) |
| Target (rental)  | Monthly price in UAH                              |
| Target (sale)    | Price per m² in USD                               |

---

## Market Segments

| Segment     | Observations | Target        |
|-------------|-------------|---------------|
| Kyiv — Rental  | ~310,000    | UAH / month   |
| Kyiv — Sale    | ~240,000    | USD / m²      |
| Lviv — Rental  | ~260,000    | UAH / month   |
| Lviv — Sale    | ~155,000    | USD / m²      |

---

## Model Performance (LightGBM, rolling-origin CV)

MAPE = Mean Absolute Percentage Error on the held-out test set
(October 2025 – January 2026).

| Segment     | CV MAPE |
|-------------|---------|
| Kyiv — Rental  | 21.54 % |
| Kyiv — Sale    | 17.82 % |
| Lviv — Rental  | 18.53 % |
| Lviv — Sale    | 16.41 % |

LightGBM (num\_leaves=255) is the best-performing model across all four
segments. Ridge and MLP serve as linear and neural-network baselines;
XGBoost performance is within 0.5 pp of LightGBM on most segments.

---

## Installation

```bash
# 1. Clone / download the repository
git clone <repo-url>
cd <repo-root>

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r thesis_package/requirements.txt
```

Raw data files (CSV / XLS / XLSX) must be placed in the repository root
exactly as listed in `thesis_package/src/config.py`.

---

## How to Run

All scripts are run from the **repository root** (not from inside
`thesis_package/`).

### Rolling-origin cross-validation

```bash
python thesis_package/scripts/run_cv.py
```

Runs 5 models x 4 segments x 4 folds and saves results to
`thesis_package/cv_results.csv`.

### Ablation study

```bash
python thesis_package/scripts/run_ablation.py
```

Measures the marginal contribution of each feature block
(structural / macro / security) using XGBoost per segment.
Results saved to `thesis_package/ablation_results.csv`.

### VIF diagnostics

```bash
python thesis_package/scripts/compute_vif.py
```

Computes Variance Inflation Factor for the macro and security feature block
on the training split of each segment.
Results saved to `thesis_package/vif_results.csv`.

---

## Project Structure

```
thesis_package/
├── src/
│   ├── __init__.py         Package init
│   ├── config.py           Global constants, paths, split dates
│   ├── preprocessing.py    Encoding + train/val/test split (no leakage)
│   ├── features.py         Feature engineering (time, area, building, autonomy)
│   └── models.py           Model factory: Ridge, RF, XGBoost, LightGBM, MLP
├── scripts/
│   ├── run_cv.py           4-fold rolling-origin cross-validation
│   ├── run_ablation.py     3-condition ablation study (A / B / C)
│   └── compute_vif.py      VIF diagnostics for the macro feature block
├── notebooks/
│   └── README.md           Pointer to the main analysis notebook
├── requirements.txt
└── README.md               This file

src/                        Original pipeline modules (outside thesis_package)
├── pipeline.py             End-to-end data loading and transformation steps
├── cleaning.py             Raw data cleaning
├── data.py                 Split helpers and utilities
├── geo.py                  Overpass API geospatial enrichment
├── macro.py                Macro-economic and alert data loading
├── evaluation.py           Metric computation and result reporting
├── tuning.py               Optuna hyperparameter search
└── ...

Бакалаврська_робота_v2.ipynb   Main analysis notebook (EDA, SHAP, plots)
```

---

## Temporal Split Design

To prevent data leakage and simulate a realistic forecasting scenario,
the dataset is split chronologically with gap months between splits:

```
Train : 2023-05 – 2025-04  (~73 %)
Gap 1 : 2025-05             (excluded — breaks listing autocorrelation)
Val   : 2025-06 – 2025-08  (~9 %,  hyperparameter tuning)
Gap 2 : 2025-09             (excluded)
Test  : 2025-10 – 2026-01  (~12 %, held-out; touched only once)
```

---

## Key Design Decisions

- **Log-transform target**: all models are trained on log1p(price) to handle
  right-skewed price distributions; predictions are back-transformed with expm1.
- **No leakage**: TargetEncoder for `district` and StandardScaler for MLP/Ridge
  are fit exclusively on training-set rows.
- **Wartime features**: four autonomy sub-features (heat, power, water, internet)
  capture the wartime premium for self-sufficient apartments.
- **High VIF is expected**: macro time-series are highly collinear (VIF 122–1979)
  but this does not affect gradient-boosting models.

---

## Citation

```bibtex
@thesis{charnosh2026ua_realestate,
  author  = {Charnosh, Veronika},
  title   = {Forecasting Residential Real Estate Prices under Wartime Conditions:
             Evidence from Ukraine},
  school  = {Ukrainian Catholic University},
  year    = {2026},
  type    = {Bachelor's Thesis},
}
```
