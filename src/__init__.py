"""
thesis_package/src
==================
Core modules for Ukrainian real estate price forecasting.

Modules
-------
config          Global constants, file paths, temporal split boundaries.
data            Temporal split utilities: drop_gaps, split_mask.
preprocessing   Data cleaning, encoding, and train/val/test split (no leakage).
features        Feature engineering: time, structural, building, autonomy.
models          Model factory: Ridge, RF, XGBoost, LightGBM, MLP.
"""

from . import config, data, preprocessing, features, models

__all__ = ["config", "data", "preprocessing", "features", "models"]
