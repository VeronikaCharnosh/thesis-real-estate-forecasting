"""
data.py
=======
Temporal split utilities shared across notebooks and scripts.

Functions
---------
drop_gaps(df)               — Remove gap-month rows (May 2025, Sep 2025).
split_mask(df, partition)   — Boolean mask for 'train' / 'val' / 'test' partition.
"""

import pandas as pd
from .config import (
    TRAIN_END,
    GAP1_START, GAP1_END,
    VAL_START, VAL_END,
    GAP2_START, GAP2_END,
    TEST_START,
)


def drop_gaps(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Remove rows that fall inside the two gap months:
      - Gap 1: 2025-05-01 → 2025-05-31  (between train and val)
      - Gap 2: 2025-09-01 → 2025-09-30  (between val and test)

    These months are excluded to prevent autocorrelation leakage across
    the temporal split boundaries.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce")
    in_gap1 = (dates >= GAP1_START) & (dates <= GAP1_END)
    in_gap2 = (dates >= GAP2_START) & (dates <= GAP2_END)
    return df[~(in_gap1 | in_gap2)].copy()


def split_mask(df: pd.DataFrame, partition: str, date_col: str = "date") -> pd.Series:
    """
    Return a boolean Series selecting rows for the requested temporal partition.

    Parameters
    ----------
    df        : DataFrame with a date column.
    partition : one of 'train', 'val', 'test'.
    date_col  : name of the date column (default: 'date').

    Boundaries (gap months already excluded by drop_gaps):
      train — date <= 2025-04-30
      val   — 2025-06-01 → 2025-08-31
      test  — date >= 2025-10-01
    """
    dates = pd.to_datetime(df[date_col], errors="coerce")

    if partition == "train":
        return dates <= TRAIN_END

    elif partition == "val":
        return (dates >= VAL_START) & (dates <= VAL_END)

    elif partition == "test":
        return dates >= TEST_START

    else:
        raise ValueError(
            f"Unknown partition '{partition}'. Choose from: 'train', 'val', 'test'."
        )
