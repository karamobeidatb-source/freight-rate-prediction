"""Loading and cleaning of the raw load data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OUTLIER_LOG_RATIO, TARGET


def load(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fix feature-level data-quality issues. Safe to apply to unlabeled data.

    - weight: ~0.6% of rows are negative. Their rates match positive-weight rows
      of the same magnitude, so the sign is a recording error -> abs().
      Missing weights (~0.6%) are imputed with the equipment median and flagged.
    - market_index: ~0.8% missing. The index is a daily market curve plus small
      per-load noise (within-day std 0.025), so the same day's median is a
      near-lossless fill. Falls back to the overall median if a day is empty.
    """
    out = df.copy()
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight"] = out["weight"].abs()
    out["weight"] = out["weight"].fillna(out.groupby("equipment")["weight"].transform("median"))
    out["weight"] = out["weight"].fillna(out["weight"].median())

    out["market_index_missing"] = out["market_index"].isna().astype(int)
    daily = out.groupby("date")["market_index"].transform("median")
    out["market_index"] = out["market_index"].fillna(daily).fillna(out["market_index"].median())
    return out


def flag_label_outliers(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask of rows whose posted_rate looks corrupted.

    A robust (Huber) log-linear fit captures the bulk of the price structure.
    Its residuals are tightly centred (std ~0.04) apart from ~1.4% of rows that
    are off by a factor of 0.15-0.45x or 2-5x, which no feature explains and
    which are spread evenly across months, lanes and equipment.
    """
    from sklearn.linear_model import HuberRegressor

    X = _baseline_design(df)
    y = np.log(df[TARGET])
    resid = y - HuberRegressor(max_iter=1000).fit(X, y).predict(X)
    return pd.Series(np.abs(resid) > OUTLIER_LOG_RATIO, index=df.index)


def _baseline_design(df: pd.DataFrame) -> pd.DataFrame:
    log_d = np.log(df["distance"])
    return pd.DataFrame(
        {
            "log_d": log_d,
            "log_d2": log_d**2,
            "weight": df["weight"] / 1e4,
            "market_index": df["market_index"],
            "flatbed": (df["equipment"] == "Flatbed").astype(float),
            "reefer": (df["equipment"] == "Reefer").astype(float),
        },
        index=df.index,
    )
