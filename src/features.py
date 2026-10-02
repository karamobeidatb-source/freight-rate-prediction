"""Feature engineering shared by training, validation and December inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]
TIME_ORIGIN = pd.Timestamp("2025-01-01")

# Features fed to the gradient-boosted residual model. City names are left out
# on purpose: 12% of validation loads touch cities never seen in training, and
# the city-level price differences are explained by coordinates (latitude).
GBM_FEATURES = [
    "log_distance",
    "weight",
    "weight_missing",
    "equipment_code",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
    "quote_signal_dev",
    "day_of_week",
    "quarter_end_window",
]

# The quarter-end ramp spans the last ~30 days of a quarter (see eda.py).
QUARTER_END_WINDOW = 35


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_distance"] = np.log(out["distance"])
    out["equipment_code"] = pd.Categorical(out["equipment"], categories=EQUIPMENT).codes
    out["day_of_week"] = out["date"].dt.dayofweek
    quarter_end = out["date"] + pd.offsets.QuarterEnd(0)
    out["days_to_quarter_end"] = (quarter_end - out["date"]).dt.days
    # Capped so that mid-quarter dates all look the same: uncapped, this plus the
    # daily market_index can pinpoint individual training dates and the trees
    # start memorising day-level noise (worse out-of-time error in backtests).
    out["quarter_end_window"] = out["days_to_quarter_end"].clip(upper=QUARTER_END_WINDOW)
    out["t_years"] = (out["date"] - TIME_ORIGIN).dt.days / 365.0
    # The level of quote_signal is not stable: in training it shifts by
    # equipment and quarter (Flatbed/Reefer ~2.4 in Mar/Jun/Sep, ~1.75 in other
    # months), while in validation it is flat at ~2.05 for every equipment.
    # Only the deviation from the same day's equipment mean carries load-level
    # information that transfers, so the raw level is not used as a feature.
    day_mean = out.groupby(["date", "equipment"])["quote_signal"].transform("mean")
    out["quote_signal_dev"] = out["quote_signal"] - day_mean
    if "weight_missing" not in out:
        out["weight_missing"] = 0
    return out


def trend_design(df: pd.DataFrame) -> pd.DataFrame:
    """Design matrix for the parametric stage of the hybrid model.

    These are the smooth, global effects that a tree model cannot extrapolate
    past the end of the training window: the distance curve, equipment and
    weight premiums, market level and the calendar-time drift.
    """
    return pd.DataFrame(
        {
            "log_d": df["log_distance"],
            "log_d2": df["log_distance"] ** 2,
            "weight": df["weight"] / 1e4,
            "market_index": df["market_index"],
            "flatbed": (df["equipment"] == "Flatbed").astype(float),
            "reefer": (df["equipment"] == "Reefer").astype(float),
            "t_years": df["t_years"],
        },
        index=df.index,
    )
