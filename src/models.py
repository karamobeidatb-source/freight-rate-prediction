"""Candidate models. All predict posted_rate in dollars and train on log(rate)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold

from .config import SEED, TARGET
from .features import GBM_FEATURES, trend_design

# Shared by every LightGBM candidate so the comparison is like for like.
# n_estimators/num_leaves were picked on the mean MAPE of backtest folds 2-3;
# every setting tried (400-1500 trees, 15/31 leaves) landed within ~0.1pp.
N_ESTIMATORS = 400
LGBM_PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
)


class LaneMedianBaseline:
    """Median rate-per-mile by lane+equipment, falling back to equipment+distance band."""

    name = "Lane median $/mile"

    def fit(self, df: pd.DataFrame):
        rpm = df[TARGET] / df["distance"]
        self.lane = rpm.groupby([df["pickup"], df["delivery"], df["equipment"]]).median()
        self.band = rpm.groupby([df["equipment"], self._band(df)], observed=True).median()
        return self

    @staticmethod
    def _band(df):
        return pd.cut(df["distance"], [0, 150, 250, 400, 600, 900, 1300, 1800, 2500, 5000])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([df["pickup"], df["delivery"], df["equipment"]])
        lane = self.lane.reindex(keys).to_numpy()
        band = self.band.reindex(pd.MultiIndex.from_arrays([df["equipment"], self._band(df)])).to_numpy()
        return np.where(np.isnan(lane), band, lane) * df["distance"].to_numpy()


class ParametricModel:
    """Robust log-linear regression on the smooth global effects only."""

    name = "Huber log-linear"

    def fit(self, df: pd.DataFrame):
        self.reg = HuberRegressor(max_iter=1000).fit(trend_design(df), np.log(df[TARGET]))
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(trend_design(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(df))


class LaneCorrection:
    """Shrunk mean residual per lane (pickup -> delivery), added after the main model.

    Captures lane-specific pricing that distance and coordinates do not explain
    (~0.6% std in backtests). Each lane's mean is pulled toward 0 by
    `shrinkage` pseudo-observations (5 ~ noise/signal variance ratio), so thin
    lanes contribute little. Unseen lanes get exactly 0, so loads touching new
    cities are priced by the coordinate-based model alone.

    It is applied post hoc rather than fed to the trees as a feature: as a
    feature the trees leaned on it (63% of gain) instead of the coordinates,
    which made unseen-city loads ~0.3pp worse in a simulated cold-start test.
    """

    def __init__(self, shrinkage: float = 5.0):
        self.shrinkage = shrinkage

    @staticmethod
    def _key(df: pd.DataFrame) -> pd.Series:
        return df["pickup"] + "|" + df["delivery"]

    def fit(self, df: pd.DataFrame, resid: pd.Series):
        agg = pd.Series(resid, index=df.index).groupby(self._key(df)).agg(["sum", "count"])
        self.encoding = agg["sum"] / (agg["count"] + self.shrinkage)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._key(df).map(self.encoding).fillna(0.0).to_numpy()


class GBMModel:
    """LightGBM on log(rate). Optionally given calendar time as a feature."""

    def __init__(self, use_time: bool = False):
        self.use_time = use_time
        self.features = GBM_FEATURES + (["t_years"] if use_time else [])
        self.name = "LightGBM + time feature" if use_time else "LightGBM"

    def fit(self, df: pd.DataFrame):
        self.model = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, **LGBM_PARAMS)
        self.model.fit(df[self.features], np.log(df[TARGET]), categorical_feature=["equipment_code"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model.predict(df[self.features]))


class HybridModel:
    """Parametric trend stage + LightGBM on its residuals (the chosen model).

    Stage 1 (Huber) learns the smooth effects that must extrapolate into
    Nov-Dec: distance curve, equipment/weight premiums, market level and the
    upward calendar drift. Trees cannot extrapolate a trend - they flatline at
    the last value seen - so this part has to be parametric.

    Stage 2 (LightGBM) learns what is left: the quarter-end ramp and its
    equipment interaction, the non-linear quote_signal deviation effect,
    regional (latitude) pricing and the remaining distance/weight curvature.
    Its calendar features (days to quarter end, weekday) are cyclic, so they
    stay inside the range seen in training when applied to Nov-Dec.

    Stage 3 (optional) adds a shrunk per-lane correction, measured on
    out-of-fold stage-2 residuals so it only captures what stages 1-2 miss.
    """

    def __init__(self, lane_correction: bool = True):
        self.lane_correction = lane_correction
        self.name = "Hybrid + lane correction (final)" if lane_correction else "Hybrid: Huber trend + LightGBM"

    @staticmethod
    def _gbm(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
        model = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, **LGBM_PARAMS)
        return model.fit(X, y, categorical_feature=["equipment_code"])

    def fit(self, df: pd.DataFrame):
        self.trend = ParametricModel().fit(df)
        resid = np.log(df[TARGET]) - self.trend.predict_log(df)
        X = df[GBM_FEATURES]
        self.gbm = self._gbm(X, resid)
        if self.lane_correction:
            # Out-of-fold stage-2 predictions: in-sample residuals would already
            # be partly fitted by the trees and understate lane effects.
            oof = np.zeros(len(df))
            for fit_idx, pred_idx in KFold(5, shuffle=True, random_state=SEED).split(df):
                fold_model = self._gbm(X.iloc[fit_idx], resid.iloc[fit_idx])
                oof[pred_idx] = fold_model.predict(X.iloc[pred_idx])
            self.lanes = LaneCorrection().fit(df, resid - oof)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        log_rate = self.trend.predict_log(df) + self.gbm.predict(df[GBM_FEATURES])
        if self.lane_correction:
            log_rate = log_rate + self.lanes.transform(df)
        return np.exp(log_rate)

    def feature_importance(self) -> pd.Series:
        gain = self.gbm.booster_.feature_importance(importance_type="gain")
        return pd.Series(gain / gain.sum(), index=GBM_FEATURES).sort_values(ascending=False)


def candidates():
    """Zero-argument factories, so each backtest fold gets a fresh model."""
    return [
        LaneMedianBaseline,
        ParametricModel,
        lambda: GBMModel(use_time=False),
        lambda: GBMModel(use_time=True),
        lambda: HybridModel(lane_correction=False),
        HybridModel,
    ]
