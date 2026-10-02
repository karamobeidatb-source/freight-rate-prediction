"""Time-based backtesting and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import BACKTEST_FOLDS, SEED, TARGET
from .data import flag_label_outliers


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_pred - y_true
    ape = np.abs(err) / y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(np.mean(ape) * 100),
        "MdAPE": float(np.median(ape) * 100),
    }


def time_backtest(df: pd.DataFrame, make_model, outlier_mask: pd.Series) -> list[dict]:
    """Expanding-window backtest.

    Training rows are filtered for label outliers using a mask computed on the
    training window only; test rows are always scored in full ("all") and
    separately with outliers removed ("clean") for a noise-free read.
    """
    results = []
    for train_end, test_start, test_end in BACKTEST_FOLDS:
        train = df[df["date"] <= train_end]
        test = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
        train = train[~flag_label_outliers(train)]
        model = make_model().fit(train)
        pred = model.predict(test)
        clean = ~outlier_mask.loc[test.index]
        results.append(
            {
                "fold": f"train<= {train_end} | test {test_start}..{test_end}",
                "n_train": len(train),
                "n_test": len(test),
                "all": metrics(test[TARGET], pred),
                "clean": metrics(test.loc[clean, TARGET], pred[clean.to_numpy()]),
                "pred": pd.Series(pred, index=test.index),
            }
        )
    return results


def cold_start_test(df: pd.DataFrame, makers, outlier_mask: pd.Series,
                    n_draws: int = 4, n_hidden: int = 8, seed: int = 7) -> dict:
    """Simulate the 8 unseen validation cities on the last backtest fold.

    Each draw hides `n_hidden` random cities from the training window, then
    scores test loads touching them ("cold") separately from the rest ("warm").
    """
    train_end, test_start, test_end = BACKTEST_FOLDS[-1]
    cities = sorted(set(df["pickup"]) | set(df["delivery"]))
    rng = np.random.default_rng(seed)
    results: dict = {}
    for _ in range(n_draws):
        hidden = set(rng.choice(cities, n_hidden, replace=False))

        def touches(x):
            return x["pickup"].isin(hidden) | x["delivery"].isin(hidden)

        train = df[(df["date"] <= train_end) & ~touches(df)]
        train = train[~flag_label_outliers(train)]
        test = df[(df["date"] >= test_start) & (df["date"] <= test_end) & ~outlier_mask]
        cold = touches(test).to_numpy()
        for make in makers:
            model = make().fit(train)
            pred = model.predict(test)
            entry = results.setdefault(model.name, {"cold": [], "warm": []})
            entry["cold"].append(metrics(test.loc[cold, TARGET], pred[cold])["MAPE"])
            entry["warm"].append(metrics(test.loc[~cold, TARGET], pred[~cold])["MAPE"])
    return {name: {k: float(np.mean(v)) for k, v in r.items()} for name, r in results.items()}


def random_kfold(df: pd.DataFrame, make_model, outlier_mask: pd.Series, k: int = 5) -> dict:
    """Shuffled K-fold, run only to show how optimistic it is vs. the time split."""
    clean_df = df[~outlier_mask]
    preds = pd.Series(np.nan, index=clean_df.index)
    for tr, te in KFold(k, shuffle=True, random_state=SEED).split(clean_df):
        model = make_model().fit(clean_df.iloc[tr])
        preds.iloc[te] = model.predict(clean_df.iloc[te])
    return metrics(clean_df[TARGET], preds)
