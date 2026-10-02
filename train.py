"""End-to-end pipeline: clean -> backtest candidates -> fit final model -> predict.

Usage:
    python train.py

Writes:
    validation_predictions.csv          final 12,000 predictions (load_id,predicted_rate)
    data/december_chart_inputs.csv      predicted_rate filled for the fixed December lane
    artifacts/metrics.json              backtest results for every candidate
    artifacts/backtest_*.csv / *.png    tables and figures used in the report
"""
from __future__ import annotations

import json

import matplotlib.dates
import numpy as np
import pandas as pd

from src import config, data, features
from src import plotting as P
from src.models import GBMModel, HybridModel, ParametricModel, candidates
from src.validation import cold_start_test, random_kfold, time_backtest


def prepare(path) -> pd.DataFrame:
    return features.build(data.clean_features(data.load(path)))


def december_inputs(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Complete the December chart rows with the features the model needs.

    The chart file only gives lane, distance, equipment, weight and date.
    - Coordinates come from the city table in the training data.
    - market_index / quote_signal are market-wide daily signals. validation.csv
      contains ~200 real December loads per day, so the per-day median of those
      loads is used. This is input data, not labels, so there is no leakage.
    """
    chart = pd.read_csv(config.DECEMBER_PATH, parse_dates=["date"])
    coords = pd.concat(
        [
            train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
            train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        ]
    ).drop_duplicates("city").set_index("city")
    rows = chart.copy()
    rows[["pickup_lat", "pickup_lon"]] = coords.loc[rows["pickup"]].to_numpy()
    rows[["delivery_lat", "delivery_lon"]] = coords.loc[rows["delivery"]].to_numpy()
    daily = validation.groupby("date")[["market_index", "quote_signal"]].median()
    rows = rows.join(daily, on="date")
    rows["weight_missing"] = 0
    return chart, features.build(rows)


def main() -> None:
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train = prepare(config.TRAIN_PATH)
    validation = prepare(config.VALIDATION_PATH)
    outliers = data.flag_label_outliers(train)
    print(f"Train rows: {len(train):,} | label outliers flagged: {outliers.sum()} ({outliers.mean():.2%})")

    # 1. Time-based backtest of every candidate ---------------------------------
    report = {"n_train": len(train), "n_outliers": int(outliers.sum()), "candidates": {}}
    backtest_preds = {}
    for make in candidates():
        name = make().name
        folds = time_backtest(train, make, outliers)
        backtest_preds[name] = [f["pred"] for f in folds]
        report["candidates"][name] = [{k: v for k, v in f.items() if k != "pred"} for f in folds]
        summary = " | ".join(f"MAPE all {f['all']['MAPE']:.2f}% clean {f['clean']['MAPE']:.2f}%" for f in folds)
        print(f"{name:<42} {summary}")

    # Stage-1 coefficient stability per fold: with Jan-Jun only, market_index and
    # calendar time rise together and cannot be separated; from 7+ months on
    # (a full market rise and fall) the coefficients settle.
    report["trend_coefficients_by_fold"] = {}
    for train_end, _, _ in config.BACKTEST_FOLDS + [("2025-10-31", None, None)]:
        window = train[train["date"] <= train_end]
        coefs = ParametricModel().fit(window[~data.flag_label_outliers(window)]).reg.coef_
        names = features.trend_design(window).columns
        report["trend_coefficients_by_fold"][train_end] = dict(zip(names, map(float, coefs)))

    # 2. Show why a random split is the wrong yardstick here ---------------------
    report["random_kfold_hybrid"] = random_kfold(train, HybridModel, outliers)
    print(f"Hybrid random 5-fold (clean) MAPE: {report['random_kfold_hybrid']['MAPE']:.2f}%")

    # 3. Unseen-city robustness: 12% of validation loads touch a new city --------
    makers = [lambda: HybridModel(lane_correction=False), HybridModel, lambda: GBMModel(use_time=True)]
    report["cold_start"] = cold_start_test(train, makers, outliers)
    for name, r in report["cold_start"].items():
        print(f"Cold start  {name:<36} unseen-city MAPE {r['cold']:.2f}% | seen-city {r['warm']:.2f}%")

    # 4. Final model on all clean labeled data -----------------------------------
    final = HybridModel().fit(train[~outliers])
    report["final_trend_coefficients"] = dict(
        zip(features.trend_design(train).columns, map(float, final.trend.reg.coef_))
    )
    report["final_feature_importance"] = final.feature_importance().to_dict()

    val_pred = final.predict(validation)
    template = pd.read_csv(config.TEMPLATE_PATH)
    submission = template[["load_id"]].merge(
        pd.DataFrame({"load_id": validation["load_id"], "predicted_rate": np.round(val_pred, 2)}),
        on="load_id",
        how="left",
    )
    assert submission["predicted_rate"].notna().all() and len(submission) == 12_000
    submission.to_csv(config.PREDICTIONS_PATH, index=False)
    print(f"Wrote {config.PREDICTIONS_PATH.name}")

    chart, chart_rows = december_inputs(train, validation)
    chart["predicted_rate"] = np.round(final.predict(chart_rows), 2)
    report["december"] = {
        "min": float(chart["predicted_rate"].min()),
        "max": float(chart["predicted_rate"].max()),
        "dec_1": float(chart["predicted_rate"].iloc[0]),
        "dec_31": float(chart["predicted_rate"].iloc[-1]),
        "inputs": chart_rows[["date", "market_index", "quote_signal", "days_to_quarter_end"]]
        .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
    }
    chart["date"] = chart["date"].dt.strftime("%Y-%m-%d")
    chart.to_csv(config.DECEMBER_PATH, index=False)
    print(f"Wrote {config.DECEMBER_PATH.relative_to(config.ROOT)}")
    report["december"]["decomposition"] = _save_december_decomposition(final, chart_rows)

    # Quarter-end ramp the model applies to the real December loads (last 3 days),
    # to compare with the ramp measured in the training data (eda.py).
    dec_rows = validation[validation["date"].dt.month == 12]
    no_window = dec_rows.assign(quarter_end_window=features.QUARTER_END_WINDOW)
    lift = pd.Series(final.predict(dec_rows) / final.predict(no_window) - 1, index=dec_rows.index) * 100
    last3 = dec_rows["days_to_quarter_end"] <= 3
    report["december"]["model_ramp_last3days_pct"] = lift[last3].groupby(dec_rows.loc[last3, "equipment"]).mean().to_dict()

    with open(config.ARTIFACTS_DIR / "metrics.json", "w") as fh:
        json.dump(report, fh, indent=2)
    _save_backtest_artifacts(train, outliers, backtest_preds, report)


HYBRID = HybridModel().name
RIVAL = GBMModel(use_time=True).name


def _save_backtest_artifacts(train, outliers, backtest_preds, report) -> None:
    rows = []
    for name, folds in report["candidates"].items():
        for i, f in enumerate(folds, 1):
            rows.append({"model": name, "fold": i, **{f"{k}_all": v for k, v in f["all"].items()},
                         **{f"{k}_clean": v for k, v in f["clean"].items()}})
    pd.DataFrame(rows).to_csv(config.ARTIFACTS_DIR / "backtest_results.csv", index=False)

    # One panel per fold: daily mean $/mile, actual vs out-of-time predictions (clean rows).
    clean = train[~outliers]
    fig, axes = P.new(1, len(config.BACKTEST_FOLDS), size=(12.0, 3.8), sharey=True)
    for i, ax in enumerate(axes):
        idx = backtest_preds[HYBRID][i].index.intersection(clean.index)
        window = clean.loc[idx]
        actual = (window[config.TARGET] / window["distance"]).groupby(window["date"]).mean()
        ax.plot(actual.index, actual.values, color=P.INK, lw=1.4, label="Actual")
        for name, color, label in [(HYBRID, P.BLUE, "Hybrid (chosen)"), (RIVAL, P.ORANGE, "LightGBM + time")]:
            pred = (backtest_preds[name][i].loc[idx] / window["distance"]).groupby(window["date"]).mean()
            ax.plot(pred.index, pred.values, color=color, lw=1.4, label=label)
        h = report["candidates"][HYBRID][i]["clean"]["MAPE"]
        r = report["candidates"][RIVAL][i]["clean"]["MAPE"]
        train_end, test_start, test_end = config.BACKTEST_FOLDS[i]
        months = f"{pd.Timestamp(test_start):%b}-{pd.Timestamp(test_end):%b}"
        ax.set_title(f"Train to {pd.Timestamp(train_end):%b %d}, test {months}\n"
                     f"MAPE: hybrid {h:.2f}%, LightGBM + time {r:.2f}%", fontsize=10)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Daily mean $ per mile")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    P.save(fig, config.FIGURES_DIR / "backtest_daily.png")


def _save_december_decomposition(final, rows) -> dict:
    """Counterfactuals that show which inputs shape the December curve."""
    window_off = {"quarter_end_window": features.QUARTER_END_WINDOW}
    flat_market = rows.assign(market_index=rows["market_index"].mean())
    full = final.predict(rows)
    flat = final.predict(flat_market)
    drift_only = final.predict(flat_market.assign(**window_off))
    fig, ax = P.new(size=(9.0, 3.8))
    for values, color, label in [(full, P.BLUE, "Full prediction"),
                                 (flat, P.ORANGE, "market_index held at December mean"),
                                 (drift_only, P.AQUA, "... and quarter-end ramp removed")]:
        ax.plot(rows["date"], values, color=color, marker="o", ms=3, label=label)
    ax.legend(loc="upper left")
    ax.set_ylabel("Predicted rate ($)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    ax.set_title("December drivers: weekly market cycle + year-end ramp on a slow upward drift")
    P.save(fig, config.FIGURES_DIR / "december_decomposition.png")
    ramp = full / final.predict(rows.assign(**window_off)) - 1
    weekly = full / flat - 1
    return {
        "year_end_ramp_dec31_pct": float(ramp[-1] * 100),
        "drift_over_month_pct": float((drift_only[-1] / drift_only[0] - 1) * 100),
        "weekly_cycle_range_pct": [float(weekly.min() * 100), float(weekly.max() * 100)],
    }


if __name__ == "__main__":
    main()
