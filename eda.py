"""Exploratory analysis: prints the key findings and writes the report figures.

Usage:
    python eda.py

Writes:
    reports/figures/eda_*.png
    artifacts/eda_summary.json
"""
from __future__ import annotations

import json

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src import config, data, features
from src import plotting as P

STRUCTURE = ["log_d", "log_d2", "weight", "flatbed", "reefer"]


def residual(df: pd.DataFrame, extra=()) -> pd.Series:
    """log(rate) left over after the load-structure effects (+ any extra columns)."""
    X = features.trend_design(df)[STRUCTURE + list(extra)]
    y = np.log(df[config.TARGET])
    return y - LinearRegression().fit(X, y).predict(X)


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    raw_train, raw_val = data.load(config.TRAIN_PATH), data.load(config.VALIDATION_PATH)
    train = features.build(data.clean_features(raw_train))
    val = features.build(data.clean_features(raw_val))
    outliers = data.flag_label_outliers(train)
    clean = train[~outliers].copy()
    summary: dict = {}

    # ---- Data quality ------------------------------------------------------------
    unseen = sorted((set(raw_val["pickup"]) | set(raw_val["delivery"]))
                    - (set(raw_train["pickup"]) | set(raw_train["delivery"])))
    touches_unseen = raw_val["pickup"].isin(unseen) | raw_val["delivery"].isin(unseen)
    seen_lanes = set(zip(raw_train["pickup"], raw_train["delivery"]))
    summary["quality"] = {
        "train_rows": len(raw_train),
        "validation_rows": len(raw_val),
        "train_dates": [str(raw_train["date"].min().date()), str(raw_train["date"].max().date())],
        "validation_dates": [str(raw_val["date"].min().date()), str(raw_val["date"].max().date())],
        "duplicate_rows": int(raw_train.drop(columns="load_id").duplicated().sum()),
        "negative_weight": {"train": int((raw_train["weight"] < 0).sum()), "validation": int((raw_val["weight"] < 0).sum())},
        "missing_weight": {"train": int(raw_train["weight"].isna().sum()), "validation": int(raw_val["weight"].isna().sum())},
        "weight_at_47500_cap": int((raw_train["weight"].abs() == 47_500).sum()),
        "missing_market_index": {"train": int(raw_train["market_index"].isna().sum()), "validation": int(raw_val["market_index"].isna().sum())},
        "distance_at_70_floor": int((raw_train["distance"] == 70).sum()),
        "label_outliers": int(outliers.sum()),
        "label_outlier_share": float(outliers.mean()),
        "unseen_cities": unseen,
        "validation_share_touching_unseen_city": float(touches_unseen.mean()),
        "validation_share_on_seen_lane": float(pd.Series(list(zip(raw_val["pickup"], raw_val["delivery"]))).isin(seen_lanes).mean()),
    }
    neg = raw_train[raw_train["weight"] < 0]
    pos = raw_train[raw_train["weight"] > 0]
    summary["quality"]["rpm_median_negative_vs_positive_weight"] = [
        float((neg[config.TARGET] / neg["distance"]).median()),
        float((pos[config.TARGET] / pos["distance"]).median()),
    ]

    # ---- Price structure ---------------------------------------------------------
    clean["rpm"] = clean[config.TARGET] / clean["distance"]
    bands = pd.cut(clean["distance"], [0, 150, 250, 400, 600, 900, 1300, 1800, 2500, 3500])
    summary["rpm_by_distance_band"] = {str(k): float(v) for k, v in clean.groupby(bands, observed=True)["rpm"].median().items()}
    summary["rpm_by_equipment"] = clean.groupby("equipment")["rpm"].median().round(3).to_dict()
    summary["market_index_by_weekday"] = train.groupby("day_of_week")["market_index"].mean().round(3).to_dict()
    summary["market_index_monthly"] = {
        "train": train.groupby(train["date"].dt.month)["market_index"].mean().round(3).to_dict(),
        "validation": val.groupby(val["date"].dt.month)["market_index"].mean().round(3).to_dict(),
    }
    fig_rate_structure(clean)

    # ---- Time: market level, drift and quarter-end ramp ---------------------------
    clean["res_struct_mkt"] = residual(clean, ["market_index"])
    clean["res_full"] = residual(clean, ["market_index", "t_years"])
    ramp = clean[clean["days_to_quarter_end"] <= 3].groupby("equipment")["res_full"].mean() \
        - clean[clean["days_to_quarter_end"] > 30].groupby("equipment")["res_full"].mean()
    summary["quarter_end_ramp_last3days_pct"] = (np.expm1(ramp) * 100).round(2).to_dict()
    me = clean[~clean["date"].dt.month.isin([3, 6, 9])]
    summary["non_quarter_month_end_effect_pct"] = float(
        np.expm1(me[me["date"].dt.days_in_month - me["date"].dt.day <= 3]["res_full"].mean()
                 - me[me["date"].dt.days_in_month - me["date"].dt.day > 7]["res_full"].mean()) * 100
    )
    # How much of the daily price level (outside quarter-end windows) do same-day
    # market_index and a linear drift explain together?
    clean["res_struct"] = residual(clean)
    daily = clean[clean["days_to_quarter_end"] > 21].groupby("date").agg(
        level=("res_struct", "mean"), market=("market_index", "mean"), t=("t_years", "first"))
    fit = LinearRegression().fit(daily[["market", "t"]], daily["level"])
    summary["daily_level_r2_market_plus_drift"] = float(fit.score(daily[["market", "t"]], daily["level"]))
    fig_market_level(train, val, clean)
    fig_quarter_end(clean)

    # ---- quote_signal drift --------------------------------------------------------
    summary["quote_signal_by_equipment_month"] = {
        "train": train.pivot_table(index=train["date"].dt.month, columns="equipment", values="quote_signal").round(3).to_dict(),
        "validation": val.pivot_table(index=val["date"].dt.month, columns="equipment", values="quote_signal").round(3).to_dict(),
    }
    summary["corr_rate_residual_vs_abs_quote_dev"] = float(clean["res_full"].corr(clean["quote_signal_dev"].abs()))
    fig_quote_signal(train, val)

    # ---- Geography -----------------------------------------------------------------
    city = clean.groupby("pickup").agg(effect=("res_full", "mean"), lat=("pickup_lat", "first"))
    summary["city_effect_std_pct"] = float(city["effect"].std() * 100)
    summary["city_effect_corr_latitude"] = float(city["effect"].corr(city["lat"]))
    fig_geography(train, val, city, unseen)

    # ---- Outliers ------------------------------------------------------------------
    summary["outlier_bands"] = fig_outliers(train)

    with open(config.ARTIFACTS_DIR / "eda_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary["quality"], indent=2, default=str))
    print("Quarter-end ramp (last 3 days, %):", summary["quarter_end_ramp_last3days_pct"])
    print(f"City effect std {summary['city_effect_std_pct']:.2f}%, corr with latitude {summary['city_effect_corr_latitude']:.2f}")


# =============================================================================== figures

def fig_rate_structure(clean: pd.DataFrame) -> None:
    fig, ax = P.new(size=(8.6, 3.8))
    edges = np.geomspace(70, 3500, 22)
    mid = np.sqrt(edges[:-1] * edges[1:])
    for eq, color in P.EQUIPMENT_COLORS.items():
        sub = clean[clean["equipment"] == eq]
        med = sub.groupby(pd.cut(sub["distance"], edges), observed=False)["rpm"].median().to_numpy()
        ax.plot(mid, med, color=color)
        P.end_label(ax, mid[-1], med[-1], eq, color)
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 500, 1000, 2000, 3000], ["100", "200", "500", "1,000", "2,000", "3,000"])
    ax.set_xlabel("Distance (miles, log scale)")
    ax.set_ylabel("Median $ per mile")
    ax.set_xlim(right=5200)
    ax.set_title("Rate per mile falls smoothly with distance; equipment adds a near-constant premium")
    P.save(fig, config.FIGURES_DIR / "eda_rate_structure.png")


def fig_market_level(train, val, clean) -> None:
    fig, (top, bottom) = P.new(2, 1, size=(9.0, 5.6), sharex=True)
    for df, color, label in [(train, P.BLUE, "train"), (val, P.ORANGE, "validation")]:
        daily = df.groupby("date")["market_index"].mean()
        top.plot(daily.index, daily.values, color=color, lw=0.6, alpha=0.35)
        smooth = daily.rolling(7, center=True, min_periods=4).mean()
        top.plot(smooth.index, smooth.values, color=color, lw=2.0, label=label)
    top.legend(loc="upper right")
    top.set_ylabel("market_index")
    top.set_title("market_index: seasonal curve with a weekly cycle (thin = daily, bold = 7-day mean)")

    level = clean.groupby("date")["res_struct_mkt"].mean().rolling(7, center=True, min_periods=4).mean()
    bottom.plot(level.index, np.expm1(level.values) * 100, color=P.BLUE)
    for q_end in pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30"]):
        days = (q_end - pd.Timestamp(mdates.get_epoch())).days  # matplotlib date units
        bottom.axvspan(days - 20, days, color=P.GRID, alpha=0.8, lw=0)
    bottom.text(pd.Timestamp("2025-06-08").to_pydatetime(), bottom.get_ylim()[1], "last 3 weeks of quarter",
                ha="right", va="top", fontsize=8, color=P.INK_2)
    bottom.set_ylabel("Price level vs. average (%)")
    bottom.set_title("Price level after removing load structure and market_index: upward drift + quarter-end spikes")
    P.save(fig, config.FIGURES_DIR / "eda_market_level.png")


def fig_quarter_end(clean) -> None:
    fig, ax = P.new(size=(8.6, 3.8))
    days = np.arange(0, 61)
    for eq, color in P.EQUIPMENT_COLORS.items():
        sub = clean[clean["equipment"] == eq]
        base = sub.loc[sub["days_to_quarter_end"] > 30, "res_full"].mean()
        prof = sub.groupby("days_to_quarter_end")["res_full"].mean().reindex(days) - base
        prof = prof.rolling(3, center=True, min_periods=1).mean()
        ax.plot(days, np.expm1(prof.values) * 100, color=color)
        P.end_label(ax, 0, np.expm1(prof.iloc[0]) * 100, eq, color)
    ax.set_xlim(62, -9)  # reversed so time flows left to right toward quarter end
    ax.axhline(0, color=P.AXIS, lw=0.8)
    ax.set_xlabel("Days until quarter end")
    ax.set_ylabel("Rate premium vs. mid-quarter (%)")
    ax.set_title("Quarter-end ramp: rates climb over the last ~3 weeks of each quarter")
    P.save(fig, config.FIGURES_DIR / "eda_quarter_end.png")


def fig_quote_signal(train, val) -> None:
    fig, ax = P.new(size=(9.0, 3.8))
    both = pd.concat([train, val])
    weekly = both.groupby([pd.Grouper(key="date", freq="W"), "equipment"])["quote_signal"].mean().unstack()
    ax.axvspan(pd.Timestamp("2025-11-01").to_pydatetime(), weekly.index[-1].to_pydatetime(),
               color=P.GRID, alpha=0.8, lw=0)
    ax.text(pd.Timestamp("2025-11-04").to_pydatetime(), 2.62, "validation\n(Nov-Dec)",
            fontsize=8, color=P.INK_2, va="top")
    for eq, color in P.EQUIPMENT_COLORS.items():
        ax.plot(weekly.index, weekly[eq], color=color, label=eq)
    ax.legend(loc="lower left")
    ax.set_ylim(1.6, 2.65)
    ax.set_ylabel("Weekly mean quote_signal")
    ax.set_title("quote_signal level drifts: regime shifts in training, flat in validation")
    P.save(fig, config.FIGURES_DIR / "eda_quote_signal_drift.png")


def fig_geography(train, val, city, unseen) -> None:
    fig, (left, right) = P.new(1, 2, size=(10.0, 3.9))
    left.scatter(city["lat"], city["effect"] * 100, s=22, color=P.BLUE, edgecolor=P.SURFACE, lw=1.2, zorder=3)
    fit = np.polyfit(city["lat"], city["effect"] * 100, 1)
    xs = np.linspace(city["lat"].min(), city["lat"].max(), 10)
    left.plot(xs, np.polyval(fit, xs), color=P.INK_2, lw=1.2)
    for name in ["Syracuse", "Buffalo", "Dallas", "Oklahoma City"]:
        left.annotate(name, (city.loc[name, "lat"], city.loc[name, "effect"] * 100), xytext=(5, 0),
                      textcoords="offset points", fontsize=7.5, color=P.INK_2, va="center")
    left.set_xlabel("Pickup city latitude")
    left.set_ylabel("Pickup city price effect (%)")
    left.set_title(f"City effect tracks latitude (r = {city['effect'].corr(city['lat']):.2f})")

    coords = pd.concat([
        df[[f"{s}", f"{s}_lat", f"{s}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        for df in (train, val) for s in ("pickup", "delivery")
    ]).drop_duplicates("city").set_index("city")
    seen = coords.drop(index=unseen)
    right.scatter(seen["lon"], seen["lat"], s=16, color=P.BLUE, edgecolor=P.SURFACE, lw=1, label="in training", zorder=3)
    new = coords.loc[unseen]
    right.scatter(new["lon"], new["lat"], s=34, color=P.ORANGE, edgecolor=P.SURFACE, lw=1.2,
                  label="validation only", zorder=4)
    for name, row in new.iterrows():
        right.annotate(name, (row["lon"], row["lat"]), xytext=(5, 2), textcoords="offset points",
                       fontsize=7.5, color=P.INK_2)
    right.set_xlabel("Longitude")
    right.set_ylabel("Latitude")
    right.grid(axis="x", color=P.GRID, lw=0.7)
    right.legend(loc="lower left")
    right.set_title("8 validation cities never appear in training")
    P.save(fig, config.FIGURES_DIR / "eda_geography.png")


def fig_outliers(train) -> None:
    from sklearn.linear_model import HuberRegressor

    X = data._baseline_design(train)
    y = np.log(train[config.TARGET])
    ratio = np.exp(y - HuberRegressor(max_iter=1000).fit(X, y).predict(X))
    fig, ax = P.new(size=(8.6, 3.6))
    bins = np.geomspace(0.1, 8, 90)
    counts, edges = np.histogram(ratio, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    is_out = (centers < np.exp(-config.OUTLIER_LOG_RATIO)) | (centers > np.exp(config.OUTLIER_LOG_RATIO))
    ax.bar(centers, counts, width=np.diff(edges) * 0.85, color=np.where(is_out, P.ORANGE, P.BLUE), align="center")
    for cut in (np.exp(-config.OUTLIER_LOG_RATIO), np.exp(config.OUTLIER_LOG_RATIO)):
        ax.axvline(cut, color=P.INK_2, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.15, 0.25, 0.5, 1, 2, 3, 5], ["0.15x", "0.25x", "0.5x", "1x", "2x", "3x", "5x"])
    ax.set_xlabel("Actual rate / robust baseline prediction (log scale)")
    ax.set_ylabel("Loads (log scale)")
    low = ratio[ratio < np.exp(-config.OUTLIER_LOG_RATIO)]
    high = ratio[ratio > np.exp(config.OUTLIER_LOG_RATIO)]
    n_out = len(low) + len(high)
    ax.set_title(f"Corrupted labels form separate bands: {n_out} loads ({n_out / len(ratio):.1%}) at "
                 f"{low.min():.2f}-{low.max():.2f}x or {high.min():.1f}-{high.max():.1f}x")
    P.save(fig, config.FIGURES_DIR / "eda_outliers.png")
    return {"low": [float(low.min()), float(low.max())], "high": [float(high.min()), float(high.max())]}


if __name__ == "__main__":
    main()
