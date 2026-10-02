# Freight Rate Prediction

Predicts `posted_rate` for truckload freight. Trained on 48,000 loads from Jan–Oct 2025, predicts 12,000 loads from Nov–Dec 2025.

**Model:** a two-stage hybrid on log(rate):
1. A robust (Huber) log-linear trend model for the effects that must extrapolate into Nov–Dec: the distance curve, equipment and weight premiums, `market_index` and a calendar drift.
2. LightGBM on the trend model's residuals: the quarter-end ramp, geography and the `quote_signal` deviation.
3. A shrunk per-lane correction added on top.

**Validation:** expanding-window, out-of-time backtests with two-month test windows. The latest fold (train Jan–Aug, test Sep–Oct) scores **1.35% MAPE** on clean labels, against 3.45% for a lane-median baseline and 2.42% for LightGBM with a time feature.

The full write-up is in [reports/freight_rate_report.docx](reports/freight_rate_report.docx) ([PDF](reports/freight_rate_report.pdf)).

**Video walkthrough (3 min):** https://www.loom.com/share/e467702e48f44082a053b3d6da5b0e64

## Run

Requires Python 3.10+. Tested with Python 3.12 and pandas 2.3, numpy 2.5, scikit-learn 1.9, LightGBM 4.7, matplotlib 3.11.

1. Put the provided files in `data/`:
   ```
   data/train_test.csv
   data/validation.csv
   data/validation_predictions_template.csv
   data/december_chart_inputs.csv
   ```
2. Install and run:
   ```bash
   python -m venv .venv
   source .venv/bin/activate            # Windows: .venv\Scripts\activate
   pip install -r requirements.txt

   python eda.py                        # exploration: figures + artifacts/eda_summary.json
   python train.py                      # backtests, final fit, predictions (~1 min)
   python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
   python make_report.py                # rebuilds the report from the artifacts
   ```

`train.py` writes `validation_predictions.csv` (`load_id,predicted_rate`). It also fills `predicted_rate` in `data/december_chart_inputs.csv`, which `score.py` turns into `scorer_results/candidate_december.png`.

## Layout

| Path | What it does |
|---|---|
| `src/data.py` | Loading, feature cleaning, corrupted-label detection |
| `src/features.py` | Feature engineering shared by training and inference |
| `src/models.py` | Baselines, LightGBM variants and the final `HybridModel` |
| `src/validation.py` | Expanding-window backtest, cold-start test, metrics |
| `src/plotting.py` | Shared chart style |
| `eda.py` | Exploration figures and summary statistics |
| `train.py` | End-to-end pipeline |
| `make_report.py` | Builds the DOCX report from `artifacts/` and `reports/figures/` |
| `score.py` | Provided scorer (unchanged) |

## Key decisions

- **Time-based split.** The task is a forecast, so validation uses out-of-time folds that mirror the Nov–Dec horizon. A shuffled 5-fold split reports 1.14% MAPE, which overstates accuracy.
- **Corrupted labels.** 1.4% of rates are off by 0.17–0.46× or 2.2–5.4×. They are flagged inside each training window and removed from training, but still scored in evaluation.
- **`quote_signal` drift.** Its level shifts by equipment and quarter in training and is flat in validation. Only its deviation from the same day's equipment mean is used.
- **Unseen cities.** 8 validation cities never appear in training (12% of loads). The model uses coordinates, not city IDs, and the lane correction is zero for unseen lanes. A simulated cold-start test checks this.
- **December inputs.** The chart file lacks coordinates, `market_index` and `quote_signal`. Coordinates come from the training city table. The two signals use the median of that day's validation loads.
