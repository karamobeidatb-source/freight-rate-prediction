"""Project paths and constants shared by the pipeline scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = ROOT / "artifacts"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"

TARGET = "posted_rate"
SEED = 42

# A label is treated as corrupted when it is more than this factor away from a
# robust baseline fit (|log ratio| > log(1.5)). Clean rows sit within ~±10%;
# corrupted rows sit at 0.15-0.45x or 2-5x, so the cut lands in an empty gap.
OUTLIER_LOG_RATIO = 0.405

# Expanding-window backtest folds. Each test window is two months, matching the
# Nov-Dec horizon of the final predictions.
BACKTEST_FOLDS = [
    ("2025-06-30", "2025-07-01", "2025-08-31"),
    ("2025-07-31", "2025-08-01", "2025-09-30"),
    ("2025-08-31", "2025-09-01", "2025-10-31"),
]
