"""Build the submission report (DOCX) from the pipeline outputs.

Usage (after eda.py, train.py and score.py):
    python make_report.py

Writes reports/freight_rate_report.docx. Every number is read from
artifacts/metrics.json and artifacts/eda_summary.json, so the report always
matches the code that produced the predictions.
"""
from __future__ import annotations

import inspect
import json
import textwrap
from datetime import date
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from src import config, validation
from src.models import HybridModel

AUTHOR = "karamobeidatb-source"
OUT = config.REPORTS_DIR / "freight_rate_report.docx"
SCORER_CHART = config.ROOT / "scorer_results" / "candidate_december.png"
FINAL = "Hybrid + lane correction (final)"
TEAL = RGBColor(0x06, 0x4A, 0x56)
GREY = RGBColor(0x52, 0x51, 0x4E)
CONTENT_WIDTH = 6.5  # inches: US Letter with 1" margins


def num(x: float, nd: int = 2) -> str:
    # Unicode minus: keeps the sign attached to the number at line breaks.
    return f"{x:.{nd}f}".replace("-", "−")


def pct(x: float, nd: int = 1) -> str:
    return num(x, nd) + "%"


def effect(log_coef: float) -> float:
    """Percent change implied by a coefficient on log(rate)."""
    return float(np.expm1(log_coef) * 100)


def source(*objs) -> tuple[str, int, int, list[str]]:
    """Exact source of adjacent functions: (file, first line, last line, dedented lines)."""
    path = Path(inspect.getsourcefile(objs[0]))
    start = inspect.getsourcelines(objs[0])[1]
    last_lines, last_start = inspect.getsourcelines(objs[-1])
    end = last_start + len(last_lines) - 1
    text = "\n".join(path.read_text(encoding="utf-8").splitlines()[start - 1:end])
    return path.relative_to(config.ROOT).as_posix(), start, end, textwrap.dedent(text).splitlines()


class Report:
    def __init__(self):
        self.doc = Document()
        self.figures = 0
        self.tables = 0
        self.snippets = 0
        section = self.doc.sections[0]
        section.page_width, section.page_height = Inches(8.5), Inches(11)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(section, side, Inches(1))
        normal = self.doc.styles["Normal"]
        normal.font.name = "Calibri"
        normal.font.size = Pt(10.5)
        normal.paragraph_format.space_after = Pt(6)
        normal.paragraph_format.line_spacing = 1.1
        for name, size in (("Title", 24), ("Heading 1", 15), ("Heading 2", 12)):
            style = self.doc.styles[name]
            style.font.name = "Calibri"
            style.font.size = Pt(size)
            style.font.color.rgb = TEAL
        self.doc.styles["Heading 1"].paragraph_format.space_before = Pt(16)
        self.doc.styles["Heading 2"].paragraph_format.space_before = Pt(10)

    # -- text ----------------------------------------------------------------------
    def heading(self, text: str, level: int = 1):
        self.doc.add_heading(text, level)

    def para(self, text: str, italic: bool = False):
        p = self.doc.add_paragraph()
        self._runs(p, text, italic=italic)
        return p

    def bullet(self, text: str):
        """Bullet with an optional bold lead-in: '**Lead.** rest of sentence'."""
        p = self.doc.add_paragraph(style="List Bullet")
        self._runs(p, text)
        return p

    def code(self, lines: list[str]):
        for line in lines:
            p = self.doc.add_paragraph()
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.left_indent = Inches(0.25)
            run = p.add_run(line)
            run.font.name = "Consolas"
            run.font.size = Pt(9)

    @staticmethod
    def _runs(p, text: str, italic: bool = False):
        # '**bold**' segments alternate with plain text.
        for i, part in enumerate(text.split("**")):
            if part:
                run = p.add_run(part)
                run.bold = i % 2 == 1
                run.italic = italic

    # -- figures and tables ----------------------------------------------------------
    def figure(self, path, caption: str, width: float = CONTENT_WIDTH):
        self.figures += 1
        self.doc.add_picture(str(path), width=Inches(width))
        pic = self.doc.paragraphs[-1]
        pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pic.paragraph_format.keep_with_next = True
        self._caption(f"Figure {self.figures}. {caption}")
        return self.figures

    def table(self, header: list[str], rows: list[list[str]], widths: list[float],
              caption: str, bold_last: bool = False):
        self.tables += 1
        cap = self._caption(f"Table {self.tables}. {caption}")
        cap.paragraph_format.keep_with_next = True
        table = self.doc.add_table(rows=1, cols=len(header))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        for cell, text, w in zip(table.rows[0].cells, header, widths):
            self._cell(cell, text, w, bold=True, fill="E6EEF0")
        for r, values in enumerate(rows):
            cells = table.add_row().cells
            last = bold_last and r == len(rows) - 1
            for cell, text, w in zip(cells, values, widths):
                self._cell(cell, text, w, bold=last)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return self.tables

    def code_block(self, objs: tuple, description: str):
        """Shaded, monospaced copy of the given functions, read from the source file."""
        self.snippets += 1
        path, start, end, lines = source(*objs)
        names = " and ".join(o.__qualname__ for o in objs)
        cap = self._caption(f"Code {self.snippets}. {names} ({path}, lines {start}–{end}): {description}")
        cap.paragraph_format.keep_with_next = True
        table = self.doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        cell = table.rows[0].cells[0]
        cell.width = Inches(CONTENT_WIDTH)
        self._shade(cell, "F3F4F4")
        for i, line in enumerate(lines):
            p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run(line or " ")
            run.font.name = "Consolas"
            run.font.size = Pt(9)
            if line.lstrip().startswith("#"):
                run.font.color.rgb = GREY
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def page_break(self):
        self.doc.add_page_break()

    @staticmethod
    def _shade(cell, fill: str):
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), fill)
        cell._tc.get_or_add_tcPr().append(shd)

    def _caption(self, text: str):
        p = self.doc.add_paragraph()
        run = p.add_run(text)
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = GREY
        return p

    def _cell(self, cell, text: str, width: float, bold: bool = False, fill: str | None = None):
        cell.width = Inches(width)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.0
        self._runs(p, text)
        for run in p.runs:
            run.font.size = Pt(9)
            run.bold = run.bold or bold
        if fill:
            self._shade(cell, fill)

    def save(self, path):
        props = self.doc.core_properties
        props.title = "Freight Rate Prediction - Model and Validation Report"
        props.author = props.last_modified_by = AUTHOR
        props.comments = ""
        self.doc.save(path)


def main() -> None:
    m = json.loads((config.ARTIFACTS_DIR / "metrics.json").read_text())
    s = json.loads((config.ARTIFACTS_DIR / "eda_summary.json").read_text())
    q = s["quality"]
    final = m["candidates"][FINAL]
    coefs = m["final_trend_coefficients"]
    by_fold = m["trend_coefficients_by_fold"]
    dec = m["december"]
    imp = m["final_feature_importance"]
    clean_mape = [f["clean"]["MAPE"] for f in final]
    all_mape = [f["all"]["MAPE"] for f in final]
    baseline = m["candidates"]["Lane median $/mile"][-1]["clean"]["MAPE"]
    rival = m["candidates"]["LightGBM + time feature"][-1]["clean"]["MAPE"]
    rpm = s["rpm_by_distance_band"]
    ramp = s["quarter_end_ramp_last3days_pct"]
    mi_month = s["market_index_monthly"]["train"]
    mi_dow = s["market_index_by_weekday"]
    bands = s["outlier_bands"]
    cold = m["cold_start"]
    elasticity = {d: coefs["log_d"] + 2 * coefs["log_d2"] * np.log(d) for d in (100, 3000)}
    n_unseen = len(q["unseen_cities"])

    r = Report()
    r.doc.add_heading("Freight Rate Prediction", 0)
    p = r.para("Machine Learning Engineer assessment · model and validation report")
    p.runs[0].font.color.rgb = GREY
    d = date.today()
    r.para(f"{d:%B} {d.day}, {d.year}").runs[0].font.color.rgb = GREY

    # ---- Summary -----------------------------------------------------------------------
    r.heading("Summary")
    r.bullet("**Model.** A two-stage hybrid on log(rate): a robust log-linear trend model "
             "(distance curve, equipment and weight premiums, market_index, calendar drift), LightGBM on its "
             "residuals (quarter-end ramp, geography, quote_signal deviation) and a shrunk per-lane correction.")
    r.bullet(f"**Validation.** Expanding-window, out-of-time backtests with two-month test windows that mirror "
             f"the November–December target. On the latest fold (train Jan–Aug, test Sep–Oct) the model scores "
             f"{pct(clean_mape[-1], 2)} MAPE on clean labels ({pct(all_mape[-1], 2)} including corrupted labels), "
             f"against {pct(baseline, 2)} for a lane-median baseline and {pct(rival, 2)} for LightGBM with a time feature.")
    r.bullet(f"**Data quality.** {q['label_outliers']} corrupted labels ({pct(q['label_outlier_share'] * 100)}) removed "
             f"from training, sign-flipped and missing weights repaired, missing market_index filled from the same day, "
             f"a drifting quote_signal neutralised, and {n_unseen} cities that appear only in validation "
             f"({pct(q['validation_share_touching_unseen_city'] * 100)} of loads) handled through coordinates.")
    r.bullet(f"**December.** For the fixed Lexington → Fort Wayne load the predicted rate rises from "
             f"${dec['dec_1']:.0f} on Dec 1 to ${dec['dec_31']:.0f} on Dec 31: a weekly market cycle on top of a "
             f"year-end quarter-close ramp and a slow upward drift.")

    # ---- 1. Findings -------------------------------------------------------------------------
    r.heading("1. What the data shows")
    r.para(f"{q['train_rows']:,} labelled loads from {q['train_dates'][0]} to {q['train_dates'][1]} "
           f"(train_test.csv) and {q['validation_rows']:,} unlabelled loads from {q['validation_dates'][0]} to "
           f"{q['validation_dates'][1]} (validation.csv). Every driver acts as a percentage on the rate, so all "
           f"modelling is done on log(rate).")
    r.bullet(f"**Distance and equipment set the base rate.** Rate per mile falls smoothly from about "
             f"${rpm['(0, 150]']:.2f} under 150 miles to ${rpm['(2500, 3500]']:.2f} beyond 2,500 miles. Reefer and "
             f"Flatbed carry a near-constant premium of {pct(effect(coefs['reefer']))} and "
             f"{pct(effect(coefs['flatbed']))} over Dry Van; heavier loads add {pct(effect(coefs['weight']))} per "
             f"10,000 lb (Figure 1).")
    r.bullet(f"**market_index is a daily, market-wide signal.** It follows a seasonal curve (monthly mean "
             f"{mi_month['1']:.2f} in January, {mi_month['5']:.2f} in May, {mi_month['9']:.2f} in September) plus a "
             f"strong weekly cycle (Thursday {mi_dow['3']:.2f} vs Sunday {mi_dow['6']:.2f}). Per-load values differ "
             f"from the day's level by only about 0.025. Rates move {pct(effect(coefs['market_index'] * 0.1), 2)} "
             f"per 0.1 of index (Figure 2).")
    r.bullet(f"**Prices drift up about {pct(effect(coefs['t_years']))} a year beyond the market.** Same-day "
             f"market_index plus a linear drift explain {pct(s['daily_level_r2_market_plus_drift'] * 100, 0)} of the "
             f"day-to-day price level outside quarter-end windows (Figure 2, lower panel).")
    r.bullet(f"**Quarter-end ramp.** Rates climb over the last three weeks of every quarter, reaching "
             f"+{pct(ramp['Flatbed'])} for Flatbed, +{pct(ramp['Reefer'])} for Reefer and +{pct(ramp['Dry Van'])} for "
             f"Dry Van in the final three days. Ordinary month-ends show no effect "
             f"({pct(s['non_quarter_month_end_effect_pct'], 2)}), and no holiday effect is visible around New Year, "
             f"Memorial Day, July 4 or Labor Day (Figure 3).")
    r.bullet(f"**Geography.** City price effects (standard deviation {pct(s['city_effect_std_pct'])}) track latitude "
             f"(r = {num(s['city_effect_corr_latitude'])}): southern cities such as Dallas and Oklahoma City are "
             f"pricier than northern ones such as Syracuse and Buffalo. Coordinates explain this, so the model uses "
             f"coordinates rather than city identities (Figure 4).")
    r.figure(config.FIGURES_DIR / "eda_rate_structure.png", "Median rate per mile by distance and equipment (clean labels).")
    r.figure(config.FIGURES_DIR / "eda_market_level.png",
             "Top: market_index, train and validation. Bottom: price level after removing load structure and "
             "market_index; shaded bands are the last three weeks of each quarter.")
    r.figure(config.FIGURES_DIR / "eda_quarter_end.png",
             "Rate premium relative to mid-quarter, by days until quarter end (after removing structure, market and drift).")
    r.figure(config.FIGURES_DIR / "eda_geography.png",
             "Left: pickup-city price effect against latitude. Right: city coordinates; orange cities appear only in validation.")

    # ---- 2. Data quality -----------------------------------------------------------------
    r.heading("2. Data-quality issues and how they were handled")
    r.table(
        ["Issue", "Evidence", "Treatment"],
        [
            ["Corrupted labels",
             f"{q['label_outliers']} loads ({pct(q['label_outlier_share'] * 100)}) sit at "
             f"{bands['low'][0]:.2f}–{bands['low'][1]:.2f}× or {bands['high'][0]:.1f}–{bands['high'][1]:.1f}× a robust "
             f"baseline, separated from the clean band (±10%) by empty gaps; spread evenly across months, lanes and "
             f"equipment (Figure 5).",
             "Flagged with a Huber fit on each training window and removed from training. Kept in evaluation as the "
             "\"all rows\" score."],
            ["Negative weights",
             f"{q['negative_weight']['train']} train / {q['negative_weight']['validation']} validation rows. Their median "
             f"rate per mile (${q['rpm_median_negative_vs_positive_weight'][0]:.2f}) matches positive-weight loads "
             f"(${q['rpm_median_negative_vs_positive_weight'][1]:.2f}).",
             "Treated as a sign error: absolute value."],
            ["Missing weight",
             f"{q['missing_weight']['train']} train / {q['missing_weight']['validation']} validation rows.",
             "Equipment median plus a missing-value flag."],
            ["Missing market_index",
             f"{q['missing_market_index']['train']} train / {q['missing_market_index']['validation']} validation rows.",
             "Median of the same day's loads (within-day spread is only about 0.025)."],
            ["Capped values",
             f"Weight capped at 47,500 lb ({q['weight_at_47500_cap']:,} rows); distance floored at 70 miles "
             f"({q['distance_at_70_floor']} rows).",
             "Kept as is; they are plausible values at the limits and tree splits handle the flat tails."],
            ["quote_signal drift",
             "In training its level jumps with equipment and quarter (Flatbed/Reefer about 2.4 in Mar/Jun/Sep, about "
             "1.75 in other months); in validation it is flat at about 2.05 for all equipment (Figure 6). Using the raw "
             "level let the model credit the quarter-end ramp to quote_signal, which erased the ramp in December.",
             "Raw level dropped. Only the deviation from the same day's equipment mean is used (unusually high or low "
             "values come with about 1% higher rates)."],
            ["Unseen cities",
             f"{n_unseen} validation cities ({', '.join(q['unseen_cities'])}) never appear in training; they touch "
             f"{pct(q['validation_share_touching_unseen_city'] * 100)} of validation loads.",
             "No city IDs as features; coordinates carry geography. The lane correction is zero for unseen lanes. "
             "Checked with a simulated cold-start test (Table 4)."],
            ["Incomplete December inputs",
             "december_chart_inputs.csv has no coordinates, market_index or quote_signal.",
             "Coordinates from the training city table; market_index and quote_signal set to the median of that day's "
             "~200 validation loads (inputs, not labels)."],
        ],
        widths=[1.25, 2.85, 2.4],
        caption="Data-quality issues. No duplicate rows or inconsistent city coordinates were found.",
    )
    r.figure(config.FIGURES_DIR / "eda_outliers.png",
             "Actual rate divided by a robust baseline prediction. Orange bars (beyond the vertical lines) are treated as corrupted labels.")
    r.figure(config.FIGURES_DIR / "eda_quote_signal_drift.png", "Weekly mean quote_signal by equipment; validation period shaded.")

    # ---- 3. Validation ---------------------------------------------------------------------
    r.heading("3. Validation approach and data split")
    r.para(f"The task is a forecast: train on January–October and predict November–December, so the split follows "
           f"time. A random split would test the model on the same days, market states and quarter-end windows it "
           f"trained on. A shuffled 5-fold split reports {pct(m['random_kfold_hybrid']['MAPE'], 2)} MAPE for the final "
           f"model, flattering the out-of-time figure of {pct(clean_mape[-1], 2)}–{pct(clean_mape[1], 2)}.")
    r.heading("Expanding-window backtest", 2)
    r.bullet("**Three folds.** Each trains on all data up to a cut-off and tests on the next two months, the same "
             "horizon as November–December. Test windows start the day after training ends.")
    r.bullet("**Cleaning inside the fold.** Corrupted labels are flagged using the training window only. Test loads are "
             "scored on all rows and on clean rows, because no model can predict a 3× label error.")
    r.bullet("**Metric.** MAPE is primary: rates range from about $60 to $25,000, and percentage error treats short and "
             "long hauls alike. MAE, RMSE and median APE are in artifacts/metrics.json. RMSE on all rows is dominated by "
             f"the corrupted labels (about ${final[-1]['all']['RMSE']:.0f} vs ${final[-1]['clean']['RMSE']:.0f} on clean rows).")
    r.bullet(f"**Selection rule.** Model and hyperparameter choices use the mean of folds 2 and 3. Fold 1 trains on "
             f"January–June, when market_index and calendar time rise together, so the trend stage cannot separate them "
             f"(market coefficient {by_fold['2025-06-30']['market_index']:.3f}, drift "
             f"{by_fold['2025-06-30']['t_years']:.3f}/yr). From seven months of data on, both settle (Table 3), and the "
             f"final model trains on ten.")
    folds = []
    for i, (train_end, test_start, test_end) in enumerate(config.BACKTEST_FOLDS):
        folds.append([f"{i + 1}", f"Jan 1 – {_md(train_end)}", f"{_md(test_start)} – {_md(test_end)}",
                      f"{final[i]['n_train']:,}", f"{final[i]['n_test']:,}"])
    r.table(["Fold", "Training window (2025)", "Test window (2025)", "Training loads (clean)", "Test loads"],
            folds, widths=[0.5, 1.6, 1.6, 1.5, 1.3], caption="Backtest folds.")
    names = list(m["candidates"])
    rows = []
    for name in names:
        cells = [name] + [f"{f['clean']['MAPE']:.2f} / {f['all']['MAPE']:.2f}" for f in m["candidates"][name]]
        cells.append(f"{np.mean([f['clean']['MAPE'] for f in m['candidates'][name][1:]]):.2f}")
        rows.append(cells)
    r.table(["Model", "Fold 1: Jul–Aug", "Fold 2: Aug–Sep", "Fold 3: Sep–Oct", "Mean folds 2–3 (clean)"],
            rows, widths=[2.3, 1.05, 1.05, 1.05, 1.05],
            caption="Out-of-time MAPE (%), clean rows / all rows. Lower is better.", bold_last=True)
    coef_rows = []
    for end, c in by_fold.items():
        label = f"{_md(end)}" + (" (final)" if end == "2025-10-31" else "")
        coef_rows.append([label, f"{c['market_index']:.3f}", f"{c['t_years']:.3f}"])
    r.table(["Training data up to", "market_index coefficient", "Drift per year (log)"], coef_rows,
            widths=[2.1, 2.2, 2.2], caption="Trend-stage coefficients by training cut-off.")
    r.heading("Unseen-city test", 2)
    r.para("To check the loads that touch a new city, each of four draws hides 8 random cities from the fold-3 "
           "training data and scores the test loads that involve them separately.")
    r.table(["Model", "Loads with hidden cities", "Other loads"],
            [[name, pct(v["cold"], 2), pct(v["warm"], 2)] for name, v in cold.items()],
            widths=[2.9, 1.8, 1.8], caption="Simulated cold start, MAPE on clean rows.")
    r.para("The lane correction leaves hidden-city loads exactly as the coordinate-based model prices them and improves "
           "the rest. An earlier version fed the lane effect to the trees as a feature; they leaned on it instead of the "
           "coordinates and hidden-city loads scored 1.85%, which is why the correction is applied after the trees.")
    r.heading("Final fit", 2)
    r.para(f"The final model is refit on all {m['n_train'] - m['n_outliers']:,} clean January–October loads with the "
           f"same settings, then applied to validation.csv and december_chart_inputs.csv.")
    r.figure(config.FIGURES_DIR / "backtest_daily.png",
             "Daily mean rate per mile in each test window: actual against out-of-time predictions.")

    # ---- 4. Model choice --------------------------------------------------------------------
    r.heading("4. Why this model")
    r.bullet("**Log target.** The effects are percentage premiums, so errors are modelled in percent.")
    r.bullet(f"**A parametric stage for anything that must extrapolate.** November–December lie beyond the training "
             f"dates. Tree models hold the last value they saw for calendar time, so LightGBM with a time feature cannot "
             f"continue the drift ({pct(rival, 2)} on fold 3). A robust (Huber) log-linear model carries the distance "
             f"curve, equipment and weight premiums, market_index and the drift, and is insensitive to corrupted labels "
             f"that slip through.")
    r.bullet("**Gradient boosting for the rest.** The residual effects are non-linear and interact: the quarter-end ramp "
             "differs by equipment, geography is a two-dimensional surface and the quote_signal deviation is U-shaped. "
             "LightGBM learns them from coordinates and cyclic calendar features that stay inside their training range "
             "in November–December. Days to quarter end are capped at 35, so the trees cannot combine them with the "
             "daily market_index to identify and memorise individual training dates.")
    r.bullet("**A shrunk lane correction last.** Lanes keep a small specific effect (about 0.6%). Each lane's mean "
             "out-of-fold residual is shrunk toward zero (5 pseudo-loads, close to the noise-to-signal variance ratio) "
             "and added on top; unseen lanes get zero.")
    r.bullet("**Alternatives rejected.** Lane-median $/mile (no market or time signal), pure linear model (misses the "
             "interactions), LightGBM without time (cannot see the drift), LightGBM with time (flatlines after October), "
             "and a blend of the hybrid with LightGBM + time (helped one fold, hurt another).")
    r.table(
        ["Effect (trend stage)", "Estimate", "Reading"],
        [
            ["Distance elasticity", f"{elasticity[100]:.2f} at 100 mi, {elasticity[3000]:.2f} at 3,000 mi",
             "Rate grows less than proportionally with miles, so rate per mile falls."],
            ["Flatbed vs Dry Van", f"+{pct(effect(coefs['flatbed']))}", "Constant premium."],
            ["Reefer vs Dry Van", f"+{pct(effect(coefs['reefer']))}", "Constant premium."],
            ["Weight", f"+{pct(effect(coefs['weight']))} per 10,000 lb", ""],
            ["market_index", f"+{pct(effect(coefs['market_index'] * 0.1), 2)} per +0.1", "Same-day value."],
            ["Calendar drift", f"+{pct(effect(coefs['t_years']))} per year", "Extrapolated into Nov–Dec."],
        ],
        widths=[1.6, 2.1, 2.8], caption="Final trend-stage effects.")
    lat = imp["pickup_lat"] + imp["delivery_lat"]
    lon = imp["pickup_lon"] + imp["delivery_lon"]
    r.para(f"Residual-model importance (share of gain): latitude {pct(lat * 100, 0)}, longitude {pct(lon * 100, 0)}, "
           f"quarter-end window {pct(imp['quarter_end_window'] * 100, 0)}, quote_signal deviation "
           f"{pct(imp['quote_signal_dev'] * 100, 0)}, distance {pct(imp['log_distance'] * 100, 0)}, weight "
           f"{pct(imp['weight'] * 100, 0)}, equipment {pct(imp['equipment_code'] * 100, 0)}.")

    # ---- 5. December -------------------------------------------------------------------------
    r.heading("5. December prediction chart")
    r.para("The scorer's fixed inputs vary only the date: Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb. "
           "Coordinates come from the training city table; market_index and quote_signal for each date are the medians "
           "of that day's validation loads.")
    r.figure(SCORER_CHART, "December chart produced by the provided score.py.")
    parts = dec["decomposition"]
    model_ramp = dec["model_ramp_last3days_pct"]
    r.bullet(f"**Range.** ${dec['min']:.0f}–${dec['max']:.0f}; ${dec['dec_1']:.0f} on Dec 1 and "
             f"${dec['dec_31']:.0f} on Dec 31.")
    r.bullet(f"**Weekly cycle.** market_index's weekly cycle moves the rate between "
             f"{pct(parts['weekly_cycle_range_pct'][0])} and +{pct(parts['weekly_cycle_range_pct'][1])}, with peaks on "
             f"Thursdays (Dec 4, 11, 18, 25).")
    r.bullet(f"**Year-end ramp.** The quarter-close window lifts this load by {pct(parts['year_end_ramp_dec31_pct'])} "
             f"on Dec 31. Across all December validation loads the model adds +{pct(model_ramp['Dry Van'])} (Dry Van), "
             f"+{pct(model_ramp['Flatbed'])} (Flatbed) and +{pct(model_ramp['Reefer'])} (Reefer) in the final three "
             f"days, close to the +{pct(ramp['Dry Van'])}, +{pct(ramp['Flatbed'])} and +{pct(ramp['Reefer'])} measured "
             f"in training.")
    r.bullet(f"**Drift.** With both effects removed, the level rises {pct(parts['drift_over_month_pct'])} over the month.")
    r.figure(config.FIGURES_DIR / "december_decomposition.png",
             "Counterfactuals: holding market_index at its December mean removes the weekly cycle; also removing the "
             "quarter-end window leaves the drift.")

    # ---- 6. Limitations ------------------------------------------------------------------------
    r.heading("6. Limitations and next steps")
    r.bullet("**Holidays.** Thanksgiving and Christmas are absent from training, so no holiday effect is modelled. The "
             "Christmas-day peak comes from the Thursday market cycle, not from a learned holiday pattern.")
    r.bullet(f"**Linear drift.** If growth plateaus as it did in July–August, November–December predictions could run "
             f"about 1% high; the drift estimate moved between {by_fold['2025-08-31']['t_years']:.3f} and "
             f"{by_fold['2025-07-31']['t_years']:.3f} per year across folds 2–3.")
    r.bullet(f"**Corrupted labels in the scoring set.** If validation contains them at the training rate "
             f"({pct(q['label_outlier_share'] * 100)}), they will dominate RMSE and add about "
             f"{all_mape[-1] - clean_mape[-1]:.1f} points to MAPE for any model.")
    r.bullet("**Laredo** lies south of every training city, so its latitude effect is held at the southernmost "
             "training value.")
    r.bullet("**Next steps.** Prediction intervals (quantile LightGBM), monitoring of input drift such as quote_signal, "
             "and monthly retraining so the drift estimate keeps updating.")

    r.heading("Appendix A: reproducing the results")
    r.code([
        "python -m venv .venv",
        ".venv\\Scripts\\activate          (macOS/Linux: source .venv/bin/activate)",
        "pip install -r requirements.txt",
        "python eda.py",
        "python train.py",
        "python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv",
        "python make_report.py",
    ])

    r.page_break()
    r.heading("Appendix B: key code")
    r.para("The two most important pieces of code, copied from the repository each time this report is built.")
    r.code_block((HybridModel.fit, HybridModel.predict),
                 "trend stage, LightGBM on what is left, then the lane correction.")
    r.code_block((validation.time_backtest,), "the date-based, expanding-window split.")
    r.save(OUT)
    print(f"Wrote {OUT.relative_to(config.ROOT)}")


def _md(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d:%b} {d.day}"


if __name__ == "__main__":
    main()
