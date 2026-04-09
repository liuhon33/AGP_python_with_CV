#!/usr/bin/env python3
"""
Create a poster-style side-by-side AGP vs UKB demographics table.

This version matches the projection notebook more closely:
- UKB is filtered by the same row-missingness rule used in RF_PCA_projection
  unless --ukb-already-filtered is supplied.
- AGP numeric code 5 is optionally treated as unknown/missing (default),
  matching the RF training notebook.

Default inputs follow the user's notebooks:
- AGP metadata: ./Data/Cleaned_data/processed_metadata.csv
- UKB metadata: variable_mapping/ukb_as_agp_metadata.csv

Outputs:
- demographics_table_poster.csv
- demographics_table_poster.xlsx
- demographics_table_poster.png
- demographics_table_poster.pdf
- ukb_filtered_for_poster.csv
- ukb_filter_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_AGP_PATH = "./Data/Cleaned_data/processed_metadata.csv"
DEFAULT_UKB_PATH = "variable_mapping/ukb_as_agp_metadata.csv"
DEFAULT_OUTPUT_DIR = "./hongrui_result/poster_demographics"
DEFAULT_INDEX_COL = "sample_name"
DEFAULT_COGNITION_COL = "fluid_intelligence_score"

# Same defaults as the projection notebook Cell 2
DEFAULT_SKIP_COLS = [
    "sugar_sweetened_drink_frequency",
    "free_sugar_scaled_0_5",
    "artificial_sweeteners",
    "one_liter_of_water_a_day_frequency",
    "olive_oil",
    "prepared_meals_frequency",
    "ready_to_eat_meals_frequency",
    "probiotic_frequency",
    "whole_eggs",
    "sugary_sweets_frequency",
]
DEFAULT_MAX_MISSING_FRAC = 0.10

EM_DASH = "—"


# ============================================================
# Helpers
# ============================================================
def load_csv(path: str, index_col: Optional[str] = DEFAULT_INDEX_COL) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        return pd.read_csv(path, index_col=index_col)
    except ValueError:
        return pd.read_csv(path)


def apply_agp_unknown_code(df: pd.DataFrame, unknown_code: Optional[float] = 5) -> pd.DataFrame:
    if unknown_code is None:
        return df.copy()
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace(unknown_code, np.nan)
    return out


def filter_rows_by_missingness(
    df: pd.DataFrame,
    *,
    skip_cols: Sequence[str],
    required_cols: Sequence[str],
    max_missing_frac: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Reproduces the projection notebook filtering logic:
    - calculate per-row missingness on all columns except skip_cols
    - keep rows with missing_frac <= max_missing_frac
    - then drop rows missing required_cols that are present in the dataframe
    """
    skip_cols = list(skip_cols)
    required_cols = list(required_cols)

    missing_skip = [c for c in skip_cols if c not in df.columns]
    cols_check = [c for c in df.columns if c not in set(skip_cols)]
    if len(cols_check) == 0:
        raise ValueError("After applying skip_cols, no columns remain for missingness filtering.")

    missing_frac = df[cols_check].isna().mean(axis=1)
    n_in = int(df.shape[0])

    keep_mask = missing_frac <= max_missing_frac
    df_f = df.loc[keep_mask].copy()

    required_present = [c for c in required_cols if c in df_f.columns]
    before_required = int(df_f.shape[0])
    if required_present:
        df_f = df_f.dropna(subset=required_present)

    report = {
        "n_in": n_in,
        "n_after_missing_frac": before_required,
        "n_out": int(df_f.shape[0]),
        "dropped_total": n_in - int(df_f.shape[0]),
        "dropped_by_missing_frac": n_in - before_required,
        "dropped_by_required_cols": before_required - int(df_f.shape[0]),
        "cols_check_n": len(cols_check),
        "skip_cols_requested": skip_cols,
        "skip_cols_missing_from_dataframe": missing_skip,
        "required_cols_requested": required_cols,
        "required_cols_present": required_present,
        "max_missing_frac": max_missing_frac,
        "missing_frac_summary_before_filter": {
            k: (None if pd.isna(v) else float(v))
            for k, v in missing_frac.describe().to_dict().items()
        },
    }
    return df_f, report


def fmt_n(n: int) -> str:
    return f"{int(n):,}"


def fmt_mean_sd(series: pd.Series, digits: int = 1) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return EM_DASH
    return f"{s.mean():.{digits}f} ± {s.std(ddof=1):.{digits}f}"


def fmt_median_iqr(series: pd.Series, digits: int = 1) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return EM_DASH
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    return f"{s.median():.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def fmt_pct(numer: int, denom: int, digits: int = 1) -> str:
    if denom <= 0:
        return EM_DASH
    return f"{numer:,} ({100 * numer / denom:.{digits}f}%)"


def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return df[col]


def pretty_label(col: str) -> str:
    custom = {
        "age_corrected": "Age, years",
        "bmi": "BMI, kg/m²",
        "sex_female": "Female sex",
        "fluid_intelligence_score": "Fluid intelligence score",
        "highest_education": "Highest education",
        "race": "Race / ethnicity",
        "alcohol_frequency": "Alcohol frequency (0–5)",
        "smoking_frequency": "Smoking frequency (0–5)",
        "exercise_frequency": "Exercise frequency (0–5)",
        "fruit_frequency": "Fruit frequency (0–5)",
        "vegetable_frequency": "Vegetable frequency (0–5)",
        "seafood_frequency": "Seafood / oily fish frequency (0–5)",
        "red_meat_frequency": "Red meat frequency (0–5)",
        "high_fat_red_meat_frequency": "Processed meat frequency (0–5)",
        "milk_cheese_frequency": "Milk / cheese frequency (0–5)",
        "whole_grain_frequency": "Whole grain / cereal frequency (0–5)",
        "bowel_movement_frequency": "Bowel movement frequency (0–5)",
        "sleep_duration": "Sleep duration (0–5)",
    }
    return custom.get(col, col.replace("_", " ").title())


def normalize_binary_female(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)

    if pd.api.types.is_numeric_dtype(series):
        s = pd.to_numeric(series, errors="coerce")
        out = pd.Series(np.nan, index=s.index, dtype=float)
        out.loc[s == 1] = 1.0
        out.loc[s == 0] = 0.0
        return out

    xs = series.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=xs.index, dtype=float)
    out.loc[xs.isin(["female", "f", "woman"])] = 1.0
    out.loc[xs.isin(["male", "m", "man"])] = 0.0
    return out


def summarize_n(df: pd.DataFrame) -> str:
    return fmt_n(df.shape[0])


def summarize_continuous(df: pd.DataFrame, col: str, *, digits: int = 1) -> str:
    if col not in df.columns:
        return EM_DASH
    return fmt_mean_sd(pd.to_numeric(df[col], errors="coerce"), digits=digits)


def summarize_ordinal(df: pd.DataFrame, col: str, *, digits: int = 1) -> str:
    if col not in df.columns:
        return EM_DASH
    return fmt_median_iqr(pd.to_numeric(df[col], errors="coerce"), digits=digits)


def summarize_female(df: pd.DataFrame) -> str:
    if "sex" not in df.columns:
        return EM_DASH
    s = normalize_binary_female(df["sex"]).dropna()
    if s.empty:
        return EM_DASH
    return fmt_pct(int((s == 1).sum()), int(s.shape[0]))


def summarize_categorical_level(df: pd.DataFrame, col: str, level: str) -> str:
    if col not in df.columns:
        return EM_DASH
    s = df[col].astype("string").dropna()
    if s.empty:
        return EM_DASH
    count = int((s == level).sum())
    return fmt_pct(count, int(s.shape[0]))


def summarize_categorical_top_levels(
    df: pd.DataFrame,
    col: str,
    *,
    levels: Optional[Sequence[str]] = None,
    top_n_if_missing: int = 5,
) -> list[str]:
    """
    Returns the levels to display for a categorical variable.
    If levels are supplied, preserves that order and appends any extras present in either cohort later.
    """
    if col not in df.columns:
        return list(levels) if levels is not None else []
    s = df[col].astype("string").dropna()
    if s.empty:
        return list(levels) if levels is not None else []
    if levels is not None:
        return list(levels)
    return list(s.value_counts().head(top_n_if_missing).index.astype(str))


# ============================================================
# Table assembly
# ============================================================
def build_table(agp_df: pd.DataFrame, ukb_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def add_row(section: str, variable: str, agp_val: str, ukb_val: str) -> None:
        rows.append(
            {
                "Section": section,
                "Variable": variable,
                "AGP": agp_val,
                "UKB": ukb_val,
            }
        )

    # Overall
    add_row("Overall", "N", summarize_n(agp_df), summarize_n(ukb_df))

    # Continuous demographics
    add_row("Demographics", pretty_label("age_corrected"),
            summarize_continuous(agp_df, "age_corrected"),
            summarize_continuous(ukb_df, "age_corrected"))

    add_row("Demographics", pretty_label("bmi"),
            summarize_continuous(agp_df, "bmi"),
            summarize_continuous(ukb_df, "bmi"))

    add_row("Demographics", pretty_label("sex_female"),
            summarize_female(agp_df),
            summarize_female(ukb_df))

    # Race / ethnicity
    race_order = [
        "Caucasian",
        "Asian or Pacific Islander",
        "African American",
        "Hispanic",
        "Other",
    ]
    if "race" in agp_df.columns or "race" in ukb_df.columns:
        add_row("Demographics", pretty_label("race"), "", "")
        agp_levels = set(get_series(agp_df, "race").astype("string").dropna().unique().tolist())
        ukb_levels = set(get_series(ukb_df, "race").astype("string").dropna().unique().tolist())
        extra_levels = sorted((agp_levels | ukb_levels) - set(race_order))
        for lvl in race_order + extra_levels:
            add_row(
                "Demographics",
                f"  {lvl}",
                summarize_categorical_level(agp_df, "race", lvl),
                summarize_categorical_level(ukb_df, "race", lvl),
            )

    # Education
    edu_preferred = ["Degree", "Alevel_vocational_professional", "Secondary", "None", "Other"]
    if "highest_education" in agp_df.columns or "highest_education" in ukb_df.columns:
        add_row("Demographics", pretty_label("highest_education"), "", "")
        agp_levels = set(get_series(agp_df, "highest_education").astype("string").dropna().unique().tolist())
        ukb_levels = set(get_series(ukb_df, "highest_education").astype("string").dropna().unique().tolist())
        combined = agp_levels | ukb_levels
        ordered = [x for x in edu_preferred if x in combined]
        ordered += sorted(combined - set(ordered))
        for lvl in ordered:
            add_row(
                "Demographics",
                f"  {lvl}",
                summarize_categorical_level(agp_df, "highest_education", lvl),
                summarize_categorical_level(ukb_df, "highest_education", lvl),
            )

    # Lifestyle / proxy features (compact poster version)
    ordinal_vars = [
        "alcohol_frequency",
        "smoking_frequency",
        "exercise_frequency",
        "fruit_frequency",
        "vegetable_frequency",
        "seafood_frequency",
        "red_meat_frequency",
        "high_fat_red_meat_frequency",
        "milk_cheese_frequency",
        "whole_grain_frequency",
        "bowel_movement_frequency",
        "sleep_duration",
    ]
    for col in ordinal_vars:
        if col in agp_df.columns or col in ukb_df.columns:
            add_row(
                "Lifestyle variables",
                pretty_label(col),
                summarize_ordinal(agp_df, col),
                summarize_ordinal(ukb_df, col),
            )

    # UKB outcome shown on the right, as requested
    if "fluid_intelligence_score" in agp_df.columns or "fluid_intelligence_score" in ukb_df.columns:
        add_row(
            "UKB outcome",
            pretty_label("fluid_intelligence_score"),
            summarize_continuous(agp_df, "fluid_intelligence_score"),
            summarize_continuous(ukb_df, "fluid_intelligence_score"),
        )

    return pd.DataFrame(rows)


# ============================================================
# Rendering
# ============================================================
def render_table_as_figure(table_df: pd.DataFrame, output_png: str, output_pdf: str, title: str) -> None:
    display_df = table_df[["Variable", "AGP", "UKB"]].copy()

    n_rows = display_df.shape[0]
    fig_h = max(7.5, 0.36 * (n_rows + 5))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=18)

    table = ax.table(
        cellText=display_df.values,
        colLabels=["Variable", "AGP", "UKB"],
        loc="upper center",
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0, 1, 0.96],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    # Column widths
    widths = {0: 0.44, 1: 0.28, 2: 0.28}
    for (r, c), cell in table.get_celld().items():
        if c in widths:
            cell.set_width(widths[c])

        # Header
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(cell.get_height() * 1.2)

    # Bold section header rows using the original table_df
    for i, row in table_df.reset_index(drop=True).iterrows():
        # matplotlib table row index includes header, so +1
        rr = i + 1
        is_section_header = (row["AGP"] == "" and row["UKB"] == "")
        if is_section_header:
            for cc in range(3):
                cell = table[(rr, cc)]
                cell.set_text_props(weight="bold")
                cell.set_height(cell.get_height() * 1.05)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build AGP vs filtered-UKB poster demographics table.")
    p.add_argument("--agp", default=DEFAULT_AGP_PATH, help="Path to AGP metadata CSV.")
    p.add_argument("--ukb", default=DEFAULT_UKB_PATH, help="Path to UKB AGP-like metadata CSV.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    p.add_argument("--index-col", default=DEFAULT_INDEX_COL, help="CSV index column if present.")
    p.add_argument("--agp-unknown-code", type=float, default=5,
                   help="Numeric code treated as unknown in AGP (default 5). Use -1 to disable.")
    p.add_argument("--ukb-already-filtered", action="store_true",
                   help="Use UKB file as-is; do not re-apply projection-style row filtering.")
    p.add_argument("--cognition-col", default=DEFAULT_COGNITION_COL,
                   help="Outcome column required in UKB row filtering.")
    p.add_argument("--max-missing-frac", type=float, default=DEFAULT_MAX_MISSING_FRAC,
                   help="Maximum allowed per-row missing fraction for UKB filtering.")
    p.add_argument("--title", default="Poster Demographics Table: AGP vs UKB",
                   help="Figure title.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    agp_df = load_csv(args.agp, index_col=args.index_col)
    ukb_df = load_csv(args.ukb, index_col=args.index_col)

    # AGP cleaning to match RF notebook
    agp_unknown_code = None if args.agp_unknown_code == -1 else args.agp_unknown_code
    agp_df = apply_agp_unknown_code(agp_df, agp_unknown_code)

    # UKB filtering to match projection notebook
    if args.ukb_already_filtered:
        filter_report = {
            "status": "UKB file treated as already filtered; no additional filtering applied."
        }
        ukb_filtered = ukb_df.copy()
    else:
        required_cols = [args.cognition_col, "age_corrected"]
        ukb_filtered, filter_report = filter_rows_by_missingness(
            ukb_df,
            skip_cols=DEFAULT_SKIP_COLS,
            required_cols=required_cols,
            max_missing_frac=args.max_missing_frac,
        )

    # Build poster table
    table_df = build_table(agp_df, ukb_filtered)

    # Save outputs
    csv_path = os.path.join(args.output_dir, "demographics_table_poster.csv")
    xlsx_path = os.path.join(args.output_dir, "demographics_table_poster.xlsx")
    png_path = os.path.join(args.output_dir, "demographics_table_poster.png")
    pdf_path = os.path.join(args.output_dir, "demographics_table_poster.pdf")
    ukb_filtered_path = os.path.join(args.output_dir, "ukb_filtered_for_poster.csv")
    report_path = os.path.join(args.output_dir, "ukb_filter_report.json")

    table_df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        table_df.to_excel(writer, index=False, sheet_name="Poster_table")
        pd.DataFrame([filter_report]).to_excel(writer, index=False, sheet_name="UKB_filter_report")

    ukb_filtered.to_csv(ukb_filtered_path)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(filter_report, f, indent=2)

    render_table_as_figure(table_df, png_path, pdf_path, args.title)

    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {xlsx_path}")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    print(f"  {ukb_filtered_path}")
    print(f"  {report_path}")
    print("\nUKB filter report:")
    print(json.dumps(filter_report, indent=2))


if __name__ == "__main__":
    main()
