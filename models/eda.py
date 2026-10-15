"""
models/eda.py
-------------
Exploratory Data Analysis calculations and visualisations for CropSense.
"""

import sys
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models.load import (
    load_data,
    get_categorical_cols,
    get_numeric_cols,
    get_datetime_cols,
)

sns.set_theme(style="whitegrid", palette="muted")

PLOTS_DIR = _ROOT / "static" / "plots"

TARGET_COL = "Yield"
SCATTER_SAMPLE = 10000
RNG_SEED = 42

def _to_records(df):
    out = df.reset_index()
    out = out.rename(columns={"index": "column"})
    return out.to_dict(orient="records")

def get_dataset_info(df):
    numeric_cols = get_numeric_cols(df)
    cat_cols = get_categorical_cols(df)
    datetime_cols = get_datetime_cols(df)
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "datetime_cols": datetime_cols,
        "non_null": {col: int(df[col].notna().sum()) for col in df.columns},
    }

def get_statistical_summary(df):
    numeric_cols = get_numeric_cols(df)
    if not numeric_cols:
        return []
    describe = df[numeric_cols].describe().T.round(3)
    describe["variance"] = df[numeric_cols].var().round(3)
    describe = describe.rename(columns={"50%": "median"})
    return _to_records(describe)

def get_missing_summary(df):
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return [], 0
    rows = []
    for col in missing.index:
        rows.append({
            "column": col,
            "count": int(missing[col]),
            "percent": round(float(missing[col] / len(df) * 100), 3),
        })
    return rows, int(missing.sum())

def get_duplicate_summary(df):
    return int(df.duplicated().sum())

def get_unique_summary(df):
    return [{"column": col, "unique": int(df[col].nunique())} for col in df.columns]

def get_categorical_frequencies(df):
    result = []
    cols_to_check = [c for c in ["Crop", "Season", "State", "UNDERPERFORMANCE_FLAG", "HIGH_PRODUCTIVITY"] if c in df.columns]
    for col in cols_to_check:
        counts = df[col].astype(str).value_counts().head(10)
        result.append({
            "column": col,
            "top_values": [{"value": k, "count": int(v)} for k, v in counts.items()],
        })
    return result

def get_outlier_summary(df):
    numeric_cols = [c for c in get_numeric_cols(df) if c not in ["Crop_Year", "Crop_Median_Yield"]]
    outliers = []
    for col in numeric_cols:
        ser = df[col].dropna()
        if len(ser) == 0:
            continue
        q25 = ser.quantile(0.25)
        q75 = ser.quantile(0.75)
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        count = int(((ser < lower) | (ser > upper)).sum())
        outliers.append({
            "column": col,
            "q25": round(float(q25), 3),
            "q75": round(float(q75), 3),
            "iqr": round(float(iqr), 3),
            "lower_bound": round(float(lower), 3),
            "upper_bound": round(float(upper), 3),
            "outlier_count": count,
            "outlier_percent": round(float(count / len(df) * 100), 2),
        })
    return outliers

def get_correlation_matrix(df):
    numeric_cols = [c for c in get_numeric_cols(df) if c not in ["Crop_Year", "Crop_Median_Yield", "Production"]]
    if not numeric_cols:
        return {}
    corr = df[numeric_cols].corr().round(3)
    return {
        "columns": list(corr.columns),
        "data": corr.values.tolist(),
        "dict": corr.to_dict(),
    }

def get_correlation_with_target(df, target=TARGET_COL):
    if target not in df.columns:
        return {}
    numeric_cols = [c for c in get_numeric_cols(df) if c not in [target, "Crop_Year", "Crop_Median_Yield", "Production"]]
    if not numeric_cols:
        return {}
    corr = df[numeric_cols + [target]].corr()[target].drop(target)
    return corr.sort_values(ascending=False).round(3).to_dict()

def get_target_analysis(df, target=TARGET_COL):
    if target not in df.columns:
        return None
    ser = df[target].dropna()
    return {
        "target": target,
        "note": (
            "Crop Yield (harvest production per hectare) is the primary target variable "
            "for agricultural modeling. Models evaluate soil, weather, management, and geospatial inputs."
        ),
        "stats": {
            "count": int(ser.size),
            "mean": round(float(ser.mean()), 3),
            "median": round(float(ser.median()), 3),
            "std": round(float(ser.std()), 3),
            "min": round(float(ser.min()), 3),
            "max": round(float(ser.max()), 3),
            "zero_share_pct": round(float((ser == 0).mean() * 100), 2),
        },
        "correlation_with_target": get_correlation_with_target(df, target),
    }

def _ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)

def _save(fig, filename):
    _ensure_plots_dir()
    path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return filename

def generate_plots(df):
    _ensure_plots_dir()
    plots = []

    plot_df = df
    if len(df) > SCATTER_SAMPLE:
        plot_df = df.sample(SCATTER_SAMPLE, random_state=RNG_SEED)

    # 1. Distribution of Target (Yield)
    fig, ax = plt.subplots(figsize=(7, 4))
    p99 = df[TARGET_COL].quantile(0.99)
    sns.histplot(df[df[TARGET_COL] <= p99][TARGET_COL], kde=True, ax=ax, color="#2e7d32", bins=40)
    ax.set_title("Distribution of Crop Yield (<= 99th Percentile)")
    ax.set_xlabel("Yield (t/ha)")
    ax.set_ylabel("Frequency")
    fname = "yield_distribution.png"
    _save(fig, fname)
    plots.append({"file": fname, "title": "Distribution of Crop Yield", "section": "numerical"})

    # 2. Key numeric distributions
    for col in ["Annual_Rainfall", "Fertilizer", "Pesticide", "Area"]:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            p99_col = df[col].quantile(0.99)
            sns.histplot(df[df[col] <= p99_col][col], kde=True, ax=ax, color="#1976d2", bins=35)
            ax.set_title(f"Distribution of {col} (<= 99th Percentile)")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            fname = f"{col.lower()}_distribution.png"
            _save(fig, fname)
            plots.append({"file": fname, "title": f"Distribution of {col}", "section": "numerical"})

    # 3. Numeric Boxplots
    numeric_box_cols = ["Annual_Rainfall", "Fertilizer_per_Area", "Pesticide_per_Area", "Yield"]
    available_cols = [c for c in numeric_box_cols if c in df.columns]
    if available_cols:
        fig, axes = plt.subplots(nrows=1, ncols=len(available_cols), figsize=(4 * len(available_cols), 4))
        axes = np.atleast_1d(axes)
        for ax, col in zip(axes, available_cols):
            ax.boxplot(df[col].dropna().values)
            ax.set_title(col, fontsize=10)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Outlier Boxplots for Agricultural Features", y=1.02)
        fname = "numeric_boxplots.png"
        _save(fig, fname)
        plots.append({"file": fname, "title": "Boxplots of Agricultural Features", "section": "numerical"})

    # 4. Scatter plots vs Yield
    scatter_pairs = [
        ("Annual_Rainfall", TARGET_COL, "#388e3c"),
        ("Fertilizer", TARGET_COL, "#f57c00"),
        ("Area", TARGET_COL, "#0288d1"),
    ]
    for x, y, color in scatter_pairs:
        if x in df.columns and y in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sub = plot_df[(plot_df[y] <= df[y].quantile(0.99)) & (plot_df[x] <= df[x].quantile(0.99))]
            ax.scatter(sub[x], sub[y], s=10, alpha=0.3, color=color)
            ax.set_title(f"{x} vs {y}")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            fname = f"{x.lower()}_vs_{y.lower()}.png"
            _save(fig, fname)
            plots.append({"file": fname, "title": f"{x} vs {y}", "section": "bivariate"})

    # 5. Categorical: Yield by Season
    if "Season" in df.columns and TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sub_season = df[df[TARGET_COL] <= df[TARGET_COL].quantile(0.95)]
        sns.boxplot(data=sub_season, x="Season", y=TARGET_COL, ax=ax, hue="Season", palette="Set3", legend=False)
        ax.set_title("Crop Yield Distribution by Season")
        ax.tick_params(axis="x", rotation=25)
        fname = "yield_by_season.png"
        _save(fig, fname)
        plots.append({"file": fname, "title": "Crop Yield by Season", "section": "categorical"})

    # 6. Top 10 Crops by Yield
    if "Crop" in df.columns and TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        top_crops = df.groupby("Crop")[TARGET_COL].mean().sort_values(ascending=False).head(10)
        top_crops.plot.barh(ax=ax, color="#43a047")
        ax.set_title("Top 10 Crops by Average Yield")
        ax.set_xlabel("Average Yield")
        fname = "top_crops_yield.png"
        _save(fig, fname)
        plots.append({"file": fname, "title": "Top Crops by Average Yield", "section": "categorical"})

    # 7. Correlation Heatmap
    numeric_for_corr = [c for c in get_numeric_cols(df) if c not in ["Crop_Year", "Crop_Median_Yield", "Production"]]
    if len(numeric_for_corr) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        corr = df[numeric_for_corr].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, ax=ax)
        ax.set_title("Correlation Heatmap of Agricultural Variables")
        fname = "correlation_heatmap.png"
        _save(fig, fname)
        plots.append({"file": fname, "title": "Correlation Heatmap", "section": "multivariate"})

    return plots

def _generate_findings(df):
    findings = []
    n_rows = len(df)
    n_crops = df["Crop"].nunique() if "Crop" in df.columns else 0
    n_states = df["State"].nunique() if "State" in df.columns else 0
    findings.append(
        f"The dataset contains <strong>{n_rows:,} agricultural observations</strong> spanning <strong>{n_crops} distinct crops</strong> across <strong>{n_states} states</strong>."
    )

    if TARGET_COL in df.columns:
        mean_y = df[TARGET_COL].mean()
        med_y = df[TARGET_COL].median()
        findings.append(
            f"Overall crop yield exhibits significant variation with a median of <strong>{med_y:.2f} t/ha</strong> and mean of <strong>{mean_y:.2f} t/ha</strong>, reflecting differing physiological yield potentials across grain, pulse, and plantation crops."
        )

    if "Annual_Rainfall" in df.columns:
        mean_rain = df["Annual_Rainfall"].mean()
        findings.append(
            f"Annual rainfall averages <strong>{mean_rain:.1f} mm</strong>, which acts as a primary environmental driver across diverse agro-climatic zones."
        )

    if "UNDERPERFORMANCE_FLAG" in df.columns:
        under_pct = df["UNDERPERFORMANCE_FLAG"].mean() * 100
        findings.append(
            f"The derived underperformance indicator flags <strong>{under_pct:.1f}%</strong> of crop cycles where rainfall and fertilization were adequate but realized yield fell in the lower quartile, highlighting operational yield-gap risk."
        )

    findings.append(
        "Production is strictly held out from regression predictors to prevent target leakage, ensuring models predict harvest yield based on land size, weather, and input dosages."
    )
    return findings

_CACHE = None

def run_full_eda(df=None, force_recompute=False):
    global _CACHE
    if _CACHE is not None and not force_recompute:
        return _CACHE
    if df is None:
        df = load_data()

    _CACHE = {
        "info": get_dataset_info(df),
        "stats": get_statistical_summary(df),
        "missing": get_missing_summary(df)[0],
        "missing_total": get_missing_summary(df)[1],
        "duplicates": get_duplicate_summary(df),
        "uniques": get_unique_summary(df),
        "cat_frequencies": get_categorical_frequencies(df),
        "outliers": get_outlier_summary(df),
        "correlation_matrix": get_correlation_matrix(df),
        "target": get_target_analysis(df, TARGET_COL),
        "plots": generate_plots(df),
        "findings": _generate_findings(df),
    }
    return _CACHE

if __name__ == "__main__":
    df = load_data()
    res = run_full_eda(df)
    print("EDA Complete. Generated plots:", len(res["plots"]))
