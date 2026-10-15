"""
models/load.py
--------------
Dataset loading and basic inspection helpers for CropSense.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "crop_yield_processed.csv"
RAW_PATH = BASE_DIR / "data" / "raw" / "crop_yield.csv"

def load_data():
    """Load the processed crop dataset, or build it if missing."""
    if not DATA_PATH.exists():
        from ml.data_loader import build_dataset
        df, _ = build_dataset(save=True)
        return df
    return pd.read_csv(DATA_PATH)

def get_first_rows(df, n=5):
    return df.head(n)

def get_last_rows(df, n=5):
    return df.tail(n)

def get_shape(df):
    return df.shape

def get_column_names(df):
    return list(df.columns)

def get_dtypes(df):
    return {col: str(dtype) for col, dtype in df.dtypes.items()}

def get_info(df):
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
        "non_null_counts": {col: int(df[col].notna().sum()) for col in df.columns},
    }

def get_description(df):
    numeric_cols = get_numeric_cols(df)
    if not numeric_cols:
        return {}
    describe = df[numeric_cols].describe().T.round(3)
    return describe.to_dict(orient="index")

def get_missing_values(df):
    missing = df.isna().sum()
    missing = missing[missing > 0]
    return {
        col: {
            "count": int(missing[col]),
            "percent": round(float(missing[col] / len(df) * 100), 3),
        }
        for col in missing.index
    }

def get_duplicate_count(df):
    return int(df.duplicated().sum())

def get_unique_counts(df):
    return {col: int(df[col].nunique()) for col in df.columns}

def get_numeric_cols(df):
    numeric_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ["UNDERPERFORMANCE_FLAG", "HIGH_PRODUCTIVITY"]:
            continue
        numeric_cols.append(col)
    return numeric_cols

def get_categorical_cols(df):
    cat_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    for col in ["UNDERPERFORMANCE_FLAG", "HIGH_PRODUCTIVITY"]:
        if col in df.columns and col not in cat_cols:
            cat_cols.append(col)
    return cat_cols

def get_datetime_cols(df):
    return [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

if __name__ == "__main__":
    df = load_data()
    print("Shape:", get_shape(df))
    print("Columns:", get_column_names(df))
