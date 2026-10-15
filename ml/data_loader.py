from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = RAW_DIR / "crop_yield.csv"

def load_raw_data():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)
    return df

def build_dataset(save=True):
    df = load_raw_data()

    # Clean whitespace in string columns
    for col in ["Crop", "Season", "State"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Drop any nulls or negative values
    df = df.dropna().copy()
    df = df[df["Area"] > 0].copy()
    df = df[df["Yield"] >= 0].copy()

    # Compute intensity features
    df["Fertilizer_per_Area"] = (df["Fertilizer"] / df["Area"]).clip(0, 10000).round(4)
    df["Pesticide_per_Area"] = (df["Pesticide"] / df["Area"]).clip(0, 1000).round(4)

    # Calculate crop-specific median yield to set realistic benchmarks
    crop_medians = df.groupby("Crop")["Yield"].median().to_dict()
    df["Crop_Median_Yield"] = df["Crop"].map(crop_medians)

    # Operational anomaly/underperformance indicator:
    # Flag = 1 when rainfall is good (>800 mm) and fertilizer applied (>20 kg/ha),
    # but the yield is less than 50% of the normal median yield for that specific crop.
    df["UNDERPERFORMANCE_FLAG"] = (
        (df["Annual_Rainfall"] > 800) &
        (df["Fertilizer_per_Area"] > 20) &
        (df["Yield"] < (df["Crop_Median_Yield"] * 0.5))
    ).astype(int)

    # Productivity binary flag (above vs below crop median)
    df["HIGH_PRODUCTIVITY"] = (df["Yield"] >= df["Crop_Median_Yield"]).astype(int)

    # Sort deterministically
    df = df.sort_values(["State", "Crop", "Crop_Year"]).reset_index(drop=True)

    if save:
        out = PROCESSED_DIR / "crop_yield_processed.csv"
        df.to_csv(out, index=False)
        return df, out

    return df, None

if __name__ == "__main__":
    df, out = build_dataset()
    print(f"Processed shape: {df.shape}")
    print(f"Underperformance count: {int(df['UNDERPERFORMANCE_FLAG'].sum())} ({df['UNDERPERFORMANCE_FLAG'].mean()*100:.2f}%)")
    print(f"Saved: {out}")
