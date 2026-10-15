"""
ml/model.py
-----------
Production prediction pipelines for CropSense.
1. Multi-level Yield Regressor (Crop + State + Season + Climate/Fertilizer elasticity)
2. Underperformance / Risk Classifier
3. Input validation & strict agronomic bounds checking

Designed with pure high-performance vectorized implementations to be 100% resilient
against Windows Smart App Control / Application Control blocking policies.
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.load import load_data

MODEL_DIR = _ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
CACHE_FILE = MODEL_DIR / "predictor_tables.json"

CATEGORICAL = ["Crop", "Season", "State"]
NUMERIC_FEATURES = ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Crop_Year"]

# Practical agronomic input limits
VALIDATION_LIMITS = {
    "Area": {
        "min": 0.1,
        "max": 10000000.0,
        "unit": "hectares (ha)",
        "desc": "Cultivated land area must be between 0.1 ha and 10,000,000 ha."
    },
    "Annual_Rainfall": {
        "min": 50.0,
        "max": 7000.0,
        "unit": "mm",
        "desc": "Annual rainfall must be between 50 mm (arid) and 7,000 mm (monsoon extreme)."
    },
    "Fertilizer": {
        "min": 0.0,
        "max": 5000000000.0,
        "unit": "kg",
        "desc": "Fertilizer must be non-negative and not exceed 5,000,000,000 kg."
    },
    "Pesticide": {
        "min": 0.0,
        "max": 20000000.0,
        "unit": "kg",
        "desc": "Pesticide must be non-negative and not exceed 20,000,000 kg."
    },
    "Crop_Year": {
        "min": 1990,
        "max": 2035,
        "unit": "year",
        "desc": "Crop year must be between 1990 and 2035."
    }
}

def validate_agricultural_inputs(data_dict):
    """
    Validates agricultural input fields:
    - Rejects negative values
    - Checks realistic scientific bounds
    - Checks fertilizer intensity per hectare (e.g. max 5,000 kg/ha)
    Returns: list of error strings (empty if valid)
    """
    errors = []
    
    # 1. Area validation
    try:
        area = float(data_dict.get("Area", data_dict.get("area", 0)))
        if area <= 0:
            errors.append("Cultivated Area cannot be zero or negative. Must be at least 0.1 hectares.")
        elif area > VALIDATION_LIMITS["Area"]["max"]:
            errors.append(f"Cultivated Area ({area:,.1f} ha) exceeds maximum realistic limit of {VALIDATION_LIMITS['Area']['max']:,.0f} ha.")
    except (ValueError, TypeError):
        errors.append("Invalid cultivated area value. Must be a valid positive number.")
        area = 1.0

    # 2. Rainfall validation
    try:
        rf = float(data_dict.get("Annual_Rainfall", data_dict.get("annual_rainfall", 0)))
        if rf < 0:
            errors.append("Annual Rainfall cannot be negative. Rainfall cannot be less than 0 mm.")
        elif rf < VALIDATION_LIMITS["Annual_Rainfall"]["min"]:
            errors.append(f"Annual Rainfall ({rf:.1f} mm) is too low for agricultural feasibility (minimum {VALIDATION_LIMITS['Annual_Rainfall']['min']:.0f} mm).")
        elif rf > VALIDATION_LIMITS["Annual_Rainfall"]["max"]:
            errors.append(f"Annual Rainfall ({rf:,.1f} mm) exceeds maximum observed terrestrial limit of {VALIDATION_LIMITS['Annual_Rainfall']['max']:.0f} mm.")
    except (ValueError, TypeError):
        errors.append("Invalid rainfall value. Must be a valid non-negative number.")

    # 3. Fertilizer validation
    try:
        fert = float(data_dict.get("Fertilizer", data_dict.get("fertilizer", 0)))
        if fert < 0:
            errors.append("Fertilizer quantity cannot be negative.")
        elif fert > VALIDATION_LIMITS["Fertilizer"]["max"]:
            errors.append(f"Fertilizer amount ({fert:,.1f} kg) exceeds allowable ceiling of {VALIDATION_LIMITS['Fertilizer']['max']:,.0f} kg.")
        elif area > 0:
            fert_per_area = fert / area
            if fert_per_area > 6000.0:
                errors.append(f"Fertilizer dosage intensity ({fert_per_area:,.1f} kg/ha) is toxic/unrealistic. Maximum normal agricultural application is 6,000 kg/ha.")
    except (ValueError, TypeError):
        errors.append("Invalid fertilizer value. Must be a valid non-negative number.")

    # 4. Pesticide validation
    try:
        pest = float(data_dict.get("Pesticide", data_dict.get("pesticide", 0)))
        if pest < 0:
            errors.append("Pesticide quantity cannot be negative.")
        elif pest > VALIDATION_LIMITS["Pesticide"]["max"]:
            errors.append(f"Pesticide amount ({pest:,.1f} kg) exceeds allowable ceiling of {VALIDATION_LIMITS['Pesticide']['max']:,.0f} kg.")
        elif area > 0:
            pest_per_area = pest / area
            if pest_per_area > 150.0:
                errors.append(f"Pesticide application rate ({pest_per_area:,.2f} kg/ha) exceeds lethal limits. Maximum safe agricultural usage is 150 kg/ha.")
    except (ValueError, TypeError):
        errors.append("Invalid pesticide value. Must be a valid non-negative number.")

    # 5. Crop year validation
    try:
        year = int(data_dict.get("Crop_Year", data_dict.get("crop_year", 2024)))
        if year < VALIDATION_LIMITS["Crop_Year"]["min"] or year > VALIDATION_LIMITS["Crop_Year"]["max"]:
            errors.append(f"Crop Year ({year}) must be between {VALIDATION_LIMITS['Crop_Year']['min']} and {VALIDATION_LIMITS['Crop_Year']['max']}.")
    except (ValueError, TypeError):
        errors.append("Invalid crop year. Must be a 4-digit integer.")

    return errors

class ProductionPredictor:
    """
    Multi-Level Production Predictor for Crop Yield and Risk Classification.
    Features:
    - Crop + State exact lookup with regional state adjustment fallback
    - Seasonality adjustment multipliers
    - Environmental rainfall and fertilizer non-linear elasticity curves
    - 100% pure NumPy, immune to Windows Smart App Control DLL blocks
    """
    def __init__(self, table=None):
        if table is None:
            self.table = self._build_tables()
        else:
            self.table = table

    def _build_tables(self):
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        df = load_data()

        # 1. Exact Crop-State Pairings
        crop_state_stats = {}
        for (crop, state), grp in df.groupby(["Crop", "State"]):
            key = f"{crop.lower()}|{state.lower()}"
            crop_state_stats[key] = {
                "median_yield": float(grp["Yield"].median()),
                "mean_yield": float(grp["Yield"].mean()),
                "mean_rf": float(grp["Annual_Rainfall"].mean()),
                "mean_fert_area": float(grp["Fertilizer_per_Area"].mean()),
                "underperf_rate": float(grp["UNDERPERFORMANCE_FLAG"].mean()),
                "count": int(len(grp))
            }

        # 2. Crop Baseline Statistics
        crop_stats = {}
        for c in df["Crop"].unique():
            sub = df[df["Crop"] == c]
            crop_stats[c.lower()] = {
                "median_yield": float(sub["Yield"].median()),
                "mean_yield": float(sub["Yield"].mean()),
                "std_yield": float(sub["Yield"].std()) if len(sub) > 1 else 1.0,
                "mean_rf": float(sub["Annual_Rainfall"].mean()),
                "mean_fert_area": float(sub["Fertilizer_per_Area"].mean()),
                "underperf_rate": float(sub["UNDERPERFORMANCE_FLAG"].mean()),
                "count": int(len(sub))
            }

        # 3. State Normalized Multipliers
        df["_c_med"] = df.groupby("Crop")["Yield"].transform("median")
        df["_ratio"] = df["Yield"] / (df["_c_med"] + 1e-5)
        state_multipliers = {}
        for s in df["State"].unique():
            state_multipliers[s.lower()] = float(df[df["State"] == s]["_ratio"].median())

        # 4. Season Multipliers
        season_multipliers = {}
        for se in df["Season"].unique():
            season_multipliers[se.lower()] = float(df[df["Season"] == se]["_ratio"].median())

        global_stats = {
            "median_yield": float(df["Yield"].median()),
            "mean_yield": float(df["Yield"].mean()),
            "mean_rf": float(df["Annual_Rainfall"].mean()),
            "mean_fert_area": float(df["Fertilizer_per_Area"].mean()),
            "underperf_rate": float(df["UNDERPERFORMANCE_FLAG"].mean()),
        }

        full_table = {
            "crop_state_stats": crop_state_stats,
            "crop_stats": crop_stats,
            "state_multipliers": state_multipliers,
            "season_multipliers": season_multipliers,
            "global_stats": global_stats
        }

        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(full_table, f, indent=2)

        return full_table

    def predict_yield_single(self, row_dict):
        crop = str(row_dict.get("Crop", row_dict.get("crop", ""))).strip().lower()
        state = str(row_dict.get("State", row_dict.get("state", ""))).strip().lower()
        season = str(row_dict.get("Season", row_dict.get("season", ""))).strip().lower()

        area = max(0.01, float(row_dict.get("Area", row_dict.get("area", 1.0))))
        rf = max(0.0, float(row_dict.get("Annual_Rainfall", row_dict.get("annual_rainfall", 1000.0))))
        fert = max(0.0, float(row_dict.get("Fertilizer", row_dict.get("fertilizer", 1000.0))))
        pest = max(0.0, float(row_dict.get("Pesticide", row_dict.get("pesticide", 10.0))))

        pair_key = f"{crop}|{state}"
        paired_stats = self.table.get("crop_state_stats", {}).get(pair_key)

        crop_stats = self.table.get("crop_stats", {}).get(crop)
        if not crop_stats:
            crop_stats = self.table.get("global_stats", {})

        # Step A: Determine Baseline Yield with State Sensitivity
        if paired_stats:
            base_yield = paired_stats["median_yield"]
            mean_rf = paired_stats["mean_rf"]
            mean_fert_area = paired_stats["mean_fert_area"]
        else:
            # Fallback using state multiplier
            raw_base = crop_stats.get("median_yield", 2.5)
            s_mult = self.table.get("state_multipliers", {}).get(state, 1.0)
            base_yield = raw_base * s_mult
            mean_rf = crop_stats.get("mean_rf", 1100.0)
            mean_fert_area = crop_stats.get("mean_fert_area", 80.0)

        # Step B: Seasonality Multiplier
        se_mult = self.table.get("season_multipliers", {}).get(season, 1.0)

        # Step C: Rainfall and Fertilizer Elasticity
        fert_per_area = fert / area
        rf_diff = (rf - mean_rf) / (mean_rf + 1.0)
        fert_diff = (fert_per_area - mean_fert_area) / (mean_fert_area + 1.0)

        rf_effect = 0.12 * np.tanh(rf_diff)
        fert_effect = 0.08 * np.tanh(fert_diff)

        # Step D: Pesticide Protection Impact
        pest_per_area = pest / area
        pest_factor = 0.02 * np.tanh((pest_per_area - 1.5) / 2.0)

        pred = base_yield * se_mult * (1.0 + rf_effect + fert_effect + pest_factor)
        pred = max(0.05, round(float(pred), 3))
        return pred

    def predict_risk_single(self, row_dict):
        crop = str(row_dict.get("Crop", row_dict.get("crop", ""))).strip().lower()
        state = str(row_dict.get("State", row_dict.get("state", ""))).strip().lower()
        season = str(row_dict.get("Season", row_dict.get("season", ""))).strip().lower()

        area = max(0.01, float(row_dict.get("Area", row_dict.get("area", 1.0))))
        rf = max(0.0, float(row_dict.get("Annual_Rainfall", row_dict.get("annual_rainfall", 1000.0))))
        fert = max(0.0, float(row_dict.get("Fertilizer", row_dict.get("fertilizer", 1000.0))))

        pair_key = f"{crop}|{state}"
        paired_stats = self.table.get("crop_state_stats", {}).get(pair_key)

        crop_stats = self.table.get("crop_stats", {}).get(crop)
        if not crop_stats:
            crop_stats = self.table.get("global_stats", {})

        if paired_stats:
            base_risk = paired_stats.get("underperf_rate", 0.08)
        else:
            base_risk = crop_stats.get("underperf_rate", 0.11)

        fert_per_area = fert / area

        # Underperformance trigger: ample rainfall & fert, but deficit occurs
        rf_factor = 1.0 / (1.0 + np.exp(-((rf - 800.0) / 350.0)))
        fert_factor = 1.0 / (1.0 + np.exp(-((fert_per_area - 20.0) / 25.0)))

        prob = base_risk * 0.45 + 0.30 * rf_factor + 0.25 * fert_factor
        prob = float(np.clip(prob, 0.02, 0.95))
        pred_class = int(prob >= 0.5)

        return pred_class, round(prob, 4)

class YieldModelWrapper:
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            records = X.to_dict(orient="records")
        else:
            records = [X]
        return np.array([self.predictor.predict_yield_single(r) for r in records])

class RiskModelWrapper:
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            records = X.to_dict(orient="records")
        else:
            records = [X]
        return np.array([self.predictor.predict_risk_single(r)[0] for r in records])

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            records = X.to_dict(orient="records")
        else:
            records = [X]
        probs = []
        for r in records:
            p = self.predictor.predict_risk_single(r)[1]
            probs.append([1.0 - p, p])
        return np.array(probs)

def load_models():
    """
    Returns (risk_model, yield_model) with .predict() and .predict_proba() interface.
    Sensitive to Crop, State, Season, Area, Rainfall, Fertilizer, and Pesticide.
    """
    predictor = ProductionPredictor()
    return RiskModelWrapper(predictor), YieldModelWrapper(predictor)

if __name__ == "__main__":
    print("Testing multi-level state sensitivity...")
    risk_m, yield_m = load_models()
    
    for st in ["Bihar", "Tamil Nadu", "Gujarat", "Punjab", "Chhattisgarh"]:
        row = pd.DataFrame([{
            "Crop": "Bajra", "Season": "Kharif", "State": st,
            "Area": 1000.0, "Annual_Rainfall": 1000.0, "Fertilizer": 50000.0,
            "Pesticide": 300.0, "Crop_Year": 2024
        }])
        pred = yield_m.predict(row)[0]
        print(f"Bajra in {st:<15} -> Predicted Yield: {pred:.3f} t/ha")
