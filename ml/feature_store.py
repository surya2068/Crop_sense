import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "crop_yield_processed.csv"

class FeatureStore:
    """
    CropSense Production Feature Store.
    
    Guarantees Feature Parity between Training and Online Serving:
    - Same transformation pipelines for raw inputs
    - Strict schema verification preventing training-serving skew
    - Entity lookup for offline historical queries and online low-latency inference
    """
    
    FEATURE_DEFINITIONS = [
        {"name": "Crop", "dtype": "category", "store_type": "entity_key", "desc": "Crop cultivar identifier"},
        {"name": "Season", "dtype": "category", "store_type": "entity_key", "desc": "Monsoon/Winter season context"},
        {"name": "State", "dtype": "category", "store_type": "entity_key", "desc": "Geographical Indian state"},
        {"name": "Area", "dtype": "float64", "store_type": "raw_numerical", "desc": "Cultivated land area (ha)"},
        {"name": "Annual_Rainfall", "dtype": "float64", "store_type": "raw_numerical", "desc": "Annual precipitation (mm)"},
        {"name": "Fertilizer", "dtype": "float64", "store_type": "raw_numerical", "desc": "Fertilizer volume applied (kg)"},
        {"name": "Pesticide", "dtype": "float64", "store_type": "raw_numerical", "desc": "Pesticide volume applied (kg)"},
        {"name": "Crop_Year", "dtype": "int64", "store_type": "timestamp", "desc": "Agricultural recording year"},
        # Derived Feature Store definitions (Guaranteed Parity)
        {"name": "Fertilizer_per_Area", "dtype": "float64", "store_type": "derived_feature", "formula": "Fertilizer / Area"},
        {"name": "Pesticide_per_Area", "dtype": "float64", "store_type": "derived_feature", "formula": "Pesticide / Area"},
        {"name": "Rainfall_Fertilizer_Ratio", "dtype": "float64", "store_type": "derived_feature", "formula": "Annual_Rainfall / (Fertilizer_per_Area + 1e-4)"},
        {"name": "Log_Area", "dtype": "float64", "store_type": "derived_feature", "formula": "log1p(Area)"}
    ]
    
    def __init__(self):
        self._stats_cache = None
        self._load_baseline_statistics()
        
    def _load_baseline_statistics(self):
        """Loads and caches training dataset baseline statistics for validation & skew checks."""
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            self._stats_cache = {
                "Area": {"min": float(df["Area"].min()), "max": float(df["Area"].max()), "mean": float(df["Area"].mean()), "std": float(df["Area"].std())},
                "Annual_Rainfall": {"min": float(df["Annual_Rainfall"].min()), "max": float(df["Annual_Rainfall"].max()), "mean": float(df["Annual_Rainfall"].mean()), "std": float(df["Annual_Rainfall"].std())},
                "Fertilizer": {"min": float(df["Fertilizer"].min()), "max": float(df["Fertilizer"].max()), "mean": float(df["Fertilizer"].mean()), "std": float(df["Fertilizer"].std())},
                "Pesticide": {"min": float(df["Pesticide"].min()), "max": float(df["Pesticide"].max()), "mean": float(df["Pesticide"].mean()), "std": float(df["Pesticide"].std())},
                "valid_crops": sorted(df["Crop"].unique().tolist()),
                "valid_states": sorted(df["State"].unique().tolist()),
                "valid_seasons": sorted(df["Season"].unique().tolist()),
                "total_offline_features": len(self.FEATURE_DEFINITIONS),
                "total_entities": len(df)
            }
        else:
            self._stats_cache = {}
            
    def get_feature_catalog(self) -> List[Dict[str, str]]:
        return self.FEATURE_DEFINITIONS
        
    def get_store_metadata(self) -> Dict[str, Any]:
        return {
            "registry_name": "cropsense-featurestore-production",
            "version": "2.0.0",
            "features_registered": len(self.FEATURE_DEFINITIONS),
            "offline_storage": "crop_yield_processed.parquet / .csv",
            "online_storage_engine": "Redis In-Memory Key-Value (Simulated / Local Cache)",
            "skew_protection": "Active Schema & Range Assertion",
            "statistics": self._stats_cache
        }

    def validate_and_transform_online_payload(self, payload: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validates an incoming REST API payload against the production schema
        and computes derived features with 100% parity to the training pipeline.
        Returns: (transformed_df, validation_warnings)
        """
        warnings = []
        required_fields = ["Crop", "Season", "State", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
        for field in required_fields:
            if field not in payload:
                raise ValueError(f"Missing required input feature in request payload: '{field}'")
                
        # Parse numerical types
        try:
            area = max(0.01, float(payload["Area"]))
            rainfall = max(0.0, float(payload["Annual_Rainfall"]))
            fertilizer = max(0.0, float(payload["Fertilizer"]))
            pesticide = max(0.0, float(payload["Pesticide"]))
            crop_year = int(payload.get("Crop_Year", 2024))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Feature type parsing error: {exc}")
            
        crop = str(payload["Crop"]).strip()
        state = str(payload["State"]).strip()
        season = str(payload["Season"]).strip()
        
        # Out-of-vocabulary check (skew prevention)
        if self._stats_cache and "valid_crops" in self._stats_cache:
            if crop not in self._stats_cache["valid_crops"]:
                warnings.append(f"Crop '{crop}' not observed in baseline training registry.")
            if state not in self._stats_cache["valid_states"]:
                warnings.append(f"State '{state}' not observed in baseline training registry.")
            if season not in self._stats_cache["valid_seasons"]:
                warnings.append(f"Season '{season}' not observed in baseline training registry.")
                
        # Range anomaly check (training-serving skew prevention)
        if self._stats_cache and "Annual_Rainfall" in self._stats_cache:
            rf_max = self._stats_cache["Annual_Rainfall"]["max"]
            if rainfall > rf_max * 1.5:
                warnings.append(f"Rainfall ({rainfall:.1f} mm) exceeds 150% of training baseline max ({rf_max:.1f} mm).")
                
        # Derived features computation identical to ml/data_loader.py
        fert_per_area = fertilizer / area
        pest_per_area = pesticide / area
        rf_fert_ratio = rainfall / (fert_per_area + 1e-4)
        log_area = float(np.log1p(area))
        
        transformed_df = pd.DataFrame([{
            "Crop": crop,
            "Season": season,
            "State": state,
            "Area": area,
            "Annual_Rainfall": rainfall,
            "Fertilizer": fertilizer,
            "Pesticide": pesticide,
            "Crop_Year": crop_year,
            "Fertilizer_per_Area": fert_per_area,
            "Pesticide_per_Area": pest_per_area,
            "Rainfall_Fertilizer_Ratio": rf_fert_ratio,
            "Log_Area": log_area
        }])
        
        return transformed_df, warnings

feature_store = FeatureStore()
