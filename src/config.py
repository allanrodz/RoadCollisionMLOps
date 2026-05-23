from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
MODEL_DIR = PROJECT_ROOT / os.getenv("MODEL_DIR", "model")

# Bundled Central Statistics Office / data.gov.ie CSV exports.
# ROA17 is the primary monthly dataset used for the forecasting model.
BUNDLED_ROA17_PATH = RAW_DIR / os.getenv("BUNDLED_ROA17_FILE", "ROA17.20260523131010.csv")
REFERENCE_DATASETS = {
    "ROA18": RAW_DIR / "ROA18.20260523130857.csv",  # day-of-week breakdown
    "ROA19": RAW_DIR / "ROA19.20260523130918.csv",  # hour-of-day breakdown
    "ROA20": RAW_DIR / "ROA20.20260523125828.csv",  # county breakdown, older layout
    "ROA27": RAW_DIR / "ROA27.20260523130654.csv",  # county breakdown, newer layout
}

TARGET_NAME = os.getenv("TARGET_NAME", "All Fatal and Injury Collisions")
DATA_URL = os.getenv("DATA_URL", "")
ALLOW_MISSING_MONTHS = os.getenv("ALLOW_MISSING_MONTHS", "false").lower() == "true"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "road-collision-forecasting")
MODEL_NAME = os.getenv("MODEL_NAME", "road_collision_forecaster")
REGISTER_MODEL = os.getenv("REGISTER_MODEL", "false").lower() == "true"

VALIDATED_DATA_PATH = PROCESSED_DIR / "validated_collisions.csv"
FEATURES_PATH = PROCESSED_DIR / "features.csv"
FEATURE_LIST_PATH = MODEL_DIR / "feature_list.json"
MODEL_PATH = MODEL_DIR / "model.joblib"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"
MODEL_HISTORY_PATH = MODEL_DIR / "history.csv"

LAGS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]

for directory in [RAW_DIR, PROCESSED_DIR, METADATA_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
