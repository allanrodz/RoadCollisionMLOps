from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_LIST_PATH, FEATURES_PATH, LAGS, MODEL_DIR, ROLLING_WINDOWS, VALIDATED_DATA_PATH


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy().sort_values(["year", "month"]).reset_index(drop=True)
    data["date"] = pd.to_datetime(dict(year=data["year"], month=data["month"], day=1))
    data["trend"] = np.arange(len(data))
    data["month_num"] = data["date"].dt.month
    data["month_sin"] = np.sin(2 * np.pi * data["month_num"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month_num"] / 12)

    for lag in LAGS:
        data[f"lag_{lag}"] = data["collisions"].shift(lag)

    for window in ROLLING_WINDOWS:
        data[f"rolling_mean_{window}"] = data["collisions"].shift(1).rolling(window=window).mean()

    feature_cols = [
        "trend",
        "month_num",
        "month_sin",
        "month_cos",
        *[f"lag_{lag}" for lag in LAGS],
        *[f"rolling_mean_{window}" for window in ROLLING_WINDOWS],
    ]
    modelling = data.dropna(subset=feature_cols + ["collisions"]).reset_index(drop=True)
    return modelling, feature_cols


def create_feature_file(validated_path: Path = VALIDATED_DATA_PATH) -> Path:
    if not validated_path.exists():
        raise FileNotFoundError(f"Validated dataset not found: {validated_path}. Run python -m src.validate_data first.")
    df = pd.read_csv(validated_path)
    features, feature_cols = build_features(df)
    if len(features) < 24:
        raise ValueError("Not enough monthly observations after lag creation. Use at least 36 months for a useful demo.")

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False)
    FEATURE_LIST_PATH.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    print(f"Wrote {FEATURES_PATH} with {len(features)} rows")
    print(f"Feature list: {feature_cols}")
    return FEATURES_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create lag, rolling and seasonal features for monthly forecasting.")
    parser.add_argument("--validated-path", default=str(VALIDATED_DATA_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_feature_file(Path(args.validated_path))
