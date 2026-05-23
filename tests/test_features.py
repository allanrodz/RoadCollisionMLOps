from pathlib import Path

import pandas as pd

from src.config import RAW_DIR
from src.features import build_features
from src.validate_data import standardise_raw_dataframe


def latest_raw_roa17_file() -> Path:
    files = sorted(RAW_DIR.glob("roa17_raw_*.csv"))
    assert files, "No downloaded ROA17 raw snapshot found. Run: python -m src.ingest"
    return files[-1]


def test_build_features_has_shifted_lags():
    raw = pd.read_csv(latest_raw_roa17_file(), encoding="utf-8-sig")
    df = standardise_raw_dataframe(raw)

    features, feature_cols = build_features(df)

    assert len(features) > 0
    assert "lag_1" in feature_cols
    assert "lag_12" in feature_cols
    assert "rolling_mean_3" in feature_cols
    assert "rolling_mean_12" in feature_cols

    first = features.iloc[0]
    original = df.sort_values(["year", "month"]).reset_index(drop=True)

    first_period_index = original.index[
        (original["year"] == first["year"]) & (original["month"] == first["month"])
    ][0]

    assert first["lag_1"] == original.loc[first_period_index - 1, "collisions"]
    assert first["lag_12"] == original.loc[first_period_index - 12, "collisions"]