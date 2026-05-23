import pandas as pd

from src.features import build_features
from src.validate_data import standardise_raw_dataframe

RAW_ROA17 = "data/raw/ROA17.20260523131010.csv"


def test_build_features_has_shifted_lags():
    raw = pd.read_csv(RAW_ROA17, encoding="utf-8-sig")
    df = standardise_raw_dataframe(raw)
    features, feature_cols = build_features(df)
    assert len(features) > 0
    assert "lag_1" in feature_cols

    first = features.iloc[0]
    original = df.sort_values(["year", "month"]).reset_index(drop=True)
    first_period_index = original.index[
        (original["year"] == first["year"]) & (original["month"] == first["month"])
    ][0]
    assert first["lag_1"] == original.loc[first_period_index - 1, "collisions"]
    assert first["lag_12"] == original.loc[first_period_index - 12, "collisions"]
