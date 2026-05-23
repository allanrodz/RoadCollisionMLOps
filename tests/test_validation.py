import pandas as pd

from src.validate_data import standardise_raw_dataframe, validate_standardised

RAW_ROA17 = "data/raw/ROA17.20260523131010.csv"


def test_standardise_and_validate_roa17_data():
    raw = pd.read_csv(RAW_ROA17, encoding="utf-8-sig")
    standardised = standardise_raw_dataframe(raw)
    validated = validate_standardised(standardised)
    assert list(validated.columns) == ["year", "month", "collisions"]
    assert validated["month"].between(1, 12).all()
    assert (validated["collisions"] >= 0).all()
    assert len(validated) == 228
    assert validated["year"].min() == 2005
    assert validated["year"].max() == 2023
