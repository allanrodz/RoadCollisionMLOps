from pathlib import Path

import pandas as pd

from src.config import RAW_DIR
from src.validate_data import standardise_raw_dataframe, validate_standardised


def latest_raw_roa17_file() -> Path:
    files = sorted(RAW_DIR.glob("roa17_raw_*.csv"))
    assert files, "No downloaded ROA17 raw snapshot found. Run: python -m src.ingest"
    return files[-1]


def test_standardise_and_validate_roa17_data():
    raw = pd.read_csv(latest_raw_roa17_file(), encoding="utf-8-sig")
    standardised = standardise_raw_dataframe(raw)
    validated = validate_standardised(standardised)

    assert list(validated.columns) == ["year", "month", "collisions"]
    assert validated["month"].between(1, 12).all()
    assert (validated["collisions"] >= 0).all()
    assert len(validated) == 228
    assert validated["year"].min() == 2005
    assert validated["year"].max() == 2023