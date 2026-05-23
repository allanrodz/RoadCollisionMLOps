from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import pandera.pandas as pa
except Exception: 
    import pandera as pa

from src.config import ALLOW_MISSING_MONTHS, METADATA_DIR, TARGET_NAME, VALIDATED_DATA_PATH
from src.ingest import latest_raw_file, sha256_file

MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def snake(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())).strip("_")


def parse_month(value) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in {"all", "all_months", "all months", "total"}:
        return None
    if text.isdigit():
        number = int(text)
        return number if 1 <= number <= 12 else None
    lowered = text.lower()
    if lowered in MONTH_MAP:
        return MONTH_MAP[lowered]
    # Handles formats such as 2020M01 or 2020-01.
    match = re.search(r"(?:m|[-/])\s*(0?[1-9]|1[0-2])\b", lowered)
    if match:
        return int(match.group(1))
    return None


def resolve_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    columns = {snake(c): c for c in df.columns}

    year_candidates = ["year", "yyyy", "time", "period"]
    month_candidates = ["month", "months", "mmm", "period_month",
        "month_of_year",
        "c01885v02316"]
    value_candidates = [
        "value",
        "statistic_value",
        "all_fatal_and_injury_collisions",
        "collisions",
        "count",
    ]

    year_col = next((columns[c] for c in year_candidates if c in columns), None)
    month_col = next((columns[c] for c in month_candidates if c in columns), None)
    value_col = next((columns[c] for c in value_candidates if c in columns), None)

    if not year_col:
        # Some StatBank exports include a time column such as 2019M01. Try to infer year/month from it.
        for candidate in df.columns:
            if df[candidate].astype(str).str.contains(r"\d{4}M\d{2}", regex=True, na=False).any():
                year_col = candidate
                month_col = candidate
                break

    missing = [name for name, col in [("year", year_col), ("month", month_col), ("value", value_col)] if col is None]
    if missing:
        raise ValueError(f"Could not resolve required columns: {missing}. Available columns: {list(df.columns)}")
    return year_col, month_col, value_col


def standardise_raw_dataframe(df: pd.DataFrame, target_name: str = TARGET_NAME) -> pd.DataFrame:
    df = df.copy()
    label_col = None
    for preferred in ["statistic_label", "label", "statistic"]:
        for c in df.columns:
            if snake(c) == preferred:
                label_col = c
                break
        if label_col is not None:
            break

    if label_col is not None:
        mask = df[label_col].astype(str).str.contains(target_name, case=False, regex=False, na=False)
        if mask.any():
            df = df.loc[mask].copy()
        else:
            available = sorted(df[label_col].dropna().astype(str).unique().tolist())[:20]
            raise ValueError(
                f"Target '{target_name}' was not found in column '{label_col}'. "
                f"Available labels include: {available}"
            )

    year_col, month_col, value_col = resolve_columns(df)

    if year_col == month_col:
        period = df[year_col].astype(str)
        year = period.str.extract(r"(\d{4})")[0]
        month = period.str.extract(r"M(\d{2})")[0]
    else:
        year = df[year_col]
        month = df[month_col]

    out = pd.DataFrame(
        {
            "year": pd.to_numeric(year, errors="coerce"),
            "month": month.map(parse_month) if hasattr(month, "map") else pd.Series(month).map(parse_month),
            "collisions": pd.to_numeric(df[value_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["year", "month", "collisions"]).copy()
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out["collisions"] = out["collisions"].astype(float)
    out = out.groupby(["year", "month"], as_index=False)["collisions"].sum()
    out = out.sort_values(["year", "month"]).reset_index(drop=True)
    return out


def validate_standardised(df: pd.DataFrame, allow_missing_months: bool = ALLOW_MISSING_MONTHS) -> pd.DataFrame:
    schema = pa.DataFrameSchema(
        {
            "year": pa.Column(int, pa.Check.in_range(1900, 2100), nullable=False),
            "month": pa.Column(int, pa.Check.in_range(1, 12), nullable=False),
            "collisions": pa.Column(float, pa.Check.ge(0), nullable=False),
        },
        checks=[
            pa.Check(lambda d: ~d.duplicated(["year", "month"]).any(), error="Duplicate year/month rows found"),
        ],
    )
    validated = schema.validate(df, lazy=True)

    dates = pd.to_datetime(dict(year=validated["year"], month=validated["month"], day=1))
    expected = pd.date_range(dates.min(), dates.max(), freq="MS")
    missing = sorted(set(expected) - set(dates))
    if missing and not allow_missing_months:
        missing_text = ", ".join(d.strftime("%Y-%m") for d in missing[:12])
        raise ValueError(f"Missing monthly observations detected: {missing_text}")
    return validated


def validate_file(raw_path: Path | None = None) -> Path:
    raw_path = raw_path or latest_raw_file()
    raw_df = pd.read_csv(raw_path, encoding="utf-8-sig")
    standardised = standardise_raw_dataframe(raw_df)
    validated = validate_standardised(standardised)
    VALIDATED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(VALIDATED_DATA_PATH, index=False)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "stage": "data_validation",
        "created_utc": timestamp,
        "input_raw_path": raw_path.as_posix(),
        "input_sha256": sha256_file(raw_path),
        "output_path": VALIDATED_DATA_PATH.as_posix(),
        "row_count": int(len(validated)),
        "min_period": pd.to_datetime(dict(year=validated["year"], month=validated["month"], day=1)).min().strftime("%Y-%m"),
        "max_period": pd.to_datetime(dict(year=validated["year"], month=validated["month"], day=1)).max().strftime("%Y-%m"),
        "validation_rules": [
            "year is integer between 1900 and 2100",
            "month is integer between 1 and 12",
            "collisions is non-negative numeric",
            "no duplicate year/month rows",
            "no missing months unless ALLOW_MISSING_MONTHS=true",
            "Statistic Label is filtered to the configured TARGET_NAME",
            "Month of Year / All months aggregate rows are excluded",
        ],
    }
    manifest_path = METADATA_DIR / f"lineage_validation_{timestamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return VALIDATED_DATA_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw collision data and create the standard monthly dataset.")
    parser.add_argument("--raw-path", default="", help="Optional raw CSV path. Defaults to latest data/raw CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_file(Path(args.raw_path) if args.raw_path else None)
