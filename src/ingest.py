from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import BUNDLED_ROA17_PATH, DATA_URL, METADATA_DIR, RAW_DIR, REFERENCE_DATASETS


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_csv_for_metadata(path: Path) -> pd.DataFrame:
    # utf-8-sig removes the BOM that appears in some CSO/StatBank CSV exports.
    return pd.read_csv(path, encoding="utf-8-sig")


def ingest(data_url: str | None = None, use_bundled: bool = False) -> Path:
    """Acquiring the ROA17 monthly collision data.

    ROA18, ROA19, ROA20 and ROA27 are retained in data/raw as reference datasets, but the model uses
    ROA17 because it has the required monthly breakdown for time-series forecasting.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = RAW_DIR / f"roa17_raw_{timestamp}.csv"
    source = data_url or DATA_URL

    if source and not use_bundled:
        # pandas supports http(s) URLs and local file paths.
        df = pd.read_csv(source, encoding="utf-8-sig")
        df.to_csv(raw_path, index=False)
        source_used = source
        acquisition_mode = "url_or_path"
    else:
        if not BUNDLED_ROA17_PATH.exists():
            raise FileNotFoundError(
                f"Bundled ROA17 file not found: {BUNDLED_ROA17_PATH}. "
                "Add the ROA17 CSV to data/raw or set DATA_URL."
            )
        shutil.copyfile(BUNDLED_ROA17_PATH, raw_path)
        source_used = BUNDLED_ROA17_PATH.as_posix()
        acquisition_mode = "bundled_cso_csv"

    df_check = load_csv_for_metadata(raw_path)
    target_rows = int(
        df_check.get("Statistic Label", pd.Series(dtype=str))
        .astype(str)
        .str.contains("All Fatal and Injury Collisions", case=False, regex=False, na=False)
        .sum()
    )

    metadata = {
        "stage": "data_acquisition",
        "dataset_used_for_model": "ROA17 Traffic Collisions and Casualties by Month of Year",
        "reference_datasets_available": {k: v.as_posix() for k, v in REFERENCE_DATASETS.items() if v.exists()},
        "source": source_used,
        "acquisition_mode": acquisition_mode,
        "created_utc": timestamp,
        "raw_path": raw_path.as_posix(),
        "sha256": sha256_file(raw_path),
        "row_count": int(len(df_check)),
        "target_row_count_before_filtering": target_rows,
        "columns": list(df_check.columns),
        "git_sha": git_sha(),
        "notes": (
            "This manifest provides data lineage for the raw ROA17 dataset. "
            "Validation later filters the target to All Fatal and Injury Collisions and removes All months aggregates."
        ),
    }
    manifest_path = METADATA_DIR / f"lineage_raw_{timestamp}.json"
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return raw_path


def latest_raw_file() -> Path:
    # Prefer versioned ingestion outputs. If none exist, fall back to the bundled ROA17 file.
    files = sorted(RAW_DIR.glob("roa17_raw_*.csv"))
    if files:
        return files[-1]
    if BUNDLED_ROA17_PATH.exists():
        return BUNDLED_ROA17_PATH
    raise FileNotFoundError("No ROA17 raw CSV found. Run python -m src.ingest or add ROA17 to data/raw.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire the ROA17 road collision dataset and write a lineage manifest.")
    parser.add_argument("--data-url", default=os.getenv("DATA_URL", ""), help="CSV URL or local CSV path.")
    parser.add_argument(
        "--use-bundled",
        action="store_true",
        help="Force the bundled data/raw/ROA17...csv file even if DATA_URL is set.",
    )
    # Backwards-compatible flag from the first starter kit. It now means use the bundled real ROA17 CSV.
    parser.add_argument("--use-sample", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest(data_url=args.data_url or None, use_bundled=args.use_bundled or args.use_sample)
