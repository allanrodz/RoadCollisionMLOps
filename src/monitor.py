from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import requests


def check_api(base_url: str) -> dict:
    base_url = base_url.rstrip("/")
    health = requests.get(f"{base_url}/health", timeout=10)
    forecast = requests.get(f"{base_url}/forecast?horizon=3", timeout=10)
    model_info = requests.get(f"{base_url}/model-info", timeout=10)

    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "health_status_code": health.status_code,
        "forecast_status_code": forecast.status_code,
        "model_info_status_code": model_info.status_code,
        "healthy": health.ok and forecast.ok and model_info.ok,
        "health": health.json() if health.headers.get("content-type", "").startswith("application/json") else health.text,
        "model_info": model_info.json() if model_info.headers.get("content-type", "").startswith("application/json") else model_info.text,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous monitoring smoke check for the deployed API.")
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://localhost:5000"))
    args = parser.parse_args()
    result = check_api(args.base_url)
    print(json.dumps(result, indent=2))
    return 0 if result["healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())
