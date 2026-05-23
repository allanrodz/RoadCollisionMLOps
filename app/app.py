from __future__ import annotations

import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

try:
    from prometheus_flask_exporter import PrometheusMetrics
except Exception:  # pragma: no cover
    PrometheusMetrics = None

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR", "model")
MODEL_PATH = MODEL_DIR / "model.joblib"
FEATURE_LIST_PATH = MODEL_DIR / "feature_list.json"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"
MODEL_HISTORY_PATH = MODEL_DIR / "history.csv"

app = Flask(__name__)
if PrometheusMetrics is not None:
    metrics = PrometheusMetrics(app)
    metrics.info("road_collision_api_info", "Road collision forecasting API", version="1.0.0")


def load_artifacts():
    missing = [p.as_posix() for p in [MODEL_PATH, FEATURE_LIST_PATH, MODEL_HISTORY_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: " + ", ".join(missing) + ". Run python -m src.train before starting the API."
        )
    model = joblib.load(MODEL_PATH)
    feature_list = json.loads(FEATURE_LIST_PATH.read_text(encoding="utf-8"))
    history = pd.read_csv(MODEL_HISTORY_PATH)
    info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8")) if MODEL_INFO_PATH.exists() else {}
    return model, feature_list, history, info


MODEL, FEATURE_LIST, HISTORY, MODEL_INFO = load_artifacts()


def next_month(year: int, month: int) -> tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def make_feature_row(values: list[float], year: int, month: int, feature_list: list[str]) -> pd.DataFrame:
    trend = len(values)
    row = {
        "trend": trend,
        "month_num": month,
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
    }
    for lag in [1, 2, 3, 6, 12]:
        row[f"lag_{lag}"] = values[-lag] if len(values) >= lag else np.nan
    for window in [3, 6, 12]:
        row[f"rolling_mean_{window}"] = float(np.mean(values[-window:])) if len(values) >= window else np.nan

    missing = [col for col in feature_list if col not in row or pd.isna(row[col])]
    if missing:
        raise ValueError(f"Not enough history to build features: {missing}")
    return pd.DataFrame([{col: row[col] for col in feature_list}])


def generate_forecast(horizon: int) -> list[dict]:
    if horizon < 1 or horizon > 24:
        raise ValueError("horizon must be between 1 and 24")
    history = HISTORY.sort_values(["year", "month"]).copy()
    values = history["collisions"].astype(float).tolist()
    year = int(history.iloc[-1]["year"])
    month = int(history.iloc[-1]["month"])

    forecasts = []
    for _ in range(horizon):
        year, month = next_month(year, month)
        X_next = make_feature_row(values, year, month, FEATURE_LIST)
        prediction = float(MODEL.predict(X_next)[0])
        prediction = max(0.0, prediction)
        values.append(prediction)
        forecasts.append(
            {
                "year": year,
                "month": month,
                "period": f"{year}-{month:02d}",
                "forecast_all_fatal_and_injury_collisions": round(prediction, 2),
            }
        )
    return forecasts


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "road-collision-forecast-api"})


@app.get("/forecast")
def forecast():
    horizon = int(request.args.get("horizon", "3"))
    return jsonify(
        {
            "horizon": horizon,
            "target": "All Fatal and Injury Collisions",
            "forecasts": generate_forecast(horizon),
        }
    )


@app.get("/model-info")
def model_info():
    return jsonify(MODEL_INFO)


@app.errorhandler(Exception)
def handle_error(exc):
    return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
