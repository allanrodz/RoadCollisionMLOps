from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURES_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_HISTORY_PATH,
    MODEL_INFO_PATH,
    MODEL_NAME,
    MODEL_PATH,
    REGISTER_MODEL,
    VALIDATED_DATA_PATH,
)


@dataclass
class RunResult:
    name: str
    run_id: str
    holdout_mae: float
    holdout_rmse: float
    holdout_r2: float
    cv_mae: float
    model: object
    params: dict


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_training_data(features_path: Path = FEATURES_PATH) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame]:
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}. Run python -m src.features first.")
    df = pd.read_csv(features_path)
    drop_cols = {"collisions", "date", "year", "month"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df["collisions"]
    return X, y, feature_cols, df


def candidate_models() -> dict[str, object]:
    return {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42, n_estimators=120, learning_rate=0.05, max_depth=3),
    }


def evaluate_time_series_cv(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    n_splits = min(5, max(2, len(X) // 12))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []
    for train_idx, test_idx in splitter.split(X):
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fitted.predict(X.iloc[test_idx])
        maes.append(mean_absolute_error(y.iloc[test_idx], pred))
        rmses.append(rmse(y.iloc[test_idx], pred))
    return {"cv_mae": float(np.mean(maes)), "cv_rmse": float(np.mean(rmses)), "cv_splits": n_splits}



def train_and_log(register_model: bool = REGISTER_MODEL) -> RunResult:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X, y, feature_cols, full_df = load_training_data()
    holdout_size = min(12, max(6, int(len(X) * 0.2)))
    X_train, X_holdout = X.iloc[:-holdout_size], X.iloc[-holdout_size:]
    y_train, y_holdout = y.iloc[:-holdout_size], y.iloc[-holdout_size:]

    results: list[RunResult] = []
    sha = git_sha()

    for name, model in candidate_models().items():
        with mlflow.start_run(run_name=name) as run:
            params = model.get_params()
            safe_params = {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool, type(None)))}
            mlflow.log_params(safe_params)
            mlflow.set_tags(
                {
                    "project": "road-collision-forecasting",
                    "git_sha": sha,
                    "target": "All Fatal and Injury Collisions",
                    "validation_strategy": "TimeSeriesSplit plus chronological holdout",
                    "feature_set": ",".join(feature_cols),
                }
            )
            cv = evaluate_time_series_cv(model, X_train, y_train)
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            pred = fitted.predict(X_holdout)
            metrics = {
                **cv,
                "holdout_mae": float(mean_absolute_error(y_holdout, pred)),
                "holdout_rmse": rmse(y_holdout, pred),
                "holdout_r2": float(r2_score(y_holdout, pred)),
                "holdout_size_months": holdout_size,
            }
            mlflow.log_metrics(metrics)
            mlflow.log_text(json.dumps(feature_cols, indent=2), artifact_file="feature_list.json")

            result = RunResult(
                name=name,
                run_id=run.info.run_id,
                holdout_mae=metrics["holdout_mae"],
                holdout_rmse=metrics["holdout_rmse"],
                holdout_r2=metrics["holdout_r2"],
                cv_mae=metrics["cv_mae"],
                model=fitted,
                params=safe_params,
            )
            results.append(result)
            print(f"{name}: {metrics}")

    best = min(results, key=lambda r: r.holdout_mae)

    # Fit the selected model on all available feature rows for the production artifact.
    final_model = clone(candidate_models()[best.name])
    final_model.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    history = pd.read_csv(VALIDATED_DATA_PATH)
    history.to_csv(MODEL_HISTORY_PATH, index=False)

    model_info = {
        "model_name": MODEL_NAME,
        "selected_candidate": best.name,
        "source_run_id": best.run_id,
        "holdout_mae": best.holdout_mae,
        "holdout_rmse": best.holdout_rmse,
        "holdout_r2": best.holdout_r2,
        "cv_mae": best.cv_mae,
        "git_sha": sha,
        "features": feature_cols,
        "promotion_rule": "Selected lowest holdout MAE after TimeSeriesSplit evaluation.",
    }
    MODEL_INFO_PATH.write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    # Log the final production candidate once. This is faster than logging a full MLflow model for every candidate.
    with mlflow.start_run(run_name=f"production_candidate_{best.name}") as final_run:
        mlflow.log_params(best.params)
        mlflow.log_metrics(
            {
                "source_holdout_mae": best.holdout_mae,
                "source_holdout_rmse": best.holdout_rmse,
                "source_holdout_r2": best.holdout_r2,
                "source_cv_mae": best.cv_mae,
            }
        )
        mlflow.set_tags(
            {
                "project": "road-collision-forecasting",
                "git_sha": sha,
                "selected_candidate": best.name,
                "source_run_id": best.run_id,
                "stage": "production_candidate",
            }
        )
        mlflow.log_artifact(MODEL_INFO_PATH.as_posix(), artifact_path="metadata")
        mlflow.log_artifact(MODEL_HISTORY_PATH.as_posix(), artifact_path="metadata")
        input_example = X.head(1)
        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            input_example=input_example,
            pip_requirements=["scikit-learn", "pandas", "numpy"],
        )
        model_info["production_candidate_run_id"] = final_run.info.run_id
        MODEL_INFO_PATH.write_text(json.dumps(model_info, indent=2), encoding="utf-8")

        if register_model or REGISTER_MODEL:
            try:
                registered = mlflow.register_model(f"runs:/{final_run.info.run_id}/model", MODEL_NAME)
                print(f"Registered model {registered.name} version {registered.version}")
            except Exception as exc:
                print(f"MLflow model registration skipped or failed: {exc}")

    print(json.dumps(model_info, indent=2))
    return best

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train candidate forecasting models and log experiments with MLflow.")
    parser.add_argument("--register", action="store_true", help="Register the best model in MLflow Model Registry.")
    parser.add_argument("--no-register", action="store_true", help="Do not register in MLflow Model Registry.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_log(register_model=args.register and not args.no_register)
