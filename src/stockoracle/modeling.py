from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def _zscore(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    std = np.nanstd(array)
    if std == 0 or np.isnan(std):
        return np.zeros_like(array, dtype=float)
    return (array - np.nanmean(array)) / std


def _rank_ic(frame: pd.DataFrame) -> float:
    correlations: list[float] = []
    for _, day_frame in frame.groupby("date"):
        if len(day_frame) < 4:
            continue
        correlation = day_frame["score"].rank().corr(day_frame["target_return"].rank())
        if pd.notna(correlation):
            correlations.append(float(correlation))
    return float(np.mean(correlations)) if correlations else float("nan")


@dataclass(slots=True)
class ModelOutput:
    ranking: pd.DataFrame
    metrics: dict[str, float]
    feature_importance: pd.DataFrame
    holdout_predictions: pd.DataFrame
    backtest_curve: pd.DataFrame


class EnsembleRanker:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.return_models = self._build_models(seed_offset=0)
        self.move_models = self._build_models(seed_offset=97)

    def _build_models(self, seed_offset: int) -> list[Pipeline]:
        seed = self.random_state + seed_offset
        estimators = [
            RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=seed,
            ),
            ExtraTreesRegressor(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=seed + 1,
            ),
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.75,
                random_state=seed + 2,
            ),
        ]
        return [Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)]) for estimator in estimators]

    def fit(self, frame: pd.DataFrame, feature_columns: list[str]) -> None:
        features = frame[feature_columns]
        target_return = frame["target_return"]
        target_abs_move = frame["target_abs_move"]

        for pipeline in self.return_models:
            pipeline.fit(features, target_return)
        for pipeline in self.move_models:
            pipeline.fit(features, target_abs_move)

    def predict(self, frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        features = frame[feature_columns]
        return_predictions = np.column_stack([pipeline.predict(features) for pipeline in self.return_models])
        move_predictions = np.column_stack([pipeline.predict(features) for pipeline in self.move_models])

        avg_return = return_predictions.mean(axis=1)
        avg_move = move_predictions.mean(axis=1)
        disagreement = return_predictions.std(axis=1)
        score = 0.75 * _zscore(avg_return) + 0.25 * _zscore(avg_move) - 0.10 * _zscore(disagreement)
        confidence = 1 / (1 + disagreement.clip(min=0))

        result = frame[["date", "symbol", "close", "adj_close"]].copy()
        result["predicted_return"] = avg_return
        result["predicted_move"] = avg_move
        result["model_disagreement"] = disagreement
        result["confidence"] = confidence
        result["score"] = score
        result["model_score"] = score
        return result.sort_values("score", ascending=False).reset_index(drop=True)

    def feature_importance(self, feature_columns: list[str]) -> pd.DataFrame:
        importance_frames: list[pd.DataFrame] = []
        for pipeline in self.return_models:
            model = pipeline.named_steps["model"]
            if not hasattr(model, "feature_importances_"):
                continue
            importance_frames.append(
                pd.DataFrame(
                    {
                        "feature": feature_columns,
                        "importance": model.feature_importances_,
                    }
                )
            )

        if not importance_frames:
            return pd.DataFrame(columns=["feature", "importance"])

        merged = pd.concat(importance_frames, ignore_index=True)
        return (
            merged.groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def evaluate_holdout(
    full_frame: pd.DataFrame,
    feature_columns: list[str],
    holdout_days: int,
    top_k: int,
    random_state: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    unique_dates = sorted(full_frame["date"].dropna().unique())
    if len(unique_dates) <= holdout_days + 20:
        return {}, pd.DataFrame(columns=["date", "symbol", "score", "target_return"])

    holdout_dates = unique_dates[-holdout_days:]
    predictions: list[pd.DataFrame] = []
    minimum_train_rows = max(60, len(feature_columns) * 3)

    for test_date in holdout_dates:
        train_frame = full_frame.loc[full_frame["date"] < test_date].dropna(subset=["target_return", "target_abs_move"])
        active_feature_columns = [column for column in feature_columns if train_frame[column].notna().any()]
        test_frame = full_frame.loc[full_frame["date"] == test_date]
        test_frame = test_frame.dropna(subset=active_feature_columns, how="all") if active_feature_columns else pd.DataFrame()
        if len(train_frame) < minimum_train_rows or test_frame.empty or not active_feature_columns:
            continue

        ranker = EnsembleRanker(random_state=random_state)
        ranker.fit(train_frame, active_feature_columns)
        scored = ranker.predict(test_frame, active_feature_columns)
        merged = scored.merge(
            full_frame.loc[full_frame["date"] == test_date, ["date", "symbol", "target_return"]],
            on=["date", "symbol"],
            how="left",
        )
        predictions.append(merged)

    if not predictions:
        return {}, pd.DataFrame(columns=["date", "symbol", "score", "target_return"])

    holdout_predictions = pd.concat(predictions, ignore_index=True)
    top_picks = holdout_predictions.groupby("date", group_keys=False).head(top_k)

    metrics = {
        "avg_top_k_return": float(top_picks["target_return"].mean()),
        "top_k_hit_rate": float((top_picks["target_return"] > 0).mean()),
        "avg_rank_ic": _rank_ic(holdout_predictions),
        "holdout_days_evaluated": float(holdout_predictions["date"].nunique()),
    }
    return metrics, holdout_predictions