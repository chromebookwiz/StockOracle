from __future__ import annotations

from dataclasses import dataclass
from inspect import signature

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
        score_column = "opportunity_score" if "opportunity_score" in day_frame.columns else "score"
        target_column = "realized_return" if "realized_return" in day_frame.columns else "target_return"
        correlation = day_frame[score_column].rank().corr(day_frame[target_column].rank())
        if pd.notna(correlation):
            correlations.append(float(correlation))
    return float(np.mean(correlations)) if correlations else float("nan")


def build_recency_weights(frame: pd.DataFrame, half_life_sessions: float = 15.0) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=float)

    ordered_dates = pd.Index(sorted(pd.to_datetime(frame["date"]).dropna().unique()))
    if ordered_dates.empty:
        return np.ones(len(frame), dtype=float)

    date_rank = pd.Series(np.arange(len(ordered_dates), dtype=float), index=ordered_dates)
    mapped_ranks = pd.to_datetime(frame["date"]).map(date_rank)
    latest_rank = float(date_rank.iloc[-1])
    ages = (latest_rank - mapped_ranks).fillna(latest_rank).to_numpy(dtype=float)
    weights = np.power(0.5, ages / max(half_life_sessions, 1.0))
    return np.clip(weights, 0.05, 1.0)


@dataclass(slots=True)
class ModelOutput:
    ranking: pd.DataFrame
    metrics: dict[str, float]
    feature_importance: pd.DataFrame
    holdout_predictions: pd.DataFrame
    backtest_curve: pd.DataFrame
    execution_plan: pd.DataFrame


class EnsembleRanker:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.return_models = self._build_models(seed_offset=0)
        self.move_models = self._build_models(seed_offset=97)
        self.direction_models = self._build_classifiers(seed_offset=193)
        self.lower_return_models = self._build_quantile_models(seed_offset=257, alpha=0.2)
        self.upper_return_models = self._build_quantile_models(seed_offset=353, alpha=0.8)

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
            HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_depth=6,
                max_iter=300,
                min_samples_leaf=12,
                l2_regularization=0.05,
                random_state=seed + 3,
            ),
            MLPRegressor(
                hidden_layer_sizes=(96, 48),
                activation="relu",
                solver="adam",
                alpha=0.0005,
                batch_size=10_000,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.15,
                random_state=seed + 4,
            ),
        ]
        pipelines: list[Pipeline] = []
        for estimator in estimators:
            steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
            if isinstance(estimator, MLPRegressor):
                steps.append(("scaler", StandardScaler()))
            steps.append(("model", estimator))
            pipelines.append(Pipeline(steps))
        return pipelines

    def _build_classifiers(self, seed_offset: int) -> list[Pipeline]:
        seed = self.random_state + seed_offset
        estimators = [
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=seed,
            ),
            ExtraTreesClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=seed + 1,
            ),
            HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_depth=6,
                max_iter=250,
                min_samples_leaf=12,
                l2_regularization=0.05,
                random_state=seed + 2,
            ),
            MLPClassifier(
                hidden_layer_sizes=(96, 48),
                activation="relu",
                solver="adam",
                alpha=0.0005,
                batch_size=10_000,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.15,
                random_state=seed + 3,
            ),
        ]
        pipelines: list[Pipeline] = []
        for estimator in estimators:
            steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
            if isinstance(estimator, MLPClassifier):
                steps.append(("scaler", StandardScaler()))
            steps.append(("model", estimator))
            pipelines.append(Pipeline(steps))
        return pipelines

    def _build_quantile_models(self, seed_offset: int, alpha: float) -> list[Pipeline]:
        seed = self.random_state + seed_offset
        estimators = [
            GradientBoostingRegressor(
                loss="quantile",
                alpha=alpha,
                n_estimators=220,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.75,
                random_state=seed,
            ),
            GradientBoostingRegressor(
                loss="quantile",
                alpha=alpha,
                n_estimators=320,
                learning_rate=0.02,
                max_depth=2,
                subsample=0.8,
                random_state=seed + 1,
            ),
        ]
        return [Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)]) for estimator in estimators]

    def _fit_pipeline(
        self,
        pipeline: Pipeline,
        features: pd.DataFrame,
        target: pd.Series,
        sample_weight: np.ndarray | None,
    ) -> None:
        estimator = pipeline.named_steps["model"]
        if isinstance(estimator, (MLPRegressor, MLPClassifier)):
            validation_fraction = float(getattr(estimator, "validation_fraction", 0.0) or 0.0)
            effective_batch_size = len(features)
            if getattr(estimator, "early_stopping", False):
                effective_batch_size = max(1, int(len(features) * (1 - validation_fraction)))
            estimator.set_params(batch_size=effective_batch_size)

        if sample_weight is None:
            pipeline.fit(features, target)
            return

        estimator_signature = signature(estimator.fit)
        if "sample_weight" in estimator_signature.parameters:
            pipeline.fit(features, target, model__sample_weight=sample_weight)
            return

        pipeline.fit(features, target)

    def fit(self, frame: pd.DataFrame, feature_columns: list[str], sample_weight: np.ndarray | None = None) -> None:
        features = frame[feature_columns]
        target_return = frame["target_return"]
        target_abs_move = frame["target_abs_move"]
        target_direction = (target_return > 0).astype(int)

        for pipeline in self.return_models:
            self._fit_pipeline(pipeline, features, target_return, sample_weight)
        for pipeline in self.move_models:
            self._fit_pipeline(pipeline, features, target_abs_move, sample_weight)
        for pipeline in self.direction_models:
            self._fit_pipeline(pipeline, features, target_direction, sample_weight)
        for pipeline in self.lower_return_models:
            self._fit_pipeline(pipeline, features, target_return, sample_weight)
        for pipeline in self.upper_return_models:
            self._fit_pipeline(pipeline, features, target_return, sample_weight)

    def predict(self, frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        features = frame[feature_columns]
        return_predictions = np.column_stack([pipeline.predict(features) for pipeline in self.return_models])
        move_predictions = np.column_stack([pipeline.predict(features) for pipeline in self.move_models])
        direction_probabilities = np.column_stack([pipeline.predict_proba(features)[:, 1] for pipeline in self.direction_models])
        lower_predictions = np.column_stack([pipeline.predict(features) for pipeline in self.lower_return_models])
        upper_predictions = np.column_stack([pipeline.predict(features) for pipeline in self.upper_return_models])

        avg_return = return_predictions.mean(axis=1)
        avg_move = move_predictions.mean(axis=1)
        disagreement = return_predictions.std(axis=1)
        probability_up = direction_probabilities.mean(axis=1)
        lower_return = lower_predictions.mean(axis=1)
        upper_return = upper_predictions.mean(axis=1)
        interval_width = np.clip(upper_return - lower_return, a_min=0.0, a_max=None)
        risk_adjusted_return = avg_return / (interval_width + 1e-4)
        directional_edge = probability_up - 0.5
        score = (
            0.55 * _zscore(avg_return)
            + 0.20 * _zscore(directional_edge)
            + 0.10 * (_zscore(avg_move) * np.sign(avg_return))
            + 0.15 * _zscore(risk_adjusted_return)
            - 0.10 * _zscore(disagreement)
            - 0.10 * _zscore(interval_width)
        )
        directional_confidence = np.maximum(probability_up, 1 - probability_up)
        disagreement_penalty = 1 / (1 + disagreement.clip(min=0) * 50)
        interval_penalty = 1 / (1 + interval_width * 25)
        confidence = np.clip(0.5 * directional_confidence + 0.25 * disagreement_penalty + 0.25 * interval_penalty, 0.0, 1.0)
        opportunity_score = np.abs(score) + 0.15 * _zscore(avg_move) + 0.10 * _zscore(confidence)
        signal_side = np.where(score >= 0, "long", "short")

        result = frame[["date", "symbol", "close", "adj_close"]].copy()
        result["predicted_return"] = avg_return
        result["predicted_return_lower"] = lower_return
        result["predicted_return_upper"] = upper_return
        result["prediction_interval"] = interval_width
        result["predicted_move"] = avg_move
        result["probability_up"] = probability_up
        result["risk_adjusted_return"] = risk_adjusted_return
        result["model_disagreement"] = disagreement
        result["confidence"] = confidence
        result["score"] = score
        result["model_score"] = score
        result["signal_side"] = signal_side
        result["opportunity_score"] = opportunity_score
        return result.sort_values(["opportunity_score", "confidence"], ascending=[False, False]).reset_index(drop=True)

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
                        "source": type(model).__name__,
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
        ranker.fit(train_frame, active_feature_columns, sample_weight=build_recency_weights(train_frame))
        scored = ranker.predict(test_frame, active_feature_columns)
        merged = scored.merge(
            full_frame.loc[full_frame["date"] == test_date, ["date", "symbol", "target_return"]],
            on=["date", "symbol"],
            how="left",
        )
        merged["realized_return"] = np.where(merged["signal_side"] == "short", -merged["target_return"], merged["target_return"])
        merged["direction_correct"] = np.where(merged["signal_side"] == "short", merged["target_return"] < 0, merged["target_return"] > 0)
        predictions.append(merged)

    if not predictions:
        return {}, pd.DataFrame(columns=["date", "symbol", "score", "target_return"])

    holdout_predictions = pd.concat(predictions, ignore_index=True)
    top_picks = holdout_predictions.sort_values(["date", "opportunity_score"], ascending=[True, False]).groupby("date", group_keys=False).head(top_k)

    metrics = {
        "avg_top_k_return": float(top_picks["realized_return"].mean()),
        "top_k_hit_rate": float((top_picks["realized_return"] > 0).mean()),
        "avg_rank_ic": _rank_ic(holdout_predictions),
        "directional_accuracy": float(top_picks["direction_correct"].mean()),
        "avg_prediction_interval": float(top_picks["prediction_interval"].mean()),
        "holdout_days_evaluated": float(holdout_predictions["date"].nunique()),
    }
    return metrics, holdout_predictions