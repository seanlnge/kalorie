from __future__ import annotations

import math
import re
from bisect import bisect_left
from collections import defaultdict
from itertools import product

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from kalorie.domain.models import FeatureVector, Prediction
from kalorie.ml.datasets import HistoricalTrainingExample
from kalorie.ml.priors import (
    HierarchicalPriorModel,
    fit_hierarchical_prior_model,
    market_prior_features,
    phrase_category,
)


def _clamp_probability(value: float) -> float:
    return min(0.99, max(0.01, round(value, 6)))


class CompanyFineTune(BaseModel):
    model_config = ConfigDict(extra="forbid")

    training_rows: int = Field(ge=1)
    intercept: float
    coefficients: dict[str, float]


class IsotonicCalibrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_thresholds: list[float]
    y_thresholds: list[float]

    def calibrate(self, probability: float) -> float:
        if not self.x_thresholds or not self.y_thresholds:
            return probability
        if len(self.x_thresholds) == 1:
            return _clamp_probability(self.y_thresholds[0])
        index = bisect_left(self.x_thresholds, probability)
        if index <= 0:
            return _clamp_probability(self.y_thresholds[0])
        if index >= len(self.x_thresholds):
            return _clamp_probability(self.y_thresholds[-1])
        left_x = self.x_thresholds[index - 1]
        right_x = self.x_thresholds[index]
        left_y = self.y_thresholds[index - 1]
        right_y = self.y_thresholds[index]
        if right_x <= left_x:
            return _clamp_probability(right_y)
        weight = (probability - left_x) / (right_x - left_x)
        calibrated = left_y + weight * (right_y - left_y)
        return _clamp_probability(calibrated)


class TemperatureCalibrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(gt=0)

    def calibrate(self, probability: float) -> float:
        clipped = min(0.999999, max(0.000001, probability))
        logit = math.log(clipped / (1.0 - clipped)) / self.temperature
        return _clamp_probability(1.0 / (1.0 + math.exp(-logit)))


class MentionModelArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_version: str
    feature_columns: list[str]
    base_intercept: float
    base_coefficients: dict[str, float]
    blend_weight: float = Field(ge=0.0, le=1.0)
    min_company_rows: int = Field(ge=2)
    regularization_c: float = Field(default=1.0, gt=0)
    class_weight_balanced: bool = False
    include_target_indicator: bool = False
    temperature_calibration: TemperatureCalibrationModel | None = None
    isotonic_calibration: IsotonicCalibrationModel | None = None
    company_overrides: dict[str, CompanyFineTune] = Field(default_factory=dict)
    category_overrides: dict[str, CompanyFineTune] = Field(default_factory=dict)
    category_blend_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    min_category_rows: int = Field(default=8, ge=2)
    include_market_features: bool = False
    include_hierarchical_priors: bool = True
    hierarchical_priors: HierarchicalPriorModel | None = None


class CompanyRetrainedModelArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_version: str
    company_symbol: str
    training_rows: int = Field(ge=2)
    feature_columns: list[str]
    intercept: float
    coefficients: dict[str, float]
    regularization_c: float = Field(default=1.0, gt=0)
    class_weight_balanced: bool = False
    include_target_indicator: bool = False
    recency_ema_half_life_quarters: float | None = None
    source_global_model_version: str | None = None


class Model1OptimizationTrial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regularization_c: float
    min_company_rows: int
    blend_weight: float
    class_weight_balanced: bool
    include_target_indicator: bool
    isotonic_calibration: bool
    temperature_calibration: bool = False
    holdout_brier_score: float
    holdout_log_loss: float
    company_override_count: int = Field(ge=0)


class Model1OptimizationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split_strategy: str
    train_sample_count: int = Field(ge=0)
    holdout_sample_count: int = Field(ge=0)
    best_trial: Model1OptimizationTrial
    holdout_brier_score: float = Field(ge=0)
    holdout_log_loss: float = Field(ge=0)
    trials: list[Model1OptimizationTrial]


def train_model1(
    examples: list[HistoricalTrainingExample],
    *,
    min_company_rows: int = 25,
    blend_weight: float = 0.35,
    regularization_c: float = 1.0,
    class_weight_balanced: bool = False,
    include_target_indicator: bool = False,
    min_category_rows: int = 8,
    category_blend_weight: float = 0.25,
    include_market_features: bool = False,
    include_hierarchical_priors: bool = True,
    enable_temperature_calibration: bool = False,
    enable_isotonic_calibration: bool = False,
    calibration_fraction: float = 0.2,
    model_version: str = "mention-base-company-v1",
) -> MentionModelArtifact:
    if len(examples) < 4:
        raise ValueError("at least 4 examples are required")
    hierarchical_priors = (
        fit_hierarchical_prior_model(examples)
        if include_hierarchical_priors
        else None
    )
    frame = _to_frame(
        examples,
        include_target_indicator=include_target_indicator,
        include_market_features=include_market_features,
        hierarchical_priors=hierarchical_priors,
    )
    feature_columns = sorted(
        column
        for column in frame.columns
        if column not in {"label", "company_symbol", "phrase_category"}
    )
    base_model = _fit_logistic(
        frame[feature_columns],
        frame["label"],
        regularization_c=regularization_c,
        class_weight_balanced=class_weight_balanced,
    )

    company_overrides: dict[str, CompanyFineTune] = {}
    for company_symbol, company_frame in frame.groupby("company_symbol"):
        if len(company_frame) < min_company_rows:
            continue
        if len(set(company_frame["label"].tolist())) < 2:
            continue
        company_model = _fit_logistic(
            company_frame[feature_columns],
            company_frame["label"],
            regularization_c=regularization_c,
            class_weight_balanced=class_weight_balanced,
        )
        company_overrides[company_symbol] = CompanyFineTune(
            training_rows=len(company_frame),
            intercept=float(company_model.intercept_[0]),
            coefficients={
                column: float(weight)
                for column, weight in zip(feature_columns, company_model.coef_[0], strict=True)
            },
        )

    category_overrides: dict[str, CompanyFineTune] = {}
    for category, category_frame in frame.groupby("phrase_category"):
        if len(category_frame) < min_category_rows:
            continue
        if len(set(category_frame["label"].tolist())) < 2:
            continue
        category_model = _fit_logistic(
            category_frame[feature_columns],
            category_frame["label"],
            regularization_c=regularization_c,
            class_weight_balanced=class_weight_balanced,
        )
        category_overrides[str(category)] = CompanyFineTune(
            training_rows=len(category_frame),
            intercept=float(category_model.intercept_[0]),
            coefficients={
                column: float(weight)
                for column, weight in zip(feature_columns, category_model.coef_[0], strict=True)
            },
        )

    temperature_calibration = None
    if enable_temperature_calibration:
        temperature_calibration = _fit_temperature_calibration(
            examples=examples,
            min_company_rows=min_company_rows,
            blend_weight=blend_weight,
            regularization_c=regularization_c,
            class_weight_balanced=class_weight_balanced,
            include_target_indicator=include_target_indicator,
            calibration_fraction=calibration_fraction,
        )

    isotonic_calibration = None
    if enable_isotonic_calibration:
        isotonic_calibration = _fit_isotonic_calibration(
            examples=examples,
            min_company_rows=min_company_rows,
            blend_weight=blend_weight,
            regularization_c=regularization_c,
            class_weight_balanced=class_weight_balanced,
            include_target_indicator=include_target_indicator,
            calibration_fraction=calibration_fraction,
        )

    return MentionModelArtifact(
        model_version=model_version,
        feature_columns=feature_columns,
        base_intercept=float(base_model.intercept_[0]),
        base_coefficients={
            column: float(weight)
            for column, weight in zip(feature_columns, base_model.coef_[0], strict=True)
        },
        blend_weight=blend_weight,
        min_company_rows=min_company_rows,
        regularization_c=regularization_c,
        class_weight_balanced=class_weight_balanced,
        include_target_indicator=include_target_indicator,
        min_category_rows=min_category_rows,
        category_blend_weight=category_blend_weight,
        include_market_features=include_market_features,
        include_hierarchical_priors=include_hierarchical_priors,
        temperature_calibration=temperature_calibration,
        isotonic_calibration=isotonic_calibration,
        company_overrides=company_overrides,
        category_overrides=category_overrides,
        hierarchical_priors=hierarchical_priors,
    )


def train_company_model1(
    examples: list[HistoricalTrainingExample],
    *,
    company_symbol: str,
    min_company_rows: int = 20,
    regularization_c: float = 1.0,
    class_weight_balanced: bool = False,
    include_target_indicator: bool = False,
    recency_ema_half_life_quarters: float | None = None,
    source_global_model_version: str | None = None,
) -> CompanyRetrainedModelArtifact:
    if min_company_rows < 2:
        raise ValueError("min_company_rows must be at least 2")
    symbol = company_symbol.upper()
    company_examples = [example for example in examples if example.company_symbol == symbol]
    if len(company_examples) < min_company_rows:
        raise ValueError(
            f"not enough rows for {symbol}: {len(company_examples)} < {min_company_rows}"
        )
    frame = _to_frame(company_examples, include_target_indicator=include_target_indicator)
    labels = frame["label"].tolist()
    if len(set(labels)) < 2:
        raise ValueError(f"company examples for {symbol} must include both label classes")
    if recency_ema_half_life_quarters is not None and recency_ema_half_life_quarters <= 0:
        raise ValueError("recency_ema_half_life_quarters must be greater than 0")
    feature_columns = sorted(
        column
        for column in frame.columns
        if column not in {"label", "company_symbol", "phrase_category"}
    )
    sample_weight = (
        _ema_recency_weights(
            company_examples,
            half_life_quarters=recency_ema_half_life_quarters,
        )
        if recency_ema_half_life_quarters is not None
        else None
    )
    model = _fit_logistic(
        frame[feature_columns],
        frame["label"],
        regularization_c=regularization_c,
        class_weight_balanced=class_weight_balanced,
        sample_weight=sample_weight,
    )
    return CompanyRetrainedModelArtifact(
        model_version="mention-company-retrained-v1",
        company_symbol=symbol,
        training_rows=len(company_examples),
        feature_columns=feature_columns,
        intercept=float(model.intercept_[0]),
        coefficients={
            column: float(weight)
            for column, weight in zip(feature_columns, model.coef_[0], strict=True)
        },
        regularization_c=regularization_c,
        class_weight_balanced=class_weight_balanced,
        include_target_indicator=include_target_indicator,
        recency_ema_half_life_quarters=recency_ema_half_life_quarters,
        source_global_model_version=source_global_model_version,
    )


def predict_model1(
    artifact: MentionModelArtifact,
    *,
    company_symbol: str,
    feature_vector: FeatureVector,
    market_probability: float | None = None,
) -> Prediction:
    raw_features = _prediction_features(
        artifact=artifact,
        company_symbol=company_symbol,
        target_phrase=feature_vector.target_phrase,
        features=feature_vector.features,
        market_probability=market_probability,
    )
    model_features, model_feature_columns, used_unseen_target_fallback = _prediction_context(
        target_phrase=feature_vector.target_phrase,
        raw_features=raw_features,
        feature_columns=artifact.feature_columns,
        include_target_indicator=artifact.include_target_indicator,
    )
    base_probability = _predict_linear_probability(
        intercept=artifact.base_intercept,
        coefficients=artifact.base_coefficients,
        feature_columns=model_feature_columns,
        features=model_features,
    )
    reasons = ["base_logistic"]
    if artifact.include_market_features:
        reasons.append("market_prior")
    if artifact.hierarchical_priors is not None:
        reasons.append("hierarchical_priors")
    blended_probability = base_probability

    category = phrase_category(feature_vector.target_phrase)
    category_override = artifact.category_overrides.get(category)
    if category_override is not None:
        category_probability = _predict_linear_probability(
            intercept=category_override.intercept,
            coefficients=category_override.coefficients,
            feature_columns=model_feature_columns,
            features=model_features,
        )
        blended_probability = (
            (1.0 - artifact.category_blend_weight) * blended_probability
            + artifact.category_blend_weight * category_probability
        )
        reasons.append("category_finetune")

    company_override = artifact.company_overrides.get(company_symbol.upper())
    if company_override is not None:
        company_probability = _predict_linear_probability(
            intercept=company_override.intercept,
            coefficients=company_override.coefficients,
            feature_columns=model_feature_columns,
            features=model_features,
        )
        blended_probability = (
            (1.0 - artifact.blend_weight) * base_probability
            + artifact.blend_weight * company_probability
        )
        reasons.append("company_finetune")

    if used_unseen_target_fallback:
        reasons.append("unseen_target_fallback")

    if artifact.temperature_calibration is not None:
        blended_probability = artifact.temperature_calibration.calibrate(blended_probability)
        reasons.append("temperature_calibration")

    if artifact.isotonic_calibration is not None:
        blended_probability = artifact.isotonic_calibration.calibrate(blended_probability)
        reasons.append("isotonic_calibration")

    return Prediction(
        target_phrase=feature_vector.target_phrase,
        model_version=artifact.model_version,
        probability=_clamp_probability(blended_probability),
        reasons=reasons,
    )


def predict_company_model1(
    artifact: CompanyRetrainedModelArtifact,
    *,
    feature_vector: FeatureVector,
) -> Prediction:
    model_features, model_feature_columns, used_unseen_target_fallback = _prediction_context(
        target_phrase=feature_vector.target_phrase,
        raw_features=feature_vector.features,
        feature_columns=artifact.feature_columns,
        include_target_indicator=artifact.include_target_indicator,
    )
    probability = _predict_linear_probability(
        intercept=artifact.intercept,
        coefficients=artifact.coefficients,
        feature_columns=model_feature_columns,
        features=model_features,
    )
    reasons = ["company_retrained_logistic", artifact.company_symbol]
    if used_unseen_target_fallback:
        reasons.append("unseen_target_fallback")
    return Prediction(
        target_phrase=feature_vector.target_phrase,
        model_version=artifact.model_version,
        probability=_clamp_probability(probability),
        reasons=reasons,
    )


def optimize_model1_for_brier(
    examples: list[HistoricalTrainingExample],
    *,
    test_fraction: float = 0.25,
    regularization_values: list[float] | None = None,
    min_company_rows_values: list[int] | None = None,
    blend_weight_values: list[float] | None = None,
    class_weight_balanced_values: list[bool] | None = None,
    include_target_indicator_values: list[bool] | None = None,
    enable_temperature_calibration: bool = False,
    enable_isotonic_calibration: bool = True,
    calibration_fraction: float = 0.2,
    model_version: str = "mention-base-company-v2",
) -> tuple[MentionModelArtifact, Model1OptimizationReport]:
    if len(examples) < 12:
        raise ValueError("at least 12 examples are required for model1 optimization")
    train_examples, test_examples = _time_split_examples(examples, test_fraction=test_fraction)
    regularization_grid = regularization_values or [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
    min_rows_grid = min_company_rows_values or [8, 12, 20, 25, 35]
    blend_grid = blend_weight_values or [0.15, 0.25, 0.35, 0.5, 0.7]
    class_weight_grid = class_weight_balanced_values or [False, True]
    target_indicator_grid = include_target_indicator_values or [False, True]

    trials: list[Model1OptimizationTrial] = []
    for (
        regularization_c,
        min_company_rows,
        blend_weight,
        class_weight_balanced,
        include_target_indicator,
    ) in product(
        regularization_grid,
        min_rows_grid,
        blend_grid,
        class_weight_grid,
        target_indicator_grid,
    ):
        artifact = train_model1(
            train_examples,
            min_company_rows=min_company_rows,
            blend_weight=blend_weight,
            regularization_c=regularization_c,
            class_weight_balanced=class_weight_balanced,
            include_target_indicator=include_target_indicator,
            enable_temperature_calibration=enable_temperature_calibration,
            enable_isotonic_calibration=enable_isotonic_calibration,
            calibration_fraction=calibration_fraction,
            model_version=model_version,
        )
        probabilities = [
            predict_model1(
                artifact,
                company_symbol=example.company_symbol,
                feature_vector=FeatureVector(
                    target_phrase=example.target_phrase,
                    features=example.features,
                ),
            ).probability
            for example in test_examples
        ]
        labels = [example.label for example in test_examples]
        trials.append(
            Model1OptimizationTrial(
                regularization_c=regularization_c,
                min_company_rows=min_company_rows,
                blend_weight=blend_weight,
                class_weight_balanced=class_weight_balanced,
                include_target_indicator=include_target_indicator,
                isotonic_calibration=enable_isotonic_calibration,
                temperature_calibration=enable_temperature_calibration,
                holdout_brier_score=_brier_score(probabilities, labels),
                holdout_log_loss=_safe_log_loss(labels, probabilities),
                company_override_count=len(artifact.company_overrides),
            )
        )

    if not trials:
        raise ValueError("no optimization trials were produced")
    best_trial = min(trials, key=lambda trial: (trial.holdout_brier_score, trial.holdout_log_loss))
    best_artifact = train_model1(
        examples,
        min_company_rows=best_trial.min_company_rows,
        blend_weight=best_trial.blend_weight,
        regularization_c=best_trial.regularization_c,
        class_weight_balanced=best_trial.class_weight_balanced,
        include_target_indicator=best_trial.include_target_indicator,
        enable_temperature_calibration=best_trial.temperature_calibration,
        enable_isotonic_calibration=best_trial.isotonic_calibration,
        calibration_fraction=calibration_fraction,
        model_version=model_version,
    )
    report = Model1OptimizationReport(
        split_strategy="time-event",
        train_sample_count=len(train_examples),
        holdout_sample_count=len(test_examples),
        best_trial=best_trial,
        holdout_brier_score=best_trial.holdout_brier_score,
        holdout_log_loss=best_trial.holdout_log_loss,
        trials=trials,
    )
    return best_artifact, report


def _predict_linear_probability(
    *,
    intercept: float,
    coefficients: dict[str, float],
    feature_columns: list[str],
    features: dict[str, float],
) -> float:
    linear = intercept + sum(
        coefficients.get(column, 0.0) * float(features.get(column, 0.0))
        for column in feature_columns
    )
    return 1.0 / (1.0 + math.exp(-linear))


def _to_frame(
    examples: list[HistoricalTrainingExample],
    *,
    include_target_indicator: bool = False,
    include_market_features: bool = False,
    hierarchical_priors: HierarchicalPriorModel | None = None,
) -> pd.DataFrame:
    rows = []
    for example in examples:
        row = {key: float(value) for key, value in example.features.items()}
        row.update(
            _model_side_features(
                company_symbol=example.company_symbol,
                target_phrase=example.target_phrase,
                market_probability=float(example.market_probability),
                include_market_features=include_market_features,
                hierarchical_priors=hierarchical_priors,
            )
        )
        if include_target_indicator:
            row[_target_indicator_key(example.target_phrase)] = 1.0
        row["label"] = example.label
        row["company_symbol"] = example.company_symbol
        row["phrase_category"] = phrase_category(example.target_phrase)
        rows.append(row)
    frame = pd.DataFrame(rows)
    feature_columns = [
        column
        for column in frame.columns
        if column not in {"label", "company_symbol", "phrase_category"}
    ]
    frame[feature_columns] = frame[feature_columns].fillna(0.0)
    return frame


def _prediction_features(
    *,
    artifact: MentionModelArtifact,
    company_symbol: str,
    target_phrase: str,
    features: dict[str, float],
    market_probability: float | None,
) -> dict[str, float]:
    enriched = dict(features)
    enriched.update(
        _model_side_features(
            company_symbol=company_symbol,
            target_phrase=target_phrase,
            market_probability=market_probability,
            include_market_features=artifact.include_market_features,
            hierarchical_priors=artifact.hierarchical_priors,
        )
    )
    return enriched


def _model_side_features(
    *,
    company_symbol: str,
    target_phrase: str,
    market_probability: float | None,
    include_market_features: bool,
    hierarchical_priors: HierarchicalPriorModel | None,
) -> dict[str, float]:
    features: dict[str, float] = {}
    if include_market_features:
        features.update(market_prior_features(market_probability))
    if hierarchical_priors is not None:
        features.update(
            hierarchical_priors.features_for(
                company_symbol=company_symbol,
                target_phrase=target_phrase,
            )
        )
    return features


def _fit_logistic(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    regularization_c: float,
    class_weight_balanced: bool,
    sample_weight: list[float] | None = None,
) -> LogisticRegression:
    model = LogisticRegression(
        random_state=0,
        solver="liblinear",
        C=regularization_c,
        max_iter=2000,
        class_weight="balanced" if class_weight_balanced else None,
    )
    model.fit(features, labels, sample_weight=sample_weight)
    return model


def _event_quarter_index(example: HistoricalTrainingExample) -> int:
    return (example.fiscal_year * 4) + (example.fiscal_quarter - 1)


def _ema_recency_weights(
    examples: list[HistoricalTrainingExample],
    *,
    half_life_quarters: float,
) -> list[float]:
    if half_life_quarters <= 0:
        raise ValueError("half_life_quarters must be greater than 0")
    if not examples:
        return []
    newest_index = max(_event_quarter_index(example) for example in examples)
    return [
        float(0.5 ** ((newest_index - _event_quarter_index(example)) / half_life_quarters))
        for example in examples
    ]


def _target_indicator_key(target_phrase: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", target_phrase.lower()).strip("_")
    if not normalized:
        normalized = "unknown"
    return f"target_indicator__{normalized}"


def _prediction_context(
    *,
    target_phrase: str,
    raw_features: dict[str, float],
    feature_columns: list[str],
    include_target_indicator: bool,
) -> tuple[dict[str, float], list[str], bool]:
    features = dict(raw_features)
    if not include_target_indicator:
        return features, feature_columns, False
    indicator_key = _target_indicator_key(target_phrase)
    indicator_seen = indicator_key in feature_columns
    if indicator_seen:
        features[indicator_key] = 1.0
        return features, feature_columns, False
    fallback_columns = [
        column for column in feature_columns if not column.startswith("target_indicator__")
    ]
    return features, fallback_columns, True


def _fit_isotonic_calibration(
    *,
    examples: list[HistoricalTrainingExample],
    min_company_rows: int,
    blend_weight: float,
    regularization_c: float,
    class_weight_balanced: bool,
    include_target_indicator: bool,
    calibration_fraction: float,
) -> IsotonicCalibrationModel | None:
    try:
        calibration_train_examples, calibration_examples = _time_split_examples(
            examples,
            test_fraction=calibration_fraction,
        )
    except ValueError:
        return None
    if len(calibration_examples) < 16:
        return None
    labels = [example.label for example in calibration_examples]
    if len(set(labels)) < 2:
        return None
    provisional_artifact = train_model1(
        calibration_train_examples,
        min_company_rows=min_company_rows,
        blend_weight=blend_weight,
        regularization_c=regularization_c,
        class_weight_balanced=class_weight_balanced,
        include_target_indicator=include_target_indicator,
        enable_isotonic_calibration=False,
    )
    probabilities = [
        predict_model1(
            provisional_artifact,
            company_symbol=example.company_symbol,
            feature_vector=FeatureVector(
                target_phrase=example.target_phrase,
                features=example.features,
            ),
        ).probability
        for example in calibration_examples
    ]
    calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    calibrator.fit(probabilities, labels)
    x_thresholds = [float(value) for value in calibrator.X_thresholds_]
    y_thresholds = [float(value) for value in calibrator.y_thresholds_]
    if not x_thresholds or not y_thresholds:
        return None
    return IsotonicCalibrationModel(
        x_thresholds=x_thresholds,
        y_thresholds=y_thresholds,
    )


def _fit_temperature_calibration(
    *,
    examples: list[HistoricalTrainingExample],
    min_company_rows: int,
    blend_weight: float,
    regularization_c: float,
    class_weight_balanced: bool,
    include_target_indicator: bool,
    calibration_fraction: float,
) -> TemperatureCalibrationModel | None:
    try:
        calibration_train_examples, calibration_examples = _time_split_examples(
            examples,
            test_fraction=calibration_fraction,
        )
    except ValueError:
        return None
    if len(calibration_examples) < 16:
        return None
    labels = [example.label for example in calibration_examples]
    if len(set(labels)) < 2:
        return None
    provisional_artifact = train_model1(
        calibration_train_examples,
        min_company_rows=min_company_rows,
        blend_weight=blend_weight,
        regularization_c=regularization_c,
        class_weight_balanced=class_weight_balanced,
        include_target_indicator=include_target_indicator,
        enable_temperature_calibration=False,
        enable_isotonic_calibration=False,
    )
    probabilities = [
        predict_model1(
            provisional_artifact,
            company_symbol=example.company_symbol,
            feature_vector=FeatureVector(
                target_phrase=example.target_phrase,
                features=example.features,
            ),
        ).probability
        for example in calibration_examples
    ]
    candidate_temperatures = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    best_temperature = min(
        candidate_temperatures,
        key=lambda temperature: (
            _brier_score(
                [
                    TemperatureCalibrationModel(temperature=temperature).calibrate(probability)
                    for probability in probabilities
                ],
                labels,
            ),
            _safe_log_loss(
                labels,
                [
                    TemperatureCalibrationModel(temperature=temperature).calibrate(probability)
                    for probability in probabilities
                ],
            ),
        ),
    )
    return TemperatureCalibrationModel(temperature=best_temperature)


def _time_split_examples(
    examples: list[HistoricalTrainingExample],
    *,
    test_fraction: float,
) -> tuple[list[HistoricalTrainingExample], list[HistoricalTrainingExample]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    events: dict[tuple[int, int, str], list[HistoricalTrainingExample]] = defaultdict(list)
    for example in examples:
        event_key = (example.fiscal_year, example.fiscal_quarter, example.company_symbol)
        events[event_key].append(example)
    ordered_keys = sorted(
        events,
        key=lambda key: (
            min(example.evidence_cutoff for example in events[key]),
            key[2],
            key[0],
            key[1],
        ),
    )
    test_event_count = max(1, round(len(ordered_keys) * test_fraction))
    train_event_count = len(ordered_keys) - test_event_count
    if train_event_count < 1:
        raise ValueError("time split leaves no training events")
    train_keys = set(ordered_keys[:train_event_count])
    train_examples = [
        example
        for key in ordered_keys
        if key in train_keys
        for example in events[key]
    ]
    test_examples = [
        example
        for key in ordered_keys
        if key not in train_keys
        for example in events[key]
    ]
    if not train_examples or not test_examples:
        raise ValueError("time split produced empty train or test examples")
    return train_examples, test_examples


def _brier_score(probabilities: list[float], labels: list[int]) -> float:
    squared_error = [
        (probability - label) ** 2 for probability, label in zip(probabilities, labels, strict=True)
    ]
    return round(sum(squared_error) / len(labels), 6)


def _safe_log_loss(labels: list[int], probabilities: list[float]) -> float:
    clipped = [min(0.99, max(0.01, probability)) for probability in probabilities]
    return round(float(log_loss(labels, clipped, labels=[0, 1])), 6)
