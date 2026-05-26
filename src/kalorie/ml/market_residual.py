from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import GradientBoostingRegressor

from kalorie.domain.models import FeatureVector, Prediction
from kalorie.ml.datasets import HistoricalTrainingExample
from kalorie.ml.grouped_calibration import (
    GroupedCalibrationExample,
    GroupedTemperatureCalibrationModel,
    evidence_strength_bucket,
    fit_grouped_temperature_calibration,
)
from kalorie.ml.priors import (
    fit_hierarchical_prior_model,
    market_prior_features,
    phrase_category,
    phrase_category_features,
)


class MarketResidualArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model_version: str = "market-residual-v1"
    feature_columns: list[str]
    residual_model: Any = Field(exclude=True)
    hierarchical_priors: Any = Field(default=None, exclude=True)
    grouped_calibration: GroupedTemperatureCalibrationModel | None = None


def market_microstructure_features(
    *,
    yes_bid: Decimal | float | None,
    yes_ask: Decimal | float | None,
) -> dict[str, float]:
    if yes_bid is None or yes_ask is None:
        return {
            "market_mid_probability": 0.5,
            "market_spread": 1.0,
            "market_wide_spread_binary": 1.0,
            "market_illiquidity_score": 1.0,
        }
    bid = float(yes_bid)
    ask = float(yes_ask)
    spread = max(0.0, ask - bid)
    mid = min(0.999999, max(0.000001, (bid + ask) / 2.0))
    return {
        "market_mid_probability": round(mid, 6),
        "market_spread": round(spread, 6),
        "market_wide_spread_binary": 1.0 if spread >= 0.5 else 0.0,
        "market_illiquidity_score": round(spread, 6),
    }


def train_market_residual(
    examples: list[HistoricalTrainingExample],
) -> MarketResidualArtifact:
    if len(examples) < 4:
        raise ValueError("at least 4 examples are required")
    hierarchical_priors = fit_hierarchical_prior_model(examples)
    rows = [
        _feature_row(
            company_symbol=example.company_symbol,
            target_phrase=example.target_phrase,
            features=example.features,
            market_probability=float(example.market_probability),
            hierarchical_priors=hierarchical_priors,
        )
        for example in examples
    ]
    frame = pd.DataFrame(rows).fillna(0.0)
    feature_columns = sorted(frame.columns)
    targets = [
        _safe_logit(0.99 if example.label else 0.01)
        - _safe_logit(float(example.market_probability))
        for example in examples
    ]
    model = GradientBoostingRegressor(random_state=0, n_estimators=50, max_depth=2)
    model.fit(frame[feature_columns], targets)
    raw_probabilities = [
        _sigmoid(
            _safe_logit(float(example.market_probability))
            + float(model.predict(frame[feature_columns].iloc[[index]])[0])
        )
        for index, example in enumerate(examples)
    ]
    grouped_calibration = fit_grouped_temperature_calibration(
        [
            GroupedCalibrationExample(
                phrase_category(example.target_phrase),
                evidence_strength_bucket(example.features),
                probability,
                example.label,
            )
            for example, probability in zip(examples, raw_probabilities, strict=True)
        ],
        min_group_rows=2,
        shrinkage=4.0,
    )
    return MarketResidualArtifact(
        feature_columns=feature_columns,
        residual_model=model,
        hierarchical_priors=hierarchical_priors,
        grouped_calibration=grouped_calibration,
    )


def predict_market_residual(
    artifact: MarketResidualArtifact,
    *,
    company_symbol: str,
    feature_vector: FeatureVector,
    market_probability: float,
) -> Prediction:
    row = _feature_row(
        company_symbol=company_symbol,
        target_phrase=feature_vector.target_phrase,
        features=feature_vector.features,
        market_probability=market_probability,
        hierarchical_priors=artifact.hierarchical_priors,
    )
    frame = pd.DataFrame([{column: row.get(column, 0.0) for column in artifact.feature_columns}])
    residual_logit = float(artifact.residual_model.predict(frame)[0])
    probability = _sigmoid(_safe_logit(market_probability) + residual_logit)
    reasons = ["market_residual", "market_anchor", "gradient_boosting"]
    if artifact.grouped_calibration is not None:
        probability = artifact.grouped_calibration.calibrate(
            probability,
            category=phrase_category(feature_vector.target_phrase),
            evidence_bucket=evidence_strength_bucket(feature_vector.features),
        )
        reasons.append("grouped_calibration")
    return Prediction(
        target_phrase=feature_vector.target_phrase,
        model_version=artifact.model_version,
        probability=_clamp_probability(probability),
        reasons=reasons,
    )


def _feature_row(
    *,
    company_symbol: str,
    target_phrase: str,
    features: dict[str, float],
    market_probability: float,
    hierarchical_priors: Any,
) -> dict[str, float]:
    row = {key: float(value) for key, value in features.items()}
    row.update(phrase_category_features(target_phrase))
    row.update(market_prior_features(market_probability))
    row.update(
        market_microstructure_features(
            yes_bid=None,
            yes_ask=None,
        )
    )
    if hierarchical_priors is not None:
        row.update(
            hierarchical_priors.features_for(
                company_symbol=company_symbol,
                target_phrase=target_phrase,
            )
        )
    return row


def _safe_logit(probability: float) -> float:
    clipped = min(0.999999, max(0.000001, float(probability)))
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _clamp_probability(value: float) -> float:
    return min(0.99, max(0.01, round(value, 6)))
