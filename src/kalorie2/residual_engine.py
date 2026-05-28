import math
from collections import defaultdict
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kalorie2.prediction_types import PredictionInputRow, prediction_row_key


class ResidualEngineModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResidualPrediction(ResidualEngineModel):
    row_key: str = ""
    market_ticker: str = ""
    event_ticker: str = ""
    probability: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    market_probability: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    residual_delta: float
    feature_values: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    training_event_tickers: list[str] = Field(default_factory=list)


class LinearResidualModel(ResidualEngineModel):
    weights: dict[str, float] = Field(default_factory=dict)
    intercept: float = 0.0
    feature_means: dict[str, float] = Field(default_factory=dict)
    feature_scales: dict[str, float] = Field(default_factory=dict)
    residual_clip: float = Field(default=2.0, gt=0.0)
    target_side: Literal["yes", "no"] = "yes"
    positive_label_weight: float = Field(default=1.0, gt=0.0)

    def predict(
        self,
        *,
        market_probability: float,
        feature_values: dict[str, float],
        market_ticker: str = "",
        event_ticker: str = "",
        row_key: str = "",
        training_event_tickers: list[str] | None = None,
    ) -> ResidualPrediction:
        residual_delta = self.intercept + sum(
            self.weights.get(key, 0.0)
            * _standardized_value(key, feature_values, self.feature_means, self.feature_scales)
            for key in self.weights
        )
        residual_delta = _clip(residual_delta, self.residual_clip)
        side_market_probability = _side_market_probability(market_probability, self.target_side)
        side_probability = apply_residual(side_market_probability, residual_delta)
        probability = _yes_probability_from_side(side_probability, self.target_side)
        return ResidualPrediction(
            row_key=row_key,
            market_ticker=market_ticker,
            event_ticker=event_ticker,
            probability=_decimal_probability(probability),
            market_probability=_decimal_probability(market_probability),
            residual_delta=round(residual_delta, 12),
            feature_values=feature_values,
            reasons=[
                "market_anchor",
                "linear_residual",
                f"target_side:{self.target_side}",
                *_feature_reasons(
                    self.weights,
                    feature_values,
                    self.feature_means,
                    self.feature_scales,
                ),
            ],
            training_event_tickers=training_event_tickers or [],
        )


def apply_residual(market_probability: float, residual_delta: float) -> float:
    probability = _sigmoid(_safe_logit(market_probability) + residual_delta)
    return min(0.999999, max(0.000001, probability))


def fit_linear_residual_model(
    rows: list[PredictionInputRow],
    feature_rows: list[dict[str, float]],
    *,
    epochs: int = 100,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    residual_clip: float = 2.0,
    target_side: Literal["yes", "no"] = "yes",
    positive_label_weight: float = 1.0,
) -> LinearResidualModel:
    if len(rows) != len(feature_rows):
        raise ValueError("rows and feature_rows must have the same length")
    if not rows:
        return LinearResidualModel()

    feature_names = sorted({key for feature_row in feature_rows for key in feature_row})
    feature_means, feature_scales = _feature_stats(feature_rows, feature_names)
    weights = {name: 0.0 for name in feature_names}
    intercept = 0.0
    total_weight = sum(
        _side_training_weight(
            row,
            target_side,
            positive_label_weight=positive_label_weight,
        )
        for row in rows
    )

    for _ in range(epochs):
        intercept_gradient = 0.0
        weight_gradients = {name: 0.0 for name in feature_names}

        for row, feature_row in zip(rows, feature_rows, strict=True):
            training_weight = _side_training_weight(
                row,
                target_side,
                positive_label_weight=positive_label_weight,
            )
            residual_delta = intercept + sum(
                weights[name]
                * _standardized_value(name, feature_row, feature_means, feature_scales)
                for name in feature_names
            )
            residual_delta = _clip(residual_delta, residual_clip)
            probability = apply_residual(
                _side_market_probability(float(row.preclose_yes_mid), target_side),
                residual_delta,
            )
            error = probability - _side_outcome_label(row, target_side)
            intercept_gradient += training_weight * error
            for name in feature_names:
                weight_gradients[name] += training_weight * error * _standardized_value(
                    name,
                    feature_row,
                    feature_means,
                    feature_scales,
                )

        intercept -= learning_rate * intercept_gradient / total_weight
        for name in feature_names:
            gradient = weight_gradients[name] / total_weight + l2 * weights[name]
            weights[name] -= learning_rate * gradient

    return LinearResidualModel(
        weights={name: weight for name, weight in weights.items() if abs(weight) > 1e-12},
        intercept=intercept,
        feature_means=feature_means,
        feature_scales=feature_scales,
        residual_clip=residual_clip,
        target_side=target_side,
        positive_label_weight=positive_label_weight,
    )


def walk_forward_predictions(
    rows: list[PredictionInputRow],
    feature_rows: list[dict[str, float]],
    *,
    min_training_events: int = 1,
    epochs: int = 100,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    residual_clip: float = 2.0,
    target_side: Literal["yes", "no"] = "yes",
    positive_label_weight: float = 1.0,
) -> list[ResidualPrediction]:
    if len(rows) != len(feature_rows):
        raise ValueError("rows and feature_rows must have the same length")

    grouped = _group_by_event(rows, feature_rows)
    prior_rows: list[PredictionInputRow] = []
    prior_feature_rows: list[dict[str, float]] = []
    prior_event_tickers: list[str] = []
    predictions: list[ResidualPrediction] = []

    for event_ticker, event_pairs in grouped:
        if len(prior_event_tickers) >= min_training_events:
            model = fit_linear_residual_model(
                prior_rows,
                prior_feature_rows,
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
                residual_clip=residual_clip,
                target_side=target_side,
                positive_label_weight=positive_label_weight,
            )
            for row, feature_row in event_pairs:
                predictions.append(
                    model.predict(
                        market_probability=float(row.preclose_yes_mid),
                        feature_values=feature_row,
                        market_ticker=row.market_ticker,
                        event_ticker=row.event_ticker,
                        row_key=prediction_row_key(row),
                        training_event_tickers=list(prior_event_tickers),
                    )
                )

        prior_rows.extend(row for row, _ in event_pairs)
        prior_feature_rows.extend(feature_row for _, feature_row in event_pairs)
        prior_event_tickers.append(event_ticker)

    return predictions


def _group_by_event(
    rows: list[PredictionInputRow],
    feature_rows: list[dict[str, float]],
) -> list[tuple[str, list[tuple[PredictionInputRow, dict[str, float]]]]]:
    grouped: dict[str, list[tuple[PredictionInputRow, dict[str, float]]]] = defaultdict(list)
    for row, feature_row in zip(rows, feature_rows, strict=True):
        grouped[row.event_ticker].append((row, feature_row))
    return sorted(
        grouped.items(),
        key=lambda item: (
            min(row.close_time for row, _ in item[1]),
            item[0],
        ),
    )


def _feature_reasons(
    weights: dict[str, float],
    feature_values: dict[str, float],
    feature_means: dict[str, float],
    feature_scales: dict[str, float],
) -> list[str]:
    contributions = [
        (
            key,
            weight * _standardized_value(key, feature_values, feature_means, feature_scales),
        )
        for key, weight in weights.items()
    ]
    positive = [
        key
        for key, contribution in sorted(contributions, key=lambda item: item[1], reverse=True)
        if contribution > 0
    ]
    negative = [
        key
        for key, contribution in sorted(contributions, key=lambda item: item[1])
        if contribution < 0
    ]
    reasons = []
    if positive:
        reasons.append(f"positive:{positive[0]}")
    if negative:
        reasons.append(f"negative:{negative[0]}")
    return reasons


def _safe_logit(probability: float) -> float:
    clipped = min(0.999999, max(0.000001, probability))
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _decimal_probability(value: float) -> Decimal:
    return Decimal(f"{value:.6f}")


def _feature_stats(
    feature_rows: list[dict[str, float]],
    feature_names: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    means = {}
    scales = {}
    count = float(len(feature_rows))
    for name in feature_names:
        values = [float(feature_row.get(name, 0.0)) for feature_row in feature_rows]
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / count
        scale = math.sqrt(variance)
        means[name] = mean
        scales[name] = scale if scale > 1e-9 else 1.0
    return means, scales


def _standardized_value(
    name: str,
    feature_values: dict[str, float],
    feature_means: dict[str, float],
    feature_scales: dict[str, float],
) -> float:
    raw_value = float(feature_values.get(name, 0.0))
    if name not in feature_means:
        return raw_value
    return (raw_value - feature_means[name]) / feature_scales.get(name, 1.0)


def _clip(value: float, limit: float) -> float:
    return min(limit, max(-limit, value))


def _side_market_probability(market_probability: float, target_side: Literal["yes", "no"]) -> float:
    if target_side == "yes":
        return market_probability
    return 1.0 - market_probability


def _yes_probability_from_side(side_probability: float, target_side: Literal["yes", "no"]) -> float:
    if target_side == "yes":
        return side_probability
    return 1.0 - side_probability


def _side_outcome_label(row: PredictionInputRow, target_side: Literal["yes", "no"]) -> int:
    if target_side == "yes":
        return row.outcome_label
    return 1 - row.outcome_label


def _side_training_weight(
    row: PredictionInputRow,
    target_side: Literal["yes", "no"],
    *,
    positive_label_weight: float,
) -> float:
    if _side_outcome_label(row, target_side) == 1:
        return positive_label_weight
    return 1.0
