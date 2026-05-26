import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kalorie2.prediction_types import PredictionInputRow
from kalorie2.residual_engine import ResidualPrediction


class MixMcpModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MixMcpTargetProbability(MixMcpModel):
    market_ticker: str = Field(min_length=1)
    target_phrase: str = Field(min_length=1)
    market_probability: float = Field(ge=0.0, le=1.0)
    llm_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)

    @field_validator("target_phrase")
    @classmethod
    def normalize_target_phrase(cls, value: str) -> str:
        return value.strip().lower()


class MixMcpEventPacket(MixMcpModel):
    event_ticker: str = Field(min_length=1)
    cutoff_time: datetime
    model: str = Field(min_length=1)
    targets: list[MixMcpTargetProbability] = Field(default_factory=list)

    @field_validator("cutoff_time")
    @classmethod
    def normalize_cutoff_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


AlphaMode = Literal["global", "side"]


def build_mixmcp_prompt(*, event: dict[str, Any], targets: list[dict[str, Any]]) -> str:
    payload = {
        "event": event,
        "targets": targets,
        "instructions": [
            "Use the market_probability as the explicit market prior.",
            "Use only cutoff-safe web evidence available before or at cutoff_time.",
            "Return an updated llm_probability for each target, not a trade recommendation.",
            "Do not use earnings-call transcripts, post-call recaps, Kalshi resolution pages, "
            "or final outcomes.",
        ],
        "required_schema": _mixmcp_schema(),
    }
    return (
        "Market-conditioned prompt for earnings mention forecasting. "
        "Return only strict JSON matching `required_schema`.\n"
        + json.dumps(payload, indent=2, sort_keys=True)
    )


def build_openai_mixmcp_payload(*, prompt: str, model: str = "gpt-5.4-mini") -> dict[str, Any]:
    return {
        "model": model,
        "input": prompt,
        "tools": [{"type": "web_search"}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "mixmcp_event_packet",
                "strict": True,
                "schema": _mixmcp_schema(),
            }
        },
    }


def parse_mixmcp_response(text: str) -> MixMcpEventPacket:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("MixMCP response must be strict JSON") from exc
    return MixMcpEventPacket.model_validate(payload)


def mix_probability(base_probability: float, llm_probability: float, *, alpha: float) -> float:
    clipped_alpha = min(1.0, max(0.0, alpha))
    mixed_logit = (
        clipped_alpha * _safe_logit(base_probability)
        + (1.0 - clipped_alpha) * _safe_logit(llm_probability)
    )
    return min(0.999999, max(0.000001, _sigmoid(mixed_logit)))


def apply_mixmcp_to_predictions(
    rows: list[PredictionInputRow],
    predictions: list[ResidualPrediction],
    packets_by_event: dict[str, MixMcpEventPacket],
    *,
    alpha_grid: list[float] | None = None,
    alpha_mode: AlphaMode = "global",
    fixed_alpha: float | None = None,
) -> list[ResidualPrediction]:
    if not packets_by_event:
        return predictions
    alpha_grid = alpha_grid or [index / 10 for index in range(11)]
    rows_by_market = {row.market_ticker: row for row in rows}
    rows_by_event = _rows_by_event(rows)
    grouped_predictions = _group_predictions_by_event(predictions)
    prior_predictions: list[ResidualPrediction] = []
    mixed_predictions: list[ResidualPrediction] = []

    for _, event_predictions in grouped_predictions:
        event_mixed = []
        for prediction in event_predictions:
            row = rows_by_market.get(prediction.market_ticker)
            target = _target_for_prediction(row, prediction, packets_by_event)
            if row is None or target is None:
                event_mixed.append(prediction)
                continue
            side = _base_side(prediction, row)
            alpha = (
                fixed_alpha
                if fixed_alpha is not None
                else _select_alpha(
                    rows_by_market,
                    rows_by_event,
                    prior_predictions,
                    packets_by_event,
                    training_event_tickers=prediction.training_event_tickers,
                    alpha_grid=alpha_grid,
                    alpha_mode=alpha_mode,
                    side=side,
                )
            )
            event_mixed.append(_mixed_prediction(prediction, target, alpha=alpha))
        mixed_predictions.extend(event_mixed)
        prior_predictions.extend(event_predictions)

    return mixed_predictions


def _select_alpha(
    rows_by_market: dict[str, PredictionInputRow],
    rows_by_event: dict[str, list[PredictionInputRow]],
    prior_predictions: list[ResidualPrediction],
    packets_by_event: dict[str, MixMcpEventPacket],
    *,
    training_event_tickers: list[str],
    alpha_grid: list[float],
    alpha_mode: AlphaMode,
    side: str,
) -> float:
    prior_by_market = {prediction.market_ticker: prediction for prediction in prior_predictions}
    candidates = []
    for event_ticker in training_event_tickers:
        for row in rows_by_event.get(event_ticker, []):
            target = _target_for_row(row, packets_by_event)
            if target is None:
                continue
            prior_prediction = prior_by_market.get(row.market_ticker)
            base_probability = (
                float(prior_prediction.probability)
                if prior_prediction is not None
                else float(row.preclose_yes_mid)
            )
            candidate_side = (
                _base_side(prior_prediction, row)
                if prior_prediction is not None
                else _side_for_probability(base_probability, row)
            )
            if alpha_mode == "side" and candidate_side != side:
                continue
            candidates.append((row, base_probability, target))
    if not candidates:
        return 1.0
    return min(alpha_grid, key=lambda alpha: _alpha_brier(candidates, alpha))


def _alpha_brier(
    candidates: list[
        tuple[PredictionInputRow, float, MixMcpTargetProbability]
    ],
    alpha: float,
) -> float:
    return sum(
        (
            mix_probability(
                base_probability,
                target.llm_probability,
                alpha=alpha,
            )
            - row.outcome_label
        )
        ** 2
        for row, base_probability, target in candidates
    ) / len(candidates)


def _mixed_prediction(
    prediction: ResidualPrediction,
    target: MixMcpTargetProbability,
    *,
    alpha: float,
) -> ResidualPrediction:
    probability = mix_probability(
        float(prediction.probability),
        target.llm_probability,
        alpha=alpha,
    )
    return prediction.model_copy(
        update={
            "probability": Decimal(f"{probability:.6f}"),
            "reasons": [
                *prediction.reasons,
                f"mixmcp_alpha:{alpha:.3f}",
                f"mixmcp_model_probability:{target.llm_probability:.6f}",
            ],
        }
    )


def _target_for_prediction(
    row: PredictionInputRow | None,
    prediction: ResidualPrediction,
    packets_by_event: dict[str, MixMcpEventPacket],
) -> MixMcpTargetProbability | None:
    packet = packets_by_event.get(prediction.event_ticker)
    return _target_from_packet(row, prediction.market_ticker, packet)


def _target_for_row(
    row: PredictionInputRow,
    packets_by_event: dict[str, MixMcpEventPacket],
) -> MixMcpTargetProbability | None:
    return _target_from_packet(row, row.market_ticker, packets_by_event.get(row.event_ticker))


def _target_from_packet(
    row: PredictionInputRow | None,
    market_ticker: str,
    packet: MixMcpEventPacket | None,
) -> MixMcpTargetProbability | None:
    if packet is None:
        return None
    for target in packet.targets:
        if target.market_ticker == market_ticker:
            return target
    if row is None:
        return None
    normalized = (row.normalized_word_said or row.word_said).strip().lower()
    for target in packet.targets:
        if target.target_phrase == normalized:
            return target
    return None


def _base_side(prediction: ResidualPrediction, row: PredictionInputRow) -> str:
    return _side_for_probability(float(prediction.probability), row)


def _side_for_probability(probability: float, row: PredictionInputRow) -> str:
    if probability > float(row.preclose_yes_ask):
        return "YES"
    if probability < float(row.preclose_yes_bid):
        return "NO"
    return "NONE"


def _rows_by_event(rows: list[PredictionInputRow]) -> dict[str, list[PredictionInputRow]]:
    grouped: dict[str, list[PredictionInputRow]] = defaultdict(list)
    for row in rows:
        grouped[row.event_ticker].append(row)
    return grouped


def _group_predictions_by_event(
    predictions: list[ResidualPrediction],
) -> list[tuple[str, list[ResidualPrediction]]]:
    grouped: dict[str, list[ResidualPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.event_ticker].append(prediction)
    return list(grouped.items())


def _mixmcp_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "event_ticker": {"type": "string"},
            "cutoff_time": {"type": "string"},
            "model": {"type": "string"},
            "targets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "market_ticker": {"type": "string"},
                        "target_phrase": {"type": "string"},
                        "market_probability": {"type": "number", "minimum": 0, "maximum": 1},
                        "llm_probability": {"type": "number", "minimum": 0, "maximum": 1},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "market_ticker",
                        "target_phrase",
                        "market_probability",
                        "llm_probability",
                        "confidence",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["event_ticker", "cutoff_time", "model", "targets"],
        "additionalProperties": False,
    }


def _safe_logit(probability: float) -> float:
    clipped = min(0.999999, max(0.000001, probability))
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
