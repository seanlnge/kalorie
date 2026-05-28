from __future__ import annotations

import math
import random
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ModelCardBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConfidenceInterval(ModelCardBase):
    low: float
    high: float
    method: str = "event_bootstrap"
    confidence_level: float = 0.95


class MetricValue(ModelCardBase):
    value: float
    unit: str | None = None
    ci95: ConfidenceInterval | None = None
    description: str | None = None


class EvaluationSplit(ModelCardBase):
    name: str
    role: Literal["train", "validation", "test", "backtest", "reference"]
    event_count: int = Field(ge=0)
    market_count: int = Field(ge=0)
    metrics: dict[str, MetricValue]
    notes: str | None = None


class ModelCard(ModelCardBase):
    schema_version: str = "1.0"
    model_name: str
    model_version: int | None = None
    model_type: str
    training_data: dict[str, int | float | str | None]
    feature_set: dict[str, int | float | str | list[str] | None]
    evaluation_splits: list[EvaluationSplit]
    caveats: list[str] = Field(default_factory=list)
    recommended_use: str | None = None

    @computed_field
    @property
    def primary_test_split(self) -> EvaluationSplit:
        for split in self.evaluation_splits:
            if split.role == "test":
                return split
        raise ValueError("model card must include at least one test split")


def build_model_card_schema() -> dict:
    return ModelCard.model_json_schema()


class EvaluationRow(ModelCardBase):
    event_ticker: str
    close_time: datetime
    outcome_label: int = Field(ge=0, le=1)
    market_probability: float = Field(ge=0.0, le=1.0)
    model_probability: float = Field(ge=0.0, le=1.0)
    yes_bid: float = Field(ge=0.0, le=1.0)
    yes_ask: float = Field(ge=0.0, le=1.0)


def parse_iso_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def latest_event_rows(rows: list[EvaluationRow], event_count: int = 30) -> list[EvaluationRow]:
    grouped: dict[str, list[EvaluationRow]] = {}
    for row in rows:
        grouped.setdefault(row.event_ticker, []).append(row)
    ordered = sorted(
        grouped.items(),
        key=lambda item: (min(entry.close_time for entry in item[1]), item[0]),
    )
    keep = {event for event, _ in ordered[-event_count:]}
    return [row for row in rows if row.event_ticker in keep]


def build_evaluation_split(
    rows: list[EvaluationRow],
    *,
    name: str,
    role: Literal["train", "validation", "test", "backtest", "reference"],
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 531,
    notes: str | None = None,
) -> EvaluationSplit:
    probability_summary = summarize_probability_metrics(rows)
    intervals = bootstrap_metric_intervals(
        rows,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    metrics = {
        "brier": MetricValue(
            value=round(probability_summary["brier"], 6),
            ci95=intervals["brier"],
            description="Mean squared probability error; lower is better.",
        ),
        "market_brier": MetricValue(
            value=round(probability_summary["market_brier"], 6),
            ci95=intervals["market_brier"],
            description="Baseline Brier score from market probability.",
        ),
        "brier_edge_vs_market": MetricValue(
            value=round(probability_summary["brier_edge_vs_market"], 6),
            ci95=intervals["brier_edge_vs_market"],
            description="Market Brier minus model Brier; positive means the model improved.",
        ),
        "event_weighted_brier": MetricValue(
            value=round(probability_summary["event_weighted_brier"], 6),
            ci95=intervals["event_weighted_brier"],
            description="Event-weighted mean squared probability error.",
        ),
        "event_weighted_market_brier": MetricValue(
            value=round(probability_summary["event_weighted_market_brier"], 6),
            ci95=intervals["event_weighted_market_brier"],
            description="Event-weighted baseline Brier score from market probability.",
        ),
        "event_weighted_brier_edge_vs_market": MetricValue(
            value=round(probability_summary["event_weighted_brier_edge_vs_market"], 6),
            ci95=intervals["event_weighted_brier_edge_vs_market"],
            description="Event-weighted market Brier minus model Brier.",
        ),
        "ece": MetricValue(
            value=round(probability_summary["ece"], 6),
            ci95=intervals["ece"],
            description="Ten-bin expected calibration error; lower is better.",
        ),
        "market_ece": MetricValue(
            value=round(probability_summary["market_ece"], 6),
            ci95=intervals["market_ece"],
            description="Baseline ECE from market probability.",
        ),
        "ece_edge_vs_market": MetricValue(
            value=round(probability_summary["ece_edge_vs_market"], 6),
            ci95=intervals["ece_edge_vs_market"],
            description="Market ECE minus model ECE; positive means the model improved.",
        ),
        "event_weighted_ece": MetricValue(
            value=round(probability_summary["event_weighted_ece"], 6),
            ci95=intervals["event_weighted_ece"],
            description="Event-weighted ten-bin expected calibration error.",
        ),
        "event_weighted_market_ece": MetricValue(
            value=round(probability_summary["event_weighted_market_ece"], 6),
            ci95=intervals["event_weighted_market_ece"],
            description="Event-weighted baseline ECE from market probability.",
        ),
        "event_weighted_ece_edge_vs_market": MetricValue(
            value=round(probability_summary["event_weighted_ece_edge_vs_market"], 6),
            ci95=intervals["event_weighted_ece_edge_vs_market"],
            description="Event-weighted market ECE minus model ECE.",
        ),
        "log_loss": MetricValue(
            value=round(probability_summary["log_loss"], 6),
            ci95=intervals["log_loss"],
            description="Cross-entropy log loss; lower is better.",
        ),
        "market_log_loss": MetricValue(
            value=round(probability_summary["market_log_loss"], 6),
            ci95=intervals["market_log_loss"],
            description="Baseline log loss from market probability.",
        ),
        "log_loss_edge_vs_market": MetricValue(
            value=round(probability_summary["log_loss_edge_vs_market"], 6),
            ci95=intervals["log_loss_edge_vs_market"],
            description="Market log loss minus model log loss; positive means the model improved.",
        ),
        "event_weighted_log_loss": MetricValue(
            value=round(probability_summary["event_weighted_log_loss"], 6),
            ci95=intervals["event_weighted_log_loss"],
            description="Event-weighted cross-entropy log loss.",
        ),
        "event_weighted_market_log_loss": MetricValue(
            value=round(probability_summary["event_weighted_market_log_loss"], 6),
            ci95=intervals["event_weighted_market_log_loss"],
            description="Event-weighted baseline log loss from market probability.",
        ),
        "event_weighted_log_loss_edge_vs_market": MetricValue(
            value=round(probability_summary["event_weighted_log_loss_edge_vs_market"], 6),
            ci95=intervals["event_weighted_log_loss_edge_vs_market"],
            description="Event-weighted market log loss minus model log loss.",
        ),
    }
    return EvaluationSplit(
        name=name,
        role=role,
        event_count=len({row.event_ticker for row in rows}),
        market_count=len(rows),
        metrics=metrics,
        notes=notes,
    )


def summarize_probability_metrics(rows: list[EvaluationRow]) -> dict[str, float]:
    if not rows:
        return {
            "brier": 0.0,
            "market_brier": 0.0,
            "brier_edge_vs_market": 0.0,
            "event_weighted_brier": 0.0,
            "event_weighted_market_brier": 0.0,
            "event_weighted_brier_edge_vs_market": 0.0,
            "ece": 0.0,
            "market_ece": 0.0,
            "ece_edge_vs_market": 0.0,
            "event_weighted_ece": 0.0,
            "event_weighted_market_ece": 0.0,
            "event_weighted_ece_edge_vs_market": 0.0,
            "log_loss": 0.0,
            "market_log_loss": 0.0,
            "log_loss_edge_vs_market": 0.0,
            "event_weighted_log_loss": 0.0,
            "event_weighted_market_log_loss": 0.0,
            "event_weighted_log_loss_edge_vs_market": 0.0,
        }
    brier = _mean(
        (row.model_probability - row.outcome_label) ** 2
        for row in rows
    )
    market_brier = _mean(
        (row.market_probability - row.outcome_label) ** 2
        for row in rows
    )
    ece = _ece(rows, probability_key="model_probability")
    market_ece = _ece(rows, probability_key="market_probability")
    log_loss = _log_loss(rows, probability_key="model_probability")
    market_log_loss = _log_loss(rows, probability_key="market_probability")
    event_weighted = _event_weighted_probability_metrics(rows)
    return {
        "brier": _mean(
            (row.model_probability - row.outcome_label) ** 2
            for row in rows
        ),
        "market_brier": market_brier,
        "brier_edge_vs_market": market_brier - brier,
        "event_weighted_brier": event_weighted["brier"],
        "event_weighted_market_brier": event_weighted["market_brier"],
        "event_weighted_brier_edge_vs_market": (
            event_weighted["market_brier"] - event_weighted["brier"]
        ),
        "ece": ece,
        "market_ece": market_ece,
        "ece_edge_vs_market": market_ece - ece,
        "event_weighted_ece": event_weighted["ece"],
        "event_weighted_market_ece": event_weighted["market_ece"],
        "event_weighted_ece_edge_vs_market": (
            event_weighted["market_ece"] - event_weighted["ece"]
        ),
        "log_loss": log_loss,
        "market_log_loss": market_log_loss,
        "log_loss_edge_vs_market": market_log_loss - log_loss,
        "event_weighted_log_loss": event_weighted["log_loss"],
        "event_weighted_market_log_loss": event_weighted["market_log_loss"],
        "event_weighted_log_loss_edge_vs_market": (
            event_weighted["market_log_loss"] - event_weighted["log_loss"]
        ),
    }


def bootstrap_metric_intervals(
    rows: list[EvaluationRow],
    *,
    samples: int,
    seed: int,
) -> dict[str, ConfidenceInterval]:
    if not rows:
        zeros = ConfidenceInterval(low=0.0, high=0.0)
        return {key: zeros for key in summarize_probability_metrics([])}
    grouped: dict[str, list[EvaluationRow]] = {}
    for row in rows:
        grouped.setdefault(row.event_ticker, []).append(row)
    groups = list(grouped.values())
    rng = random.Random(seed)
    metrics: dict[str, list[float]] = {
        key: [] for key in summarize_probability_metrics(rows)
    }
    for _ in range(samples):
        sampled_rows: list[EvaluationRow] = []
        for _ in groups:
            sampled_rows.extend(rng.choice(groups))
        probability_summary = summarize_probability_metrics(sampled_rows)
        for key, value in probability_summary.items():
            metrics[key].append(float(value))
    return {
        key: ConfidenceInterval(
            low=round(_percentile(values, 0.025), 6),
            high=round(_percentile(values, 0.975), 6),
        )
        for key, values in metrics.items()
    }


def _event_weighted_probability_metrics(rows: list[EvaluationRow]) -> dict[str, float]:
    grouped: dict[str, list[EvaluationRow]] = {}
    for row in rows:
        grouped.setdefault(row.event_ticker, []).append(row)
    event_summaries = [
        {
            "brier": _mean(
                (row.model_probability - row.outcome_label) ** 2
                for row in event_rows
            ),
            "market_brier": _mean(
                (row.market_probability - row.outcome_label) ** 2
                for row in event_rows
            ),
            "ece": _ece(event_rows, probability_key="model_probability"),
            "market_ece": _ece(event_rows, probability_key="market_probability"),
            "log_loss": _log_loss(event_rows, probability_key="model_probability"),
            "market_log_loss": _log_loss(event_rows, probability_key="market_probability"),
        }
        for event_rows in grouped.values()
    ]
    return {
        "brier": _mean(item["brier"] for item in event_summaries),
        "market_brier": _mean(item["market_brier"] for item in event_summaries),
        "ece": _mean(item["ece"] for item in event_summaries),
        "market_ece": _mean(item["market_ece"] for item in event_summaries),
        "log_loss": _mean(item["log_loss"] for item in event_summaries),
        "market_log_loss": _mean(item["market_log_loss"] for item in event_summaries),
    }


def _mean(values) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return float(sum(collected) / len(collected))


def _clip_probability(value: float) -> float:
    return min(0.999999, max(0.000001, float(value)))


def _log_loss(rows: list[EvaluationRow], *, probability_key: str) -> float:
    return _mean(
        -(
            row.outcome_label * math.log(_clip_probability(getattr(row, probability_key)))
            + (1 - row.outcome_label)
            * math.log(1 - _clip_probability(getattr(row, probability_key)))
        )
        for row in rows
    )


def _ece(rows: list[EvaluationRow], *, probability_key: str, bins: int = 10) -> float:
    if not rows:
        return 0.0
    total = len(rows)
    score = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            row
            for row in rows
            if (
                lower <= getattr(row, probability_key) < upper
                or (index == bins - 1 and getattr(row, probability_key) == 1.0)
            )
        ]
        if not bucket:
            continue
        confidence = _mean(getattr(row, probability_key) for row in bucket)
        accuracy = _mean(row.outcome_label for row in bucket)
        score += (len(bucket) / total) * abs(confidence - accuracy)
    return float(score)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
