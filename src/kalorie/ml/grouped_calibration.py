from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class GroupedCalibrationExample:
    category: str
    evidence_bucket: str
    probability: float
    label: int


class GroupedTemperatureCalibrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fallback_temperature: float = Field(gt=0)
    group_temperatures: dict[str, float] = Field(default_factory=dict)
    group_sample_counts: dict[str, int] = Field(default_factory=dict)
    shrinkage: float = Field(default=4.0, ge=0.0)

    def calibrate(self, probability: float, *, category: str, evidence_bucket: str) -> float:
        group_key = _group_key(category, evidence_bucket)
        group_temperature = self.group_temperatures.get(group_key)
        if group_temperature is None:
            return _apply_temperature(probability, self.fallback_temperature)
        sample_count = self.group_sample_counts.get(group_key, 0)
        group_weight = sample_count / (sample_count + self.shrinkage) if sample_count else 0.0
        effective_temperature = (
            group_weight * group_temperature
            + (1.0 - group_weight) * self.fallback_temperature
        )
        return _apply_temperature(probability, effective_temperature)


def evidence_strength_bucket(features: dict[str, float]) -> str:
    if (
        float(features.get("exact_match_count", 0.0)) > 0
        or float(features.get("alias_lexical_signal_binary", 0.0)) > 0
        or float(features.get("alias_max_embedding_similarity", 0.0)) >= 0.80
    ):
        return "strong"
    semantic_signal = max(
        float(features.get("semantic_signal_max_tfidf", 0.0)),
        float(features.get("max_tfidf_similarity", 0.0)),
        float(features.get("max_embedding_similarity", 0.0)),
        float(features.get("alias_max_tfidf_similarity", 0.0)),
    )
    if semantic_signal >= 0.35:
        return "medium"
    return "weak"


def fit_grouped_temperature_calibration(
    examples: list[GroupedCalibrationExample],
    *,
    min_group_rows: int = 16,
    shrinkage: float = 4.0,
) -> GroupedTemperatureCalibrationModel:
    fallback_temperature = _best_temperature(examples)
    by_group: dict[str, list[GroupedCalibrationExample]] = {}
    for example in examples:
        group_key = _group_key(example.category, example.evidence_bucket)
        by_group.setdefault(group_key, []).append(example)
    group_temperatures = {}
    group_sample_counts = {}
    for group_key, group_examples in by_group.items():
        if len(group_examples) < min_group_rows:
            continue
        group_temperatures[group_key] = _best_temperature(group_examples)
        group_sample_counts[group_key] = len(group_examples)
    return GroupedTemperatureCalibrationModel(
        fallback_temperature=fallback_temperature,
        group_temperatures=group_temperatures,
        group_sample_counts=group_sample_counts,
        shrinkage=shrinkage,
    )


def _best_temperature(examples: list[GroupedCalibrationExample]) -> float:
    if not examples:
        return 1.0
    candidates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0]
    return min(candidates, key=lambda temperature: _brier(examples, temperature))


def _brier(examples: list[GroupedCalibrationExample], temperature: float) -> float:
    return sum(
        (_apply_temperature(example.probability, temperature) - example.label) ** 2
        for example in examples
    ) / len(examples)


def _apply_temperature(probability: float, temperature: float) -> float:
    clipped = min(0.999999, max(0.000001, float(probability)))
    logit = math.log(clipped / (1.0 - clipped)) / temperature
    return min(0.99, max(0.01, round(1.0 / (1.0 + math.exp(-logit)), 6)))


def _group_key(category: str, evidence_bucket: str) -> str:
    return f"{category}::{evidence_bucket}"
