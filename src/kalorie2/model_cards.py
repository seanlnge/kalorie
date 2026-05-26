from __future__ import annotations

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
    policy: Literal["all", "no_only", "yes_only"]
    margin: float = Field(ge=0.0)
    metrics: dict[str, MetricValue]
    notes: str | None = None


class ModelCard(ModelCardBase):
    schema_version: str = "1.0"
    model_name: str
    model_version: int | None = None
    model_type: str
    default_execution_policy: Literal["all", "no_only", "yes_only"]
    default_margin: float = Field(ge=0.0)
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
