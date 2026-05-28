from datetime import datetime
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kalorie2.models import HistoricalMentionMarketRow

_LABEL_ONLY_FIELDS = frozenset(
    {
        "final_outcome",
        "outcome",
        "outcome_label",
        "label",
        "settlement_ts",
    }
)


class PredictionEngineModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictionInputRow(HistoricalMentionMarketRow):
    @property
    def outcome_label(self) -> int:
        return 1 if self.final_outcome == "yes" else 0

    def to_inference_payload(self) -> dict:
        return self.model_dump(exclude=_LABEL_ONLY_FIELDS)


def prediction_row_key(row: PredictionInputRow) -> str:
    return "|".join(
        [
            row.event_ticker,
            row.market_ticker,
            row.snapshot_target_time.isoformat(),
        ]
    )


class MarketSnapshotFeatures(PredictionEngineModel):
    yes_bid: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    yes_ask: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    yes_mid: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    spread: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    candle_end_ts: int
    snapshot_target_time: datetime
    snapshot_staleness_seconds: int = Field(ge=0)

    @classmethod
    def from_row(cls, row: PredictionInputRow) -> "MarketSnapshotFeatures":
        spread = row.preclose_yes_ask - row.preclose_yes_bid
        return cls(
            yes_bid=row.preclose_yes_bid,
            yes_ask=row.preclose_yes_ask,
            yes_mid=row.preclose_yes_mid,
            spread=spread,
            candle_end_ts=row.candle_end_ts,
            snapshot_target_time=row.snapshot_target_time,
            snapshot_staleness_seconds=row.snapshot_staleness_seconds,
        )

    @model_validator(mode="after")
    def bid_ask_must_be_ordered(self) -> "MarketSnapshotFeatures":
        if self.yes_bid > self.yes_ask:
            raise ValueError("yes_bid must be less than or equal to yes_ask")
        expected_spread = self.yes_ask - self.yes_bid
        if self.spread != expected_spread:
            raise ValueError("spread must equal yes_ask - yes_bid")
        return self


class ArtifactRetentionPolicy(PredictionEngineModel):
    canonical_source_files: set[str] = Field(default_factory=set)
    protected_full_directory: Path = Path("artifacts/full")

    def validate_output_path(self, path: Path, *, artifact_kind: str) -> Path:
        normalized_path = Path(path)
        if not _is_artifacts_full_path(normalized_path, self.protected_full_directory):
            return normalized_path

        is_canonical_source = (
            artifact_kind == "canonical_source"
            and normalized_path.name in self.canonical_source_files
        )
        if is_canonical_source:
            return normalized_path

        raise ValueError(
            "artifacts/full is reserved for canonical source datasets; "
            f"refusing to write {artifact_kind!r} artifact to {normalized_path}"
        )


class PredictionRunConfig(PredictionEngineModel):
    run_id: str = Field(min_length=1)
    decision_time_column: str = Field(min_length=1)
    artifact_retention_policy: ArtifactRetentionPolicy

    def validate_output_path(self, path: Path, *, artifact_kind: str) -> Path:
        return self.artifact_retention_policy.validate_output_path(
            path,
            artifact_kind=artifact_kind,
        )


class PredictionRecord(PredictionEngineModel):
    market_ticker: str = Field(min_length=1)
    event_ticker: str = Field(min_length=1)
    probability: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    market_probability: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    feature_values: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)

    @field_validator("feature_values")
    @classmethod
    def feature_values_must_not_include_labels(cls, values: dict[str, float]) -> dict[str, float]:
        leaked = sorted(set(values) & _LABEL_ONLY_FIELDS)
        if leaked:
            raise ValueError(f"feature_values contain label-only fields: {', '.join(leaked)}")
        return values


def _is_artifacts_full_path(path: Path, protected_full_directory: Path) -> bool:
    normalized_parts = _lower_parts(path)
    protected_parts = _lower_parts(protected_full_directory)
    if len(normalized_parts) < len(protected_parts):
        return False
    return any(
        normalized_parts[index : index + len(protected_parts)] == protected_parts
        for index in range(0, len(normalized_parts) - len(protected_parts) + 1)
    )


def _lower_parts(path: Path) -> tuple[str, ...]:
    parts = []
    for part in path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part.lower())
    return tuple(parts)
