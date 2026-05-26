from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from kalorie.domain.models import KalorieModel


class WorkflowBaseModel(KalorieModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class HistoricalSyntheticWorkflowConfig(WorkflowBaseModel):
    transcript_root: Path = Path("data/earnings_call_transcripts")
    output_root: Path = Path("artifacts/model1/workflows")
    dataset_output_root: Path = Path("artifacts/model1/datasets")
    sec_request_budget: int = Field(default=80, ge=0)
    phrase_target_min: int = Field(default=6, ge=1)
    phrase_target_max: int = Field(default=12, ge=1)
    openai_enabled: bool = True
    llm_model: str = "gpt-4o-mini"
    evidence_cutoff_lead_minutes: int = Field(default=10, ge=0)

    @field_validator("phrase_target_max")
    @classmethod
    def max_must_cover_min(cls, value: int, info) -> int:
        minimum = info.data.get("phrase_target_min")
        if minimum is not None and value < minimum:
            raise ValueError("phrase_target_max must be greater than or equal to phrase_target_min")
        return value


class RealKalshiEventPackConfig(WorkflowBaseModel):
    output_root: Path = Path("artifacts/model1/event-packs")
    snapshot_lead_minutes: int = Field(default=10, ge=0)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class TranscriptInventoryRow(WorkflowBaseModel):
    company_symbol: str
    company_name: str
    fiscal_year: int
    fiscal_quarter: int = Field(ge=1, le=4)
    transcript_path: Path
    estimated_call_time: datetime | None = None
    call_time_source: str | None = None
    diagnostics: list[str] = Field(default_factory=list)

    @field_validator("company_symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.upper()


class WorkflowSkippedRecord(WorkflowBaseModel):
    company_symbol: str | None = None
    company_name: str | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = Field(default=None, ge=1, le=4)
    path: Path | None = None
    reason: str
    detail: str | None = None


class TranscriptInventory(WorkflowBaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    rows: list[TranscriptInventoryRow] = Field(default_factory=list)
    skipped_records: list[WorkflowSkippedRecord] = Field(default_factory=list)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_records)


class SecRequestPlanRow(WorkflowBaseModel):
    company_symbol: str
    company_name: str
    transcript_count: int = Field(ge=0)
    cached_cik: str | None = None
    needs_mapping_request: bool = False
    needs_filing_query: bool = True
    projected_requests: int = Field(ge=0)

    @field_validator("company_symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.upper()


class SecRequestPlan(WorkflowBaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    request_budget: int = Field(ge=0)
    projected_requests: int = Field(ge=0)
    companies: list[SecRequestPlanRow] = Field(default_factory=list)

    @property
    def within_budget(self) -> bool:
        return self.projected_requests <= self.request_budget


class EvidenceCollectionSummary(WorkflowBaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    sec_requests_used: int = Field(default=0, ge=0)
    sec_budget_remaining: int = Field(default=0, ge=0)
    manifest_count: int = Field(default=0, ge=0)
    skipped_records: list[WorkflowSkippedRecord] = Field(default_factory=list)


class EvidenceCollectionResult(WorkflowBaseModel):
    manifests: list[object] = Field(default_factory=list)
    summary: EvidenceCollectionSummary


class PhraseCatalogEntry(WorkflowBaseModel):
    company_symbol: str
    fiscal_year: int
    fiscal_quarter: int = Field(ge=1, le=4)
    phrase: str
    label: Literal["present", "absent"]
    source: Literal["deterministic", "openai", "manual"] = "deterministic"
    match_count: int = Field(default=0, ge=0)

    @field_validator("company_symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.upper()


class PhraseCatalog(WorkflowBaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    entries: list[PhraseCatalogEntry] = Field(default_factory=list)
    skipped_records: list[WorkflowSkippedRecord] = Field(default_factory=list)


class WorkflowSummary(WorkflowBaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    workflow_name: str
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    row_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)
    diagnostics: list[str] = Field(default_factory=list)


class WorkflowVerificationReport(WorkflowBaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
