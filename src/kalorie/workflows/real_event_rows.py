import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from kalorie.data_grepping.event_scenarios import EventScenarioCatalog
from kalorie.domain.models import MentionMarketContract
from kalorie.io.public_documents import PublicDocumentManifest
from kalorie.io.transcript_corpus import TranscriptRecord
from kalorie.ml.datasets import HistoricalTrainingExample
from kalorie.ml.embeddings import EmbeddingProvider
from kalorie.ml.real_training_data import (
    build_examples_from_transcript_records,
    source_document_from_text_file,
)
from kalorie.workflows.event_dossiers import (
    phrase_variants_by_event,
    scenario_texts_by_event,
)
from kalorie.workflows.kalshi_event_pack import KalshiEventPackCandidate
from kalorie.workflows.models import WorkflowBaseModel, WorkflowSkippedRecord


class RealEventPackRowsResult(WorkflowBaseModel):
    examples: list[HistoricalTrainingExample] = Field(default_factory=list)
    skipped_records: list[WorkflowSkippedRecord] = Field(default_factory=list)


def build_real_event_pack_training_rows(
    pack_dir: Path,
    *,
    event_dossiers: list[EventScenarioCatalog] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> RealEventPackRowsResult:
    examples: list[HistoricalTrainingExample] = []
    skipped: list[WorkflowSkippedRecord] = []
    event_scenario_texts = (
        scenario_texts_by_event(event_dossiers)
        if event_dossiers
        else None
    )
    event_template_phrases = (
        phrase_variants_by_event(event_dossiers)
        if event_dossiers
        else None
    )

    for event_dir in sorted(path for path in pack_dir.iterdir() if path.is_dir()):
        event_path = event_dir / "event.json"
        if not event_path.exists():
            continue
        contracts_path = _first_existing(
            event_dir / "contracts.json",
            event_dir / "contracts-preclose.json",
        )
        contracts = _load_contracts(contracts_path)
        candidate = _load_candidate(event_path, event_dir=event_dir, contracts=contracts)
        transcript_path = _transcript_path_for_event(event_dir)
        if not transcript_path.exists():
            skipped.append(_skip(candidate, transcript_path, "missing_transcript"))
            continue
        if not contracts:
            skipped.append(_skip(candidate, contracts_path, "missing_contracts"))
            continue
        snapshots_path = _first_existing(
            event_dir / "snapshots.json",
            event_dir / "preclose_snapshots.json",
        )
        snapshots = _load_snapshots(snapshots_path)
        contracts = _contracts_with_snapshot_prices(
            contracts=contracts,
            snapshots=snapshots,
            candidate=candidate,
        )
        if not contracts:
            skipped.append(_skip(candidate, snapshots_path, "missing_t10_snapshots"))
            continue
        manifests = _load_event_manifests(event_dir)
        manifests = _manifests_before_cutoff(manifests, cutoff=candidate.call_start_at - timedelta(minutes=10))
        if not manifests:
            skipped.append(_skip(candidate, event_dir, "missing_evidence"))
            continue

        documents = [
            source_document_from_text_file(
                path=Path(manifest.raw_path),
                company_symbol=manifest.company_symbol,
                fiscal_year=manifest.fiscal_year,
                fiscal_quarter=manifest.fiscal_quarter,
                published_at=manifest.published_at,
                document_type=manifest.source_type,
            )
            for manifest in manifests
        ]
        record = TranscriptRecord(
            company_name=candidate.company_name,
            company_symbol=candidate.company_symbol,
            fiscal_year=candidate.fiscal_year,
            fiscal_quarter=candidate.fiscal_quarter,
            path=transcript_path,
            call_start_at=candidate.call_start_at,
            call_time_source="event_pack",
        )
        examples.extend(
            build_examples_from_transcript_records(
                records=[record],
                documents_by_period={
                    (
                        candidate.company_symbol,
                        candidate.fiscal_year,
                        candidate.fiscal_quarter,
                    ): documents
                },
                contracts=contracts,
                template_phrases_by_event=event_template_phrases,
                scenario_texts_by_event=event_scenario_texts,
                embedding_provider=embedding_provider,
            )
        )

    return RealEventPackRowsResult(examples=examples, skipped_records=skipped)


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _load_candidate(
    event_path: Path,
    *,
    event_dir: Path,
    contracts: list[MentionMarketContract],
) -> KalshiEventPackCandidate:
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    try:
        return KalshiEventPackCandidate.model_validate(payload)
    except ValidationError:
        transcript_manifest = PublicDocumentManifest.model_validate(
            json.loads((event_dir / "transcript_manifest.json").read_text(encoding="utf-8"))
        )
        observed_at = min(contract.observed_at for contract in contracts)
        return KalshiEventPackCandidate(
            event_ticker=str(
                payload.get("event", {}).get("event_ticker")
                or contracts[0].event_ticker
            ),
            company_symbol=transcript_manifest.company_symbol,
            company_name=_company_name_from_contract(contracts[0]),
            fiscal_year=transcript_manifest.fiscal_year,
            fiscal_quarter=transcript_manifest.fiscal_quarter,
            call_start_at=observed_at + timedelta(minutes=10),
        )


def _transcript_path_for_event(event_dir: Path) -> Path:
    normalized_path = event_dir / "transcript" / "transcript.txt"
    if normalized_path.exists():
        return normalized_path
    manifest_path = event_dir / "transcript_manifest.json"
    if not manifest_path.exists():
        return normalized_path
    manifest = PublicDocumentManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    return Path(manifest.raw_path)


def _load_contracts(path: Path) -> list[MentionMarketContract]:
    if not path.exists():
        return []
    return [
        MentionMarketContract.model_validate(row)
        for row in json.loads(path.read_text(encoding="utf-8"))
    ]


def _load_manifests(path: Path) -> list[PublicDocumentManifest]:
    if not path.exists():
        return []
    return [
        PublicDocumentManifest.model_validate(row)
        for row in json.loads(path.read_text(encoding="utf-8"))
    ]


def _load_event_manifests(event_dir: Path) -> list[PublicDocumentManifest]:
    manifests = _load_manifests(event_dir / "evidence-manifests.json")
    if manifests:
        return manifests
    raw_manifests = []
    for filename in ("sec_manifests.json", "news_manifests_defeatbeta.json"):
        raw_manifests.extend(_load_manifests(event_dir / filename))
    return raw_manifests


def _load_snapshots(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _contracts_with_snapshot_prices(
    *,
    contracts: list[MentionMarketContract],
    snapshots: list[dict[str, Any]],
    candidate: KalshiEventPackCandidate,
) -> list[MentionMarketContract]:
    snapshots_by_market = {str(snapshot.get("market_id")): snapshot for snapshot in snapshots}
    updated = []
    for contract in contracts:
        snapshot = snapshots_by_market.get(contract.market_id)
        if snapshot is None:
            continue
        cutoff = datetime.fromisoformat(
            str(snapshot["snapshot_target_time"]).replace("Z", "+00:00")
        )
        candle_ts = int(snapshot["candle_end_ts"])
        if candle_ts > int(cutoff.timestamp()):
            continue
        observed_at = datetime.fromtimestamp(candle_ts, tz=UTC)
        updated.append(
            contract.model_copy(
                update={
                    "yes_bid": _snapshot_price(
                        snapshot.get("yes_bid", snapshot.get("preclose_yes_bid"))
                    ),
                    "yes_ask": _snapshot_price(
                        snapshot.get("yes_ask", snapshot.get("preclose_yes_ask"))
                    ),
                    "observed_at": observed_at,
                    "event_ticker": candidate.event_ticker,
                }
            )
        )
    return updated


def _manifests_before_cutoff(
    manifests: list[PublicDocumentManifest],
    *,
    cutoff: datetime,
) -> list[PublicDocumentManifest]:
    return [
        manifest
        for manifest in manifests
        if _aware_datetime(manifest.published_at) <= _aware_datetime(cutoff)
    ]


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _company_name_from_contract(contract: MentionMarketContract) -> str:
    prefix = contract.title.split(" say during", maxsplit=1)[0]
    return prefix.removeprefix("What will ").strip() or contract.target_phrase.phrase


def _snapshot_price(value: Any) -> Decimal:
    decimal = Decimal(str(value))
    if decimal > 1:
        decimal = decimal / Decimal("100")
    return decimal.quantize(Decimal("0.01"))


def _skip(candidate: KalshiEventPackCandidate, path: Path, reason: str) -> WorkflowSkippedRecord:
    return WorkflowSkippedRecord(
        company_symbol=candidate.company_symbol,
        company_name=candidate.company_name,
        fiscal_year=candidate.fiscal_year,
        fiscal_quarter=candidate.fiscal_quarter,
        path=path,
        reason=reason,
    )
