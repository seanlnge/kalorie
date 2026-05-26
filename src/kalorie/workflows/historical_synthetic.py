from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import Field

from kalorie.data_grepping.event_scenarios import EventScenarioCatalog
from kalorie.io.public_documents import PublicDocumentManifest
from kalorie.io.transcript_corpus import (
    TRANSCRIPT_FILENAME,
    TranscriptRecord,
    scan_transcript_corpus,
)
from kalorie.ml.datasets import HistoricalTrainingExample
from kalorie.ml.embeddings import EmbeddingProvider
from kalorie.ml.real_training_data import (
    build_synthetic_phrase_examples_from_transcript_records,
    source_document_from_text_file,
)
from kalorie.workflows.event_dossiers import (
    phrase_variants_by_event,
    scenario_texts_by_event,
)
from kalorie.workflows.models import (
    PhraseCatalog,
    TranscriptInventory,
    TranscriptInventoryRow,
    WorkflowBaseModel,
    WorkflowSkippedRecord,
)


class HistoricalSyntheticRowsResult(WorkflowBaseModel):
    examples: list[HistoricalTrainingExample] = Field(default_factory=list)
    skipped_records: list[WorkflowSkippedRecord] = Field(default_factory=list)


def build_transcript_inventory(transcript_root: Path) -> TranscriptInventory:
    records = scan_transcript_corpus(transcript_root)
    rows = [
        TranscriptInventoryRow(
            company_symbol=record.company_symbol,
            company_name=record.company_name,
            fiscal_year=record.fiscal_year,
            fiscal_quarter=record.fiscal_quarter,
            transcript_path=record.path,
            estimated_call_time=record.call_start_at
            or _estimated_call_time(record.fiscal_year, record.fiscal_quarter),
            call_time_source=record.call_time_source or "estimated_fiscal_period_plus_50d",
        )
        for record in records
    ]
    known_paths = {record.path for record in records}
    skipped = []
    for path in sorted(transcript_root.glob("*/*")):
        if not path.is_file() or path in known_paths:
            continue
        if not TRANSCRIPT_FILENAME.match(path.name):
            skipped.append(
                WorkflowSkippedRecord(
                    company_name=path.parent.name,
                    path=path,
                    reason="unsupported_filename",
                    detail="Only YYYY_QN_SYMBOL_processed.txt transcript files are supported",
                )
            )

    return TranscriptInventory(rows=rows, skipped_records=skipped)


def build_historical_synthetic_rows(
    *,
    inventory: TranscriptInventory,
    manifests: list[PublicDocumentManifest],
    phrase_catalog: PhraseCatalog,
    event_dossiers: list[EventScenarioCatalog] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    record_concurrency: int = 1,
) -> HistoricalSyntheticRowsResult:
    if record_concurrency < 1:
        raise ValueError("record_concurrency must be at least 1")
    documents_by_period = _documents_by_period_from_manifests(manifests)
    phrases_by_period = _phrases_by_period_from_catalog(phrase_catalog)
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
    skipped: list[WorkflowSkippedRecord] = []
    jobs = []

    for index, row in enumerate(inventory.rows):
        period_key = (row.company_symbol, row.fiscal_year, row.fiscal_quarter)
        period_documents = documents_by_period.get(period_key, [])
        period_phrases = phrases_by_period.get(period_key, [])
        if not period_documents:
            skipped.append(_skip_for_row(row, "missing_evidence"))
        if not period_phrases:
            skipped.append(_skip_for_row(row, "missing_phrases"))
        if not period_documents or not period_phrases:
            continue

        jobs.append((index, row, period_key, period_documents, period_phrases))

    def build_row_examples(job) -> tuple[int, list[HistoricalTrainingExample]]:
        index, row, period_key, period_documents, period_phrases = job
        record = _inventory_row_to_transcript_record(row)
        row_examples = build_synthetic_phrase_examples_from_transcript_records(
            records=[record],
            documents_by_period={period_key: period_documents},
            target_phrases=[],
            company_target_phrases={row.company_symbol: period_phrases},
            template_phrases_by_event=event_template_phrases,
            scenario_texts_by_event=event_scenario_texts,
            embedding_provider=embedding_provider,
        )
        return index, row_examples

    examples_by_index: dict[int, list[HistoricalTrainingExample]] = {}
    if record_concurrency == 1:
        for job in jobs:
            index, row_examples = build_row_examples(job)
            examples_by_index[index] = row_examples
    else:
        with ThreadPoolExecutor(max_workers=record_concurrency) as executor:
            futures = [executor.submit(build_row_examples, job) for job in jobs]
            for future in as_completed(futures):
                index, row_examples = future.result()
                examples_by_index[index] = row_examples
    examples = [
        example
        for index in sorted(examples_by_index)
        for example in examples_by_index[index]
    ]

    return HistoricalSyntheticRowsResult(examples=examples, skipped_records=skipped)


def _documents_by_period_from_manifests(manifests: list[PublicDocumentManifest]):
    documents_by_period = {}
    for manifest in manifests:
        key = (manifest.company_symbol, manifest.fiscal_year, manifest.fiscal_quarter)
        documents_by_period.setdefault(key, []).append(
            source_document_from_text_file(
                path=Path(manifest.raw_path),
                company_symbol=manifest.company_symbol,
                fiscal_year=manifest.fiscal_year,
                fiscal_quarter=manifest.fiscal_quarter,
                published_at=manifest.published_at,
                document_type=manifest.source_type,
            )
        )
    return documents_by_period


def _phrases_by_period_from_catalog(phrase_catalog: PhraseCatalog):
    phrases_by_period = {}
    for entry in phrase_catalog.entries:
        key = (entry.company_symbol, entry.fiscal_year, entry.fiscal_quarter)
        phrases_by_period.setdefault(key, []).append(entry.phrase)
    return {
        key: list(dict.fromkeys(phrases))
        for key, phrases in phrases_by_period.items()
    }


def _inventory_row_to_transcript_record(row: TranscriptInventoryRow) -> TranscriptRecord:
    return TranscriptRecord(
        company_name=row.company_name,
        company_symbol=row.company_symbol,
        fiscal_year=row.fiscal_year,
        fiscal_quarter=row.fiscal_quarter,
        path=row.transcript_path,
        call_start_at=row.estimated_call_time,
        call_time_source=row.call_time_source,
    )


def _estimated_call_time(fiscal_year: int, fiscal_quarter: int) -> datetime:
    period_end_month = fiscal_quarter * 3
    quarter_end_day = 31 if period_end_month in {3, 12} else 30
    return datetime(
        year=fiscal_year,
        month=period_end_month,
        day=quarter_end_day,
        tzinfo=UTC,
    ) + timedelta(days=50)


def _skip_for_row(row: TranscriptInventoryRow, reason: str) -> WorkflowSkippedRecord:
    return WorkflowSkippedRecord(
        company_symbol=row.company_symbol,
        company_name=row.company_name,
        fiscal_year=row.fiscal_year,
        fiscal_quarter=row.fiscal_quarter,
        path=row.transcript_path,
        reason=reason,
    )
