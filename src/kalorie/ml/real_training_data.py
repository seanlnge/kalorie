import re
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Lock

from kalorie.domain.models import DocumentChunk, MentionMarketContract, SourceDocument, TargetPhrase
from kalorie.io.documents import chunk_text, content_hash, normalize_text
from kalorie.io.transcript_corpus import TranscriptRecord, prior_transcript_records
from kalorie.market.markets import (
    MentionMarketParseError,
    normalize_phrase,
    parse_mention_market_title,
)
from kalorie.ml.datasets import HistoricalTrainingExample, build_historical_training_examples
from kalorie.ml.embeddings import EmbeddingProvider
from kalorie.ml.features import extract_transcript_recurrence_feature_vectors
from kalorie.ml.labeling import label_document_chunks

PeriodKey = tuple[str, int, int]
# For each event snapshot, only allow evidence available strictly before
# call start. We keep a short buffer to account for publication jitter.
EVIDENCE_CUTOFF_LEAD = timedelta(minutes=10)
BASELINE_EVIDENCE_MAX_POST_CALL_LAG = timedelta(days=2)
DEFAULT_SYNTHETIC_TARGET_PHRASES = [
    "revenue",
    "margin",
    "guidance",
    "traffic",
    "automation",
    "ai",
    "tariff",
    "inflation",
    "demand",
    "pricing",
    "cloud",
    "inventory",
    "capex",
    # Kalshi-style phrase coverage for non-generic mention markets.
    "openai",
    "omnichannel",
    "salmon",
    "sweet potato",
    "auv",
]


def build_examples_from_transcript_records(
    *,
    records: list[TranscriptRecord],
    documents_by_period: dict[PeriodKey, list[SourceDocument]],
    contracts: list[MentionMarketContract],
    min_examples: int | None = None,
    template_phrases_by_target: dict[str, list[str]] | None = None,
    template_phrases_by_event: dict[str, dict[str, list[str]]] | None = None,
    scenario_texts_by_event: dict[str, list[str]] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    record_concurrency: int = 1,
) -> list[HistoricalTrainingExample]:
    if record_concurrency < 1:
        raise ValueError("record_concurrency must be at least 1")
    examples: list[HistoricalTrainingExample] = []
    chunk_cache = _ChunkCache()
    if record_concurrency > 1:
        return _build_examples_parallel(
            records=records,
            documents_by_period=documents_by_period,
            contracts=contracts,
            min_examples=min_examples,
            template_phrases_by_target=template_phrases_by_target,
            template_phrases_by_event=template_phrases_by_event,
            scenario_texts_by_event=scenario_texts_by_event,
            embedding_provider=embedding_provider,
            progress_callback=progress_callback,
            record_concurrency=record_concurrency,
            synthetic_target_phrases=None,
            company_target_phrases=None,
            chunk_cache=chunk_cache,
        )
    for index, record in enumerate(records, start=1):
        if progress_callback is not None:
            progress_callback(index, len(examples))
        period_key = (record.company_symbol, record.fiscal_year, record.fiscal_quarter)
        evidence_cutoff, evidence_documents, evidence_document_roles = (
            _evidence_selection_for_record(record, documents_by_period.get(period_key, []))
        )
        if not evidence_documents:
            continue
        event_contracts = contracts_for_company(contracts, record.company_symbol)
        if not event_contracts:
            continue
        transcript_chunks = _chunks_from_text_file(
            path=record.path,
            document_id=f"{record.company_symbol}-{record.fiscal_year}-Q{record.fiscal_quarter}-TRANSCRIPT",
            chunk_cache=chunk_cache,
        )
        evidence_chunks = []
        for document in evidence_documents:
            evidence_chunks.extend(
                _chunks_from_text_file(
                    path=Path(document.source_path),
                    document_id=document.source_id,
                    chunk_cache=chunk_cache,
                )
            )
        transcript_recurrence_features_by_target = _transcript_recurrence_features_for_record(
            record=record,
            records=records,
            targets=[contract.target_phrase for contract in event_contracts],
            chunk_cache=chunk_cache,
        )
        event_id = _event_id_for_record(record)
        record_template_phrases_by_target = _template_phrases_for_event(
            event_id=event_id,
            template_phrases_by_event=template_phrases_by_event,
            template_phrases_by_target=template_phrases_by_target,
        )
        for example in build_historical_training_examples(
            company_symbol=record.company_symbol,
            fiscal_year=record.fiscal_year,
            fiscal_quarter=record.fiscal_quarter,
            evidence_cutoff=evidence_cutoff,
            contracts=event_contracts,
            evidence_documents=evidence_documents,
            evidence_chunks=evidence_chunks,
            transcript_chunks=transcript_chunks,
            template_phrases_by_target=record_template_phrases_by_target,
            scenario_texts=scenario_texts_by_event.get(event_id, [])
            if scenario_texts_by_event
            else None,
            embedding_provider=embedding_provider,
            evidence_document_roles=evidence_document_roles,
            transcript_recurrence_features_by_target=transcript_recurrence_features_by_target,
        ):
            examples.append(example)
            if min_examples is not None and len(examples) >= min_examples:
                return examples
    return examples


def build_synthetic_phrase_examples_from_transcript_records(
    *,
    records: list[TranscriptRecord],
    documents_by_period: dict[PeriodKey, list[SourceDocument]],
    target_phrases: list[str],
    min_examples: int | None = None,
    template_phrases_by_target: dict[str, list[str]] | None = None,
    template_phrases_by_event: dict[str, dict[str, list[str]]] | None = None,
    scenario_texts_by_event: dict[str, list[str]] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    record_concurrency: int = 1,
    company_target_phrases: dict[str, list[str]] | None = None,
) -> list[HistoricalTrainingExample]:
    if record_concurrency < 1:
        raise ValueError("record_concurrency must be at least 1")
    examples: list[HistoricalTrainingExample] = []
    normalized_phrases = list(dict.fromkeys(normalize_phrase(phrase) for phrase in target_phrases))
    normalized_company_target_phrases = {
        symbol.upper(): list(dict.fromkeys(normalize_phrase(value) for value in values))
        for symbol, values in (company_target_phrases or {}).items()
    }
    chunk_cache = _ChunkCache()
    if record_concurrency > 1:
        return _build_examples_parallel(
            records=records,
            documents_by_period=documents_by_period,
            contracts=[],
            min_examples=min_examples,
            template_phrases_by_target=template_phrases_by_target,
            template_phrases_by_event=template_phrases_by_event,
            scenario_texts_by_event=scenario_texts_by_event,
            embedding_provider=embedding_provider,
            progress_callback=progress_callback,
            record_concurrency=record_concurrency,
            synthetic_target_phrases=normalized_phrases,
            company_target_phrases=normalized_company_target_phrases,
            chunk_cache=chunk_cache,
        )
    for index, record in enumerate(records, start=1):
        if progress_callback is not None:
            progress_callback(index, len(examples))
        period_key = (record.company_symbol, record.fiscal_year, record.fiscal_quarter)
        evidence_cutoff, evidence_documents, evidence_document_roles = (
            _evidence_selection_for_record(record, documents_by_period.get(period_key, []))
        )
        if not evidence_documents:
            continue
        phrases_for_record = _phrases_for_record(
            base_phrases=normalized_phrases,
            company_symbol=record.company_symbol,
            company_target_phrases=normalized_company_target_phrases,
        )
        contracts = [
            _synthetic_contract(
                company_symbol=record.company_symbol,
                fiscal_year=record.fiscal_year,
                fiscal_quarter=record.fiscal_quarter,
                phrase=phrase,
                observed_at=evidence_cutoff,
            )
            for phrase in phrases_for_record
        ]
        transcript_chunks = _chunks_from_text_file(
            path=record.path,
            document_id=f"{record.company_symbol}-{record.fiscal_year}-Q{record.fiscal_quarter}-TRANSCRIPT",
            chunk_cache=chunk_cache,
        )
        evidence_chunks = []
        for document in evidence_documents:
            evidence_chunks.extend(
                _chunks_from_text_file(
                    path=Path(document.source_path),
                    document_id=document.source_id,
                    chunk_cache=chunk_cache,
                )
            )
        transcript_recurrence_features_by_target = _transcript_recurrence_features_for_record(
            record=record,
            records=records,
            targets=[contract.target_phrase for contract in contracts],
            chunk_cache=chunk_cache,
        )
        event_id = _event_id_for_record(record)
        record_template_phrases_by_target = _template_phrases_for_event(
            event_id=event_id,
            template_phrases_by_event=template_phrases_by_event,
            template_phrases_by_target=template_phrases_by_target,
        )
        for example in build_historical_training_examples(
            company_symbol=record.company_symbol,
            fiscal_year=record.fiscal_year,
            fiscal_quarter=record.fiscal_quarter,
            evidence_cutoff=evidence_cutoff,
            contracts=contracts,
            evidence_documents=evidence_documents,
            evidence_chunks=evidence_chunks,
            transcript_chunks=transcript_chunks,
            template_phrases_by_target=record_template_phrases_by_target,
            scenario_texts=scenario_texts_by_event.get(event_id, [])
            if scenario_texts_by_event
            else None,
            embedding_provider=embedding_provider,
            evidence_document_roles=evidence_document_roles,
            transcript_recurrence_features_by_target=transcript_recurrence_features_by_target,
        ):
            examples.append(example)
            if min_examples is not None and len(examples) >= min_examples:
                return examples
    return examples


def _build_examples_parallel(
    *,
    records: list[TranscriptRecord],
    documents_by_period: dict[PeriodKey, list[SourceDocument]],
    contracts: list[MentionMarketContract],
    min_examples: int | None,
    template_phrases_by_target: dict[str, list[str]] | None,
    template_phrases_by_event: dict[str, dict[str, list[str]]] | None,
    scenario_texts_by_event: dict[str, list[str]] | None,
    embedding_provider: EmbeddingProvider | None,
    progress_callback: Callable[[int, int], None] | None,
    record_concurrency: int,
    synthetic_target_phrases: list[str] | None,
    company_target_phrases: dict[str, list[str]] | None,
    chunk_cache: "_ChunkCache",
) -> list[HistoricalTrainingExample]:
    eligible_records: list[tuple[int, TranscriptRecord]] = []
    skipped_without_evidence = 0
    for index, record in enumerate(records, start=1):
        period_key = (record.company_symbol, record.fiscal_year, record.fiscal_quarter)
        period_documents = documents_by_period.get(period_key, [])
        _, evidence_documents, _ = _evidence_selection_for_record(record, period_documents)
        has_evidence = bool(evidence_documents)
        if has_evidence:
            eligible_records.append((index, record))
        else:
            skipped_without_evidence += 1

    if progress_callback is not None and skipped_without_evidence:
        progress_callback(skipped_without_evidence, 0)

    def worker(record: TranscriptRecord) -> list[HistoricalTrainingExample]:
        return _examples_for_record(
            record=record,
            records=records,
            documents_by_period=documents_by_period,
            contracts=contracts,
            template_phrases_by_target=template_phrases_by_target,
            template_phrases_by_event=template_phrases_by_event,
            scenario_texts_by_event=scenario_texts_by_event,
            embedding_provider=embedding_provider,
            synthetic_target_phrases=synthetic_target_phrases,
            company_target_phrases=company_target_phrases,
            chunk_cache=chunk_cache,
        )

    by_index: dict[int, list[HistoricalTrainingExample]] = {}
    done = 0
    generated = 0
    with ThreadPoolExecutor(max_workers=record_concurrency) as executor:
        future_to_index = {
            executor.submit(worker, record): index
            for index, record in eligible_records
        }
        pending = set(future_to_index)
        while pending:
            done_futures, pending = wait(pending, timeout=10.0, return_when=FIRST_COMPLETED)
            if not done_futures:
                if progress_callback is not None:
                    progress_callback(skipped_without_evidence + done, generated)
                continue
            for future in done_futures:
                index = future_to_index[future]
                record_examples = future.result()
                by_index[index] = record_examples
                done += 1
                generated += len(record_examples)
                if progress_callback is not None:
                    progress_callback(skipped_without_evidence + done, generated)

    examples = [example for index in sorted(by_index) for example in by_index[index]]
    if min_examples is not None:
        return examples[:min_examples]
    return examples


def _examples_for_record(
    *,
    record: TranscriptRecord,
    records: list[TranscriptRecord],
    documents_by_period: dict[PeriodKey, list[SourceDocument]],
    contracts: list[MentionMarketContract],
    template_phrases_by_target: dict[str, list[str]] | None,
    template_phrases_by_event: dict[str, dict[str, list[str]]] | None,
    scenario_texts_by_event: dict[str, list[str]] | None,
    embedding_provider: EmbeddingProvider | None,
    synthetic_target_phrases: list[str] | None,
    company_target_phrases: dict[str, list[str]] | None,
    chunk_cache: "_ChunkCache",
) -> list[HistoricalTrainingExample]:
    period_key = (record.company_symbol, record.fiscal_year, record.fiscal_quarter)
    evidence_cutoff, evidence_documents, evidence_document_roles = (
        _evidence_selection_for_record(record, documents_by_period.get(period_key, []))
    )
    if not evidence_documents:
        return []
    transcript_chunks = _chunks_from_text_file(
        path=record.path,
        document_id=f"{record.company_symbol}-{record.fiscal_year}-Q{record.fiscal_quarter}-TRANSCRIPT",
        chunk_cache=chunk_cache,
    )
    evidence_chunks = []
    for document in evidence_documents:
        evidence_chunks.extend(
            _chunks_from_text_file(
                path=Path(document.source_path),
                document_id=document.source_id,
                chunk_cache=chunk_cache,
            )
        )
    if synthetic_target_phrases is None:
        event_contracts = contracts_for_company(contracts, record.company_symbol)
        if not event_contracts:
            return []
    else:
        phrases_for_record = _phrases_for_record(
            base_phrases=synthetic_target_phrases,
            company_symbol=record.company_symbol,
            company_target_phrases=company_target_phrases,
        )
        event_contracts = [
            _synthetic_contract(
                company_symbol=record.company_symbol,
                fiscal_year=record.fiscal_year,
                fiscal_quarter=record.fiscal_quarter,
                phrase=phrase,
                observed_at=evidence_cutoff,
            )
            for phrase in phrases_for_record
        ]
    transcript_recurrence_features_by_target = _transcript_recurrence_features_for_record(
        record=record,
        records=records,
        targets=[contract.target_phrase for contract in event_contracts],
        chunk_cache=chunk_cache,
    )
    event_id = _event_id_for_record(record)
    record_template_phrases_by_target = _template_phrases_for_event(
        event_id=event_id,
        template_phrases_by_event=template_phrases_by_event,
        template_phrases_by_target=template_phrases_by_target,
    )
    return build_historical_training_examples(
        company_symbol=record.company_symbol,
        fiscal_year=record.fiscal_year,
        fiscal_quarter=record.fiscal_quarter,
        evidence_cutoff=evidence_cutoff,
        contracts=event_contracts,
        evidence_documents=evidence_documents,
        evidence_chunks=evidence_chunks,
        transcript_chunks=transcript_chunks,
        template_phrases_by_target=record_template_phrases_by_target,
        scenario_texts=scenario_texts_by_event.get(event_id, [])
        if scenario_texts_by_event
        else None,
        embedding_provider=embedding_provider,
        evidence_document_roles=evidence_document_roles,
        transcript_recurrence_features_by_target=transcript_recurrence_features_by_target,
    )


def _template_phrases_for_event(
    *,
    event_id: str,
    template_phrases_by_event: dict[str, dict[str, list[str]]] | None,
    template_phrases_by_target: dict[str, list[str]] | None,
) -> dict[str, list[str]] | None:
    if template_phrases_by_event and event_id in template_phrases_by_event:
        return template_phrases_by_event[event_id]
    return template_phrases_by_target


def _transcript_recurrence_features_for_record(
    *,
    record: TranscriptRecord,
    records: list[TranscriptRecord],
    targets: list[TargetPhrase],
    chunk_cache: "_ChunkCache",
) -> dict[str, dict[str, float]]:
    prior_label_sets = []
    for prior_record in prior_transcript_records(records, record):
        prior_chunks = _chunks_from_text_file(
            path=prior_record.path,
            document_id=(
                f"{prior_record.company_symbol}-{prior_record.fiscal_year}"
                f"-Q{prior_record.fiscal_quarter}-TRANSCRIPT"
            ),
            chunk_cache=chunk_cache,
        )
        prior_label_sets.append(
            label_document_chunks(
                prior_chunks,
                targets,
                entity_scope="company_employee",
            )
        )
    return {
        feature_vector.target_phrase: feature_vector.features
        for feature_vector in extract_transcript_recurrence_feature_vectors(
            targets=targets,
            prior_label_sets=prior_label_sets,
        )
    }


def source_document_from_text_file(
    *,
    path: Path,
    company_symbol: str,
    fiscal_year: int,
    fiscal_quarter: int,
    published_at,
    document_type: str = "sec_ex_99_1_press_release",
) -> SourceDocument:
    text = normalize_text(path.read_text(encoding="utf-8"))
    digest = content_hash(text)
    return SourceDocument(
        source_id=f"{company_symbol.upper()}-{fiscal_year}-Q{fiscal_quarter}-{digest[:12]}",
        company_symbol=company_symbol,
        document_type=document_type,
        source_path=str(path),
        published_at=published_at,
        content_hash=digest,
    )


def contracts_for_company(
    contracts: list[MentionMarketContract],
    company_symbol: str,
) -> list[MentionMarketContract]:
    normalized_symbol = company_symbol.upper()
    matching_contracts = []
    for contract in contracts:
        try:
            market_company = parse_mention_market_title(contract.title).company_symbol
        except MentionMarketParseError:
            identifier_text = f"{contract.event_ticker} {contract.market_id}".upper()
            if not _identifier_mentions_symbol(identifier_text, normalized_symbol):
                continue
        else:
            if market_company != normalized_symbol:
                continue
        matching_contracts.append(contract)
    return matching_contracts


def _call_time_for_record(record: TranscriptRecord) -> datetime:
    if record.call_start_at is not None:
        call_start_at = record.call_start_at
        if call_start_at.tzinfo is None or call_start_at.tzinfo.utcoffset(call_start_at) is None:
            return call_start_at.replace(tzinfo=UTC)
        return call_start_at
    period_end_month = record.fiscal_quarter * 3
    quarter_end_day = 31 if period_end_month in {3, 12} else 30
    fiscal_period_end = datetime(
        year=record.fiscal_year,
        month=period_end_month,
        day=quarter_end_day,
        tzinfo=UTC,
    )
    # Use local transcript artifact timestamp as call-time proxy when available,
    # but clamp to a reasonable post-quarter window to prevent late-ingestion
    # artifacts from leaking future evidence into training examples.
    transcript_time = datetime.fromtimestamp(record.path.stat().st_mtime, tz=UTC)
    plausible_latest_call_time = fiscal_period_end + timedelta(days=120)
    return max(fiscal_period_end, min(transcript_time, plausible_latest_call_time))


def _evidence_cutoff_for_record(record: TranscriptRecord) -> datetime:
    return _call_time_for_record(record) - EVIDENCE_CUTOFF_LEAD


def _event_id_for_record(record: TranscriptRecord) -> str:
    if record.event_ticker:
        return record.event_ticker
    return f"{record.company_symbol}-{record.fiscal_year}-Q{record.fiscal_quarter}"


def _evidence_selection_for_record(
    record: TranscriptRecord,
    documents: list[SourceDocument],
) -> tuple[datetime, list[SourceDocument], dict[str, str]]:
    call_time = _call_time_for_record(record)
    evidence_cutoff = call_time - EVIDENCE_CUTOFF_LEAD
    selected_documents: list[SourceDocument] = []
    document_roles: dict[str, str] = {}
    for document in documents:
        role = _evidence_document_role(document)
        if role == "time_sensitive":
            if document.published_at > evidence_cutoff:
                continue
        elif document.published_at > call_time + BASELINE_EVIDENCE_MAX_POST_CALL_LAG:
            continue
        selected_documents.append(document)
        document_roles[document.source_id] = role
    return evidence_cutoff, selected_documents, document_roles


def _evidence_document_role(document: SourceDocument) -> str:
    if _is_time_sensitive_document(document):
        return "time_sensitive"
    return "event_baseline"


def _is_time_sensitive_document(document: SourceDocument) -> bool:
    document_type = document.document_type.lower()
    return document_type.startswith(
        (
            "news_article",
            "market_snapshot",
            "market_data",
            "reddit",
            "social",
            "analyst_preview",
        )
    )


def _synthetic_contract(
    *,
    company_symbol: str,
    fiscal_year: int,
    fiscal_quarter: int,
    phrase: str,
    observed_at,
) -> MentionMarketContract:
    normalized_symbol = company_symbol.upper()
    normalized = normalize_phrase(phrase)
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    if observed_at.tzinfo is None or observed_at.tzinfo.utcoffset(observed_at) is None:
        observed_at = observed_at.replace(tzinfo=UTC)
    return MentionMarketContract(
        venue="synthetic",
        market_id=f"{normalized_symbol}-{fiscal_year}-Q{fiscal_quarter}-{slug}",
        event_ticker=f"SYNTH-{normalized_symbol}-{fiscal_year}Q{fiscal_quarter}",
        title=f"Synthetic phrase target: {normalized}",
        rules_text=f'Synthetic phrase-presence label for "{normalized}".',
        target_phrase=TargetPhrase(phrase=normalized, normalized_phrase=normalized),
        yes_bid=Decimal("0.50"),
        yes_ask=Decimal("0.50"),
        observed_at=observed_at,
    )


def _identifier_mentions_symbol(identifier_text: str, symbol: str) -> bool:
    escaped_symbol = re.escape(symbol.upper())
    return bool(
        re.search(rf"(?<![A-Z0-9]){escaped_symbol}(?![A-Z0-9])", identifier_text)
        or re.search(rf"MENTION{escaped_symbol}(?![A-Z0-9])", identifier_text)
        or re.search(rf"MENTION{escaped_symbol}-", identifier_text)
    )


def _phrases_for_record(
    *,
    base_phrases: list[str],
    company_symbol: str,
    company_target_phrases: dict[str, list[str]] | None,
) -> list[str]:
    phrases = list(base_phrases)
    if company_target_phrases:
        phrases.extend(company_target_phrases.get(company_symbol.upper(), []))
    return list(dict.fromkeys(normalize_phrase(phrase) for phrase in phrases))


class _ChunkCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], list[DocumentChunk]] = {}
        self._lock = Lock()

    def read(self, *, path: Path, document_id: str) -> list[DocumentChunk]:
        key = (str(path.resolve()), document_id)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        text = normalize_text(path.read_text(encoding="utf-8"))
        chunks = [
            chunk.model_copy(update={"document_id": document_id})
            for chunk in chunk_text(text)
        ]
        with self._lock:
            existing = self._cache.setdefault(key, chunks)
        return existing


def _chunks_from_text_file(
    path: Path,
    document_id: str,
    *,
    chunk_cache: _ChunkCache | None = None,
) -> list[DocumentChunk]:
    if chunk_cache is not None:
        return chunk_cache.read(path=path, document_id=document_id)
    text = normalize_text(path.read_text(encoding="utf-8"))
    return [chunk.model_copy(update={"document_id": document_id}) for chunk in chunk_text(text)]
