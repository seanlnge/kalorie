from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from kalorie.clients.sec_api import SecApiFiling, select_best_company_mapping
from kalorie.io.documents import content_hash
from kalorie.io.public_documents import PublicDocumentManifest, collect_public_document
from kalorie.ml.datasets import HistoricalTrainingExample
from kalorie.workflows.models import (
    EvidenceCollectionResult,
    EvidenceCollectionSummary,
    SecRequestPlan,
    SecRequestPlanRow,
    TranscriptInventory,
    TranscriptInventoryRow,
    WorkflowSkippedRecord,
)

DocumentCollector = Callable[..., PublicDocumentManifest]
EvidenceCheckpointWriter = Callable[[EvidenceCollectionResult, dict[str, str]], None]
YFinanceNewsCollector = Callable[..., list[Any]]


def plan_sec_requests(
    inventory: TranscriptInventory,
    *,
    cached_ciks: dict[str, str],
    request_budget: int,
) -> SecRequestPlan:
    company_rows = _group_inventory_by_company(inventory.rows)
    plan_rows = []
    projected_requests = 0
    for symbol, rows in sorted(company_rows.items()):
        cached_cik = cached_ciks.get(symbol)
        needs_mapping = cached_cik is None
        row_requests = (1 if needs_mapping else 0) + 1
        projected_requests += row_requests
        plan_rows.append(
            SecRequestPlanRow(
                company_symbol=symbol,
                company_name=rows[0].company_name,
                transcript_count=len(rows),
                cached_cik=cached_cik,
                needs_mapping_request=needs_mapping,
                projected_requests=row_requests,
            )
        )

    return SecRequestPlan(
        request_budget=request_budget,
        projected_requests=projected_requests,
        companies=plan_rows,
    )


def collect_sec_ex99_manifests(
    inventory: TranscriptInventory,
    *,
    sec_client: Any,
    cached_ciks: dict[str, str],
    request_budget: int,
    raw_dir: Path,
    http_client: httpx.Client | None = None,
    document_collector: DocumentCollector = collect_public_document,
    checkpoint_writer: EvidenceCheckpointWriter | None = None,
    max_document_workers: int = 1,
) -> EvidenceCollectionResult:
    plan = plan_sec_requests(inventory, cached_ciks=cached_ciks, request_budget=request_budget)

    client = http_client or httpx.Client(timeout=30.0)
    requests_used = 0
    manifests: list[PublicDocumentManifest] = []
    skipped: list[WorkflowSkippedRecord] = []
    rows_by_company = _group_inventory_by_company(inventory.rows)

    def result_snapshot() -> EvidenceCollectionResult:
        merged = merge_document_manifests(manifests)
        return EvidenceCollectionResult(
            manifests=merged,
            summary=EvidenceCollectionSummary(
                sec_requests_used=requests_used,
                sec_budget_remaining=max(request_budget - requests_used, 0),
                manifest_count=len(merged),
                skipped_records=list(skipped),
            ),
        )

    def checkpoint() -> None:
        if checkpoint_writer is not None:
            checkpoint_writer(result_snapshot(), cached_ciks)

    try:
        for plan_row in plan.companies:
            symbol = plan_row.company_symbol
            cik = plan_row.cached_cik
            requests_needed = (1 if cik is None else 0) + 1
            if requests_used + requests_needed > request_budget:
                skipped.append(
                    WorkflowSkippedRecord(
                        company_symbol=symbol,
                        company_name=plan_row.company_name,
                        reason="sec_budget_exhausted",
                        detail=(
                            f"Skipping {symbol}: {requests_needed} requests needed, "
                            f"{max(request_budget - requests_used, 0)} remaining"
                        ),
                    )
                )
                checkpoint()
                continue
            if cik is None:
                requests_used += 1
                try:
                    mappings = sec_client.resolve_mapping(resolve_by="ticker", value=symbol)
                    cik = select_best_company_mapping(mappings).cik
                    cached_ciks[symbol] = cik
                except Exception as exc:
                    skipped.append(
                        WorkflowSkippedRecord(
                            company_symbol=symbol,
                            company_name=plan_row.company_name,
                            reason="sec_request_failed",
                            detail=f"mapping request failed: {exc}",
                        )
                    )
                    checkpoint()
                    continue
                checkpoint()

            requests_used += 1
            try:
                filings = sec_client.query_ex99_1_filings(
                    query=build_sec_ex99_query(symbol=symbol, cik=cik)
                )
            except Exception as exc:
                skipped.append(
                    WorkflowSkippedRecord(
                        company_symbol=symbol,
                        company_name=plan_row.company_name,
                        reason="sec_request_failed",
                        detail=f"filing query failed: {exc}",
                    )
                )
                checkpoint()
                continue
            checkpoint()

            pairs = _pair_records_to_filings(rows_by_company[symbol], filings)
            if max_document_workers > 1 and len(pairs) > 1:
                with ThreadPoolExecutor(max_workers=max_document_workers) as executor:
                    future_to_context = {
                        executor.submit(
                            _collect_manifest_for_pair,
                            record=record,
                            filing=filing,
                            raw_dir=raw_dir,
                            http_client=client,
                            document_collector=document_collector,
                        ): record
                        for record, filing in pairs
                    }
                    for future in as_completed(future_to_context):
                        record = future_to_context[future]
                        try:
                            manifests.append(future.result())
                        except Exception as exc:
                            skipped.append(_document_fetch_skip(record, exc))
                        checkpoint()
            else:
                for record, filing in pairs:
                    try:
                        manifests.append(
                            _collect_manifest_for_pair(
                                record=record,
                                filing=filing,
                                raw_dir=raw_dir,
                                http_client=client,
                                document_collector=document_collector,
                            )
                        )
                    except Exception as exc:
                        skipped.append(_document_fetch_skip(record, exc))
                    checkpoint()
    finally:
        if http_client is None:
            client.close()

    return result_snapshot()


def collect_historical_news_manifests(
    examples: list[HistoricalTrainingExample],
    *,
    defeatbeta_client: Any,
    raw_dir: Path,
    company_names: dict[str, str] | None = None,
    yfinance_collector: YFinanceNewsCollector | None = None,
    days_before_cutoff: int = 60,
    max_articles_per_window: int = 80,
    max_total_articles: int = 5000,
) -> EvidenceCollectionResult:
    if days_before_cutoff < 1:
        raise ValueError("days_before_cutoff must be at least 1")
    if max_articles_per_window < 1:
        raise ValueError("max_articles_per_window must be at least 1")
    if max_total_articles < 1:
        raise ValueError("max_total_articles must be at least 1")

    windows = _news_windows_from_examples(examples, days_before_cutoff=days_before_cutoff)
    company_names = {key.upper(): value for key, value in (company_names or {}).items()}
    manifests: list[PublicDocumentManifest] = []
    skipped: list[WorkflowSkippedRecord] = []
    fetched_at = datetime.now(tz=UTC)
    seen: set[tuple[str, str, int, int, str]] = set()

    for symbol, symbol_windows in sorted(_group_windows_by_company(windows).items()):
        if len(manifests) >= max_total_articles:
            break
        try:
            defeatbeta_articles = defeatbeta_client.search_stock_news(
                symbol=symbol,
                start_date=min(window["start"].date().isoformat() for window in symbol_windows),
                end_date=max(window["cutoff"].date().isoformat() for window in symbol_windows),
                max_rows=max(max_articles_per_window * len(symbol_windows) * 2, 1),
            )
        except Exception as exc:
            defeatbeta_articles = []
            for window in symbol_windows:
                skipped.append(
                    WorkflowSkippedRecord(
                        company_symbol=symbol,
                        fiscal_year=window["fiscal_year"],
                        fiscal_quarter=window["fiscal_quarter"],
                        reason="news_collection_failed",
                        detail=f"DefeatBeta failed: {exc}",
                    )
                )

        for window in symbol_windows:
            if len(manifests) >= max_total_articles:
                break
            selected = _articles_for_window(defeatbeta_articles, window)[
                :max_articles_per_window
            ]
            provider = "defeatbeta"
            if not selected and yfinance_collector is not None:
                selected = _articles_for_window(
                    yfinance_collector(
                        company_symbol=symbol,
                        company_name=company_names.get(symbol, symbol),
                        from_date=window["start"].date().isoformat(),
                        to_date=window["cutoff"].date().isoformat(),
                        max_articles=max_articles_per_window,
                    ),
                    window,
                )[:max_articles_per_window]
                provider = "yfinance"
            if not selected:
                skipped.append(
                    WorkflowSkippedRecord(
                        company_symbol=symbol,
                        fiscal_year=window["fiscal_year"],
                        fiscal_quarter=window["fiscal_quarter"],
                        reason="news_missing",
                        detail="No pre-cutoff DefeatBeta/yfinance articles found",
                    )
                )
                continue

            for article in selected:
                if len(manifests) >= max_total_articles:
                    break
                manifest = _news_manifest_for_article(
                    article=article,
                    provider=provider,
                    company_symbol=symbol,
                    fiscal_year=window["fiscal_year"],
                    fiscal_quarter=window["fiscal_quarter"],
                    fetched_at=fetched_at,
                    raw_dir=raw_dir,
                )
                key = (
                    manifest.source_url,
                    manifest.company_symbol,
                    manifest.fiscal_year,
                    manifest.fiscal_quarter,
                    manifest.content_hash,
                )
                if key in seen:
                    continue
                seen.add(key)
                manifests.append(manifest)

    merged = merge_document_manifests(manifests)
    return EvidenceCollectionResult(
        manifests=merged,
        summary=EvidenceCollectionSummary(
            sec_requests_used=0,
            sec_budget_remaining=0,
            manifest_count=len(merged),
            skipped_records=skipped,
        ),
    )


def merge_document_manifests(
    manifests: list[PublicDocumentManifest],
) -> list[PublicDocumentManifest]:
    by_key: dict[tuple[str, str, int, int, str], PublicDocumentManifest] = {}
    for manifest in manifests:
        key = (
            manifest.source_url,
            manifest.company_symbol,
            manifest.fiscal_year,
            manifest.fiscal_quarter,
            manifest.source_type,
        )
        by_key.setdefault(key, manifest)
    return list(by_key.values())


def _news_windows_from_examples(
    examples: list[HistoricalTrainingExample],
    *,
    days_before_cutoff: int,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, int, datetime], dict[str, Any]] = {}
    for example in examples:
        cutoff = _aware_datetime(example.evidence_cutoff)
        key = (
            example.company_symbol.upper(),
            example.fiscal_year,
            example.fiscal_quarter,
            cutoff,
        )
        by_key.setdefault(
            key,
            {
                "company_symbol": example.company_symbol.upper(),
                "fiscal_year": example.fiscal_year,
                "fiscal_quarter": example.fiscal_quarter,
                "cutoff": cutoff,
                "start": cutoff - timedelta(days=days_before_cutoff),
            },
        )
    return sorted(
        by_key.values(),
        key=lambda row: (
            row["company_symbol"],
            row["fiscal_year"],
            row["fiscal_quarter"],
            row["cutoff"],
        ),
    )


def _group_windows_by_company(windows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        grouped.setdefault(str(window["company_symbol"]), []).append(window)
    return grouped


def _articles_for_window(articles: list[Any], window: dict[str, Any]) -> list[Any]:
    start = window["start"]
    cutoff = window["cutoff"]
    selected = [
        article
        for article in articles
        if start <= _aware_datetime(article.published_at) <= cutoff
    ]
    return sorted(selected, key=lambda article: _aware_datetime(article.published_at), reverse=True)


def _news_manifest_for_article(
    *,
    article: Any,
    provider: str,
    company_symbol: str,
    fiscal_year: int,
    fiscal_quarter: int,
    fetched_at: datetime,
    raw_dir: Path,
) -> PublicDocumentManifest:
    text = _news_article_text(article=article, company_symbol=company_symbol)
    digest = content_hash(text)
    article_raw_dir = raw_dir / company_symbol.upper()
    article_raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = (
        article_raw_dir
        / f"{company_symbol.upper()}-{fiscal_year}-Q{fiscal_quarter}-"
        f"{digest[:12]}-{provider}.txt"
    )
    raw_path.write_text(text, encoding="utf-8")
    reliability_score = _news_source_reliability(getattr(article, "source_name", None))
    flavor = "opinion" if _is_opinion_article(article) else "relevant"
    return PublicDocumentManifest(
        source_url=getattr(article, "link", None) or f"{provider}://article/{article.article_id}",
        company_symbol=company_symbol,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        source_type=(
            f"news_article_{flavor}_reliability_{int(round(reliability_score * 100)):03d}"
        ),
        published_at=_aware_datetime(article.published_at),
        fetched_at=fetched_at,
        raw_path=str(raw_path),
        extracted_text_path=str(raw_path),
        content_hash=digest,
        extraction_method=f"{provider}_api",
    )


def _news_article_text(*, article: Any, company_symbol: str) -> str:
    parts = [
        "# News article",
        f"Company symbol: {company_symbol.upper()}",
        f"Article ID: {getattr(article, 'article_id', '')}",
        f"Published at: {getattr(article, 'published_at', '')}",
        f"Source: {getattr(article, 'source_name', '') or 'unknown'}",
        f"Datatype: {getattr(article, 'datatype', '')}",
        f"Title: {getattr(article, 'title', '')}",
        f"URL: {getattr(article, 'link', '')}",
        "",
        "Description:",
        str(getattr(article, "description", "") or ""),
        "",
        "Content:",
        str(getattr(article, "content", "") or ""),
        "",
    ]
    return "\n".join(parts).strip() + "\n"


def _news_source_reliability(source_name: str | None) -> float:
    if not source_name:
        return 0.72
    normalized = source_name.lower()
    if any(token in normalized for token in ["reuters", "ap news", "associated press"]):
        return 0.95
    if any(token in normalized for token in ["bloomberg", "dow jones", "wall street journal"]):
        return 0.92
    if any(token in normalized for token in ["cnbc", "marketwatch", "yahoo finance"]):
        return 0.82
    return 0.72


def _is_opinion_article(article: Any) -> bool:
    text = " ".join(
        [
            str(getattr(article, "title", "") or ""),
            str(getattr(article, "description", "") or ""),
            str(getattr(article, "datatype", "") or ""),
        ]
    ).lower()
    return any(token in text for token in ["opinion", "analysis", "editorial", "why "])


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=UTC)
    return value


def _group_inventory_by_company(
    rows: list[TranscriptInventoryRow],
) -> dict[str, list[TranscriptInventoryRow]]:
    grouped: dict[str, list[TranscriptInventoryRow]] = {}
    for row in rows:
        grouped.setdefault(row.company_symbol, []).append(row)
    return grouped


def _collect_manifest_for_pair(
    *,
    record: TranscriptInventoryRow,
    filing: SecApiFiling,
    raw_dir: Path,
    http_client: httpx.Client,
    document_collector: DocumentCollector,
) -> PublicDocumentManifest:
    exhibit_url = _preferred_exhibit_url(filing)
    return document_collector(
        url=exhibit_url,
        company_symbol=record.company_symbol,
        fiscal_year=record.fiscal_year,
        fiscal_quarter=record.fiscal_quarter,
        source_type=_sec_exhibit_source_type(filing, exhibit_url),
        published_at=filing.filed_at,
        fetched_at=datetime.now(tz=UTC),
        raw_dir=raw_dir / record.company_symbol,
        http_client=http_client,
    )


def _document_fetch_skip(record: TranscriptInventoryRow, exc: Exception) -> WorkflowSkippedRecord:
    return WorkflowSkippedRecord(
        company_symbol=record.company_symbol,
        company_name=record.company_name,
        fiscal_year=record.fiscal_year,
        fiscal_quarter=record.fiscal_quarter,
        path=record.transcript_path,
        reason="sec_document_fetch_failed",
        detail=str(exc),
    )


def build_sec_ex99_query(*, symbol: str, cik: str) -> str:
    return (
        f'formType:"8-K" AND cik:{cik} AND '
        '(documentFormatFiles.type:"EX-99*" OR documentFormatFiles.type:"EX-99.1")'
    )


def _pair_records_to_filings(
    records: list[TranscriptInventoryRow],
    filings: list[SecApiFiling],
) -> list[tuple[TranscriptInventoryRow, SecApiFiling]]:
    used_filing_indexes: set[int] = set()
    pairs = []
    for record in records:
        candidates = []
        for index, filing in enumerate(filings):
            if index in used_filing_indexes:
                continue
            year_distance = abs(filing.filed_at.year - record.fiscal_year)
            quarter_distance = abs(_calendar_quarter(filing.filed_at.month) - record.fiscal_quarter)
            candidates.append((year_distance, quarter_distance, index, filing))
        if not candidates:
            continue
        _, _, index, filing = min(candidates, key=lambda candidate: candidate[:3])
        used_filing_indexes.add(index)
        pairs.append((record, filing))
    return pairs


def _calendar_quarter(month: int) -> int:
    return ((month - 1) // 3) + 1


def _preferred_exhibit_url(filing: SecApiFiling) -> str:
    for exhibit in filing.exhibits:
        if Path(exhibit.document_url.lower()).suffix in {".htm", ".html"}:
            return exhibit.document_url
    return filing.exhibit_url


def _sec_exhibit_source_type(filing: SecApiFiling, exhibit_url: str) -> str:
    for exhibit in filing.exhibits:
        if exhibit.document_url == exhibit_url:
            normalized = exhibit.document_type.lower().replace("-", "_").replace(".", "_")
            return f"sec_{normalized}_supplemental"
    return "sec_ex_99_supplemental"
