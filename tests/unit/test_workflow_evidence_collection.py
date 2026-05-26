from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from kalorie.clients.defeatbeta import DefeatBetaArticle
from kalorie.clients.sec_api import SecApiExhibit, SecApiFiling
from kalorie.clients.tiingo import TiingoArticle
from kalorie.io.public_documents import PublicDocumentManifest
from kalorie.ml.datasets import HistoricalTrainingExample
from kalorie.workflows.evidence_collection import (
    build_sec_ex99_query,
    collect_historical_news_manifests,
    collect_sec_ex99_manifests,
    merge_document_manifests,
    plan_sec_requests,
)
from kalorie.workflows.models import TranscriptInventory, TranscriptInventoryRow


def _inventory(tmp_path: Path) -> TranscriptInventory:
    path = tmp_path / "2025_Q2_wmt_processed.txt"
    path.write_text("Traffic improved.", encoding="utf-8")
    return TranscriptInventory(
        rows=[
            TranscriptInventoryRow(
                company_symbol="WMT",
                company_name="Walmart",
                fiscal_year=2025,
                fiscal_quarter=2,
                transcript_path=path,
            ),
            TranscriptInventoryRow(
                company_symbol="wmt",
                company_name="Walmart",
                fiscal_year=2025,
                fiscal_quarter=3,
                transcript_path=path,
            ),
        ]
    )


def test_plan_sec_requests_groups_by_company_and_uses_cached_ciks(tmp_path: Path):
    plan = plan_sec_requests(
        _inventory(tmp_path),
        cached_ciks={"WMT": "0104169"},
        request_budget=80,
    )

    assert plan.projected_requests == 1
    assert plan.within_budget is True
    assert len(plan.companies) == 1
    assert plan.companies[0].cached_cik == "0104169"
    assert plan.companies[0].transcript_count == 2


def test_build_sec_ex99_query_uses_document_format_files_pattern():
    query = build_sec_ex99_query(symbol="WMT", cik="104169")

    assert 'formType:"8-K"' in query
    assert "cik:104169" in query
    assert 'documentFormatFiles.type:"EX-99*"' in query


def test_collect_sec_ex99_manifests_honors_budget_before_spending(tmp_path: Path):
    class FakeSecClient:
        def resolve_mapping(self, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("budget exhausted before mapping")

        def query_ex99_1_filings(self, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("budget exhausted before filing query")

    result = collect_sec_ex99_manifests(
        _inventory(tmp_path),
        sec_client=FakeSecClient(),
        cached_ciks={},
        request_budget=1,
        raw_dir=tmp_path / "raw",
    )

    assert result.manifests == []
    assert result.summary.sec_requests_used == 0
    assert result.summary.skipped_records[0].reason == "sec_budget_exhausted"


def test_collect_sec_ex99_manifests_saves_partial_progress_when_budget_runs_out(
    tmp_path: Path,
):
    transcript = tmp_path / "2025_Q2_wmt_processed.txt"
    transcript.write_text("Traffic improved.", encoding="utf-8")
    inventory = TranscriptInventory(
        rows=[
            TranscriptInventoryRow(
                company_symbol="WMT",
                company_name="Walmart",
                fiscal_year=2025,
                fiscal_quarter=2,
                transcript_path=transcript,
            ),
            TranscriptInventoryRow(
                company_symbol="TGT",
                company_name="Target",
                fiscal_year=2025,
                fiscal_quarter=2,
                transcript_path=transcript,
            ),
        ]
    )

    class FakeSecClient:
        def query_ex99_1_filings(self, **kwargs):
            return []

    checkpoints = []

    result = collect_sec_ex99_manifests(
        inventory,
        sec_client=FakeSecClient(),
        cached_ciks={"WMT": "0104169", "TGT": "0027419"},
        request_budget=1,
        raw_dir=tmp_path / "raw",
        checkpoint_writer=lambda result, cache: checkpoints.append(
            (result.summary.sec_requests_used, len(result.manifests), dict(cache))
        ),
    )

    assert result.summary.sec_requests_used == 1
    assert result.summary.skipped_records[-1].reason == "sec_budget_exhausted"
    assert checkpoints


def test_collect_sec_ex99_manifests_checkpoints_request_failures(tmp_path: Path):
    class FakeSecClient:
        def query_ex99_1_filings(self, **kwargs):
            raise RuntimeError("temporary SEC failure")

    checkpoints = []

    result = collect_sec_ex99_manifests(
        _inventory(tmp_path),
        sec_client=FakeSecClient(),
        cached_ciks={"WMT": "0104169"},
        request_budget=80,
        raw_dir=tmp_path / "raw",
        checkpoint_writer=lambda result, cache: checkpoints.append(
            (result.summary.sec_requests_used, result.summary.skipped_records[-1].reason)
        ),
    )

    assert result.manifests == []
    assert result.summary.skipped_records[-1].reason == "sec_request_failed"
    assert checkpoints[-1] == (1, "sec_request_failed")


def test_collect_sec_ex99_manifests_prefers_htm_exhibit_and_dedupes(tmp_path: Path):
    class FakeSecClient:
        def query_ex99_1_filings(self, **kwargs):
            return [
                SecApiFiling(
                    ticker="WMT",
                    cik="0104169",
                    filed_at=datetime(2025, 8, 21, tzinfo=UTC),
                    exhibit_url="https://www.sec.gov/Archives/wmt/ex991.txt",
                    exhibits=[
                        SecApiExhibit(
                            document_type="EX-99.1",
                            description="txt mirror",
                            document_url="https://www.sec.gov/Archives/wmt/ex991.txt",
                        ),
                        SecApiExhibit(
                            document_type="EX-99.1",
                            description="earnings release",
                            document_url="https://www.sec.gov/Archives/wmt/ex991.htm",
                        ),
                    ],
                )
            ]

    collected_urls: list[str] = []

    def fake_collector(**kwargs):
        collected_urls.append(kwargs["url"])
        return PublicDocumentManifest(
            source_url=kwargs["url"],
            company_symbol=kwargs["company_symbol"],
            fiscal_year=kwargs["fiscal_year"],
            fiscal_quarter=kwargs["fiscal_quarter"],
            source_type=kwargs["source_type"],
            published_at=kwargs["published_at"],
            fetched_at=datetime(2025, 8, 21, 13, 0, tzinfo=UTC),
            raw_path=str(tmp_path / "ex991.txt"),
            raw_original_path=str(tmp_path / "ex991.htm"),
            raw_original_content_hash="raw",
            extracted_text_path=str(tmp_path / "ex991.txt"),
            content_hash="hash",
            extraction_method="html_text",
        )

    result = collect_sec_ex99_manifests(
        _inventory(tmp_path),
        sec_client=FakeSecClient(),
        cached_ciks={"WMT": "0104169"},
        request_budget=80,
        raw_dir=tmp_path / "raw",
        document_collector=fake_collector,
    )
    merged = merge_document_manifests([*result.manifests, *result.manifests])

    assert collected_urls == ["https://www.sec.gov/Archives/wmt/ex991.htm"]
    assert len(result.manifests) == 1
    assert len(merged) == 1


def test_collect_historical_news_manifests_filters_after_evidence_cutoff(
    tmp_path: Path,
):
    cutoff = datetime(2025, 5, 10, 12, 0, tzinfo=UTC)
    examples = [_example(cutoff=cutoff)]

    class FakeDefeatBetaClient:
        def search_stock_news(self, **kwargs):
            return [
                DefeatBetaArticle(
                    article_id="before",
                    title="Walmart expands advertising push",
                    link="https://example.com/before",
                    source_name="Reuters",
                    description="Advertising update",
                    content="Walmart discussed retail media before the call.",
                    datatype="story",
                    published_at=datetime(2025, 5, 10, 11, 30, tzinfo=UTC),
                    tickers=["WMT"],
                ),
                DefeatBetaArticle(
                    article_id="after",
                    title="Walmart post-call recap",
                    link="https://example.com/after",
                    source_name="Reuters",
                    description="Post-call recap",
                    content="This article was published after the cutoff.",
                    datatype="story",
                    published_at=datetime(2025, 5, 10, 12, 30, tzinfo=UTC),
                    tickers=["WMT"],
                ),
            ]

    result = collect_historical_news_manifests(
        examples,
        defeatbeta_client=FakeDefeatBetaClient(),
        raw_dir=tmp_path / "raw",
        days_before_cutoff=30,
        max_articles_per_window=10,
    )

    assert len(result.manifests) == 1
    manifest = result.manifests[0]
    assert manifest.source_url == "https://example.com/before"
    assert manifest.published_at <= cutoff
    assert manifest.extraction_method == "defeatbeta_api"
    assert Path(manifest.raw_path).read_text(encoding="utf-8").count("post-call") == 0


def test_collect_historical_news_manifests_uses_yfinance_fallback_for_empty_windows(
    tmp_path: Path,
):
    cutoff = datetime(2025, 5, 10, 12, 0, tzinfo=UTC)
    examples = [_example(cutoff=cutoff)]

    class EmptyDefeatBetaClient:
        def search_stock_news(self, **kwargs):
            return []

    fallback_calls = []

    def fake_yfinance_collector(**kwargs):
        fallback_calls.append(kwargs)
        return [
            TiingoArticle(
                article_id="yf-1",
                title="Walmart earnings preview",
                link="https://finance.yahoo.com/news/wmt-preview",
                source_name="Yahoo Finance",
                description="Preview",
                content=None,
                datatype="story",
                published_at=datetime(2025, 5, 9, 15, 0, tzinfo=UTC),
                tickers=["WMT"],
            )
        ]

    result = collect_historical_news_manifests(
        examples,
        defeatbeta_client=EmptyDefeatBetaClient(),
        raw_dir=tmp_path / "raw",
        yfinance_collector=fake_yfinance_collector,
        days_before_cutoff=30,
        max_articles_per_window=10,
    )

    assert len(result.manifests) == 1
    assert fallback_calls[0]["company_symbol"] == "WMT"
    assert fallback_calls[0]["to_date"] == "2025-05-10"
    assert result.manifests[0].extraction_method == "yfinance_api"


def _example(*, cutoff: datetime) -> HistoricalTrainingExample:
    return HistoricalTrainingExample(
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        evidence_cutoff=cutoff,
        market_id="SYNTH-WMT-2025Q2-ads",
        target_phrase="advertising",
        label=1,
        features={},
        document_ids=[],
        market_probability=Decimal("0.50"),
        market_venue="synthetic",
    )
