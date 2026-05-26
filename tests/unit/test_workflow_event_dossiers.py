from datetime import UTC, datetime
from pathlib import Path

from kalorie.data_grepping.event_scenarios import EventScenarioCatalog
from kalorie.io.public_documents import PublicDocumentManifest
from kalorie.workflows.event_dossiers import (
    event_dossier_id,
    generate_event_dossiers,
    phrase_variants_by_event,
    source_digest_for_event,
)
from kalorie.workflows.models import PhraseCatalog, PhraseCatalogEntry, TranscriptInventoryRow


def test_source_digest_for_event_is_stable_across_manifest_order(tmp_path: Path):
    first = _manifest(tmp_path, url="https://example.com/a", digest="aaa")
    second = _manifest(tmp_path, url="https://example.com/b", digest="bbb")

    left = source_digest_for_event(
        manifests=[first, second],
        target_phrases=["traffic", "advertising"],
        prompt_version="v1",
    )
    right = source_digest_for_event(
        manifests=[second, first],
        target_phrases=["advertising", "traffic"],
        prompt_version="v1",
    )

    assert left == right


def test_source_digest_for_event_changes_when_generation_config_changes(tmp_path: Path):
    manifest = _manifest(tmp_path, url="https://example.com/a", digest="aaa")

    left = source_digest_for_event(
        manifests=[manifest],
        target_phrases=["traffic"],
        prompt_version="v1",
        max_items=10,
    )
    right = source_digest_for_event(
        manifests=[manifest],
        target_phrases=["traffic"],
        prompt_version="v1",
        max_items=20,
    )

    assert left != right


def test_generate_event_dossiers_reuses_cached_catalog_when_source_digest_matches(
    tmp_path: Path,
):
    row = TranscriptInventoryRow(
        company_symbol="WMT",
        company_name="Walmart",
        fiscal_year=2025,
        fiscal_quarter=2,
        transcript_path=tmp_path / "transcript.txt",
    )
    manifest = _manifest(tmp_path, url="https://example.com/a", digest="aaa")
    phrase_catalog = PhraseCatalog(
        entries=[
            PhraseCatalogEntry(
                company_symbol="WMT",
                fiscal_year=2025,
                fiscal_quarter=2,
                phrase="advertising",
                label="present",
            )
        ]
    )
    cached = EventScenarioCatalog(
        event_id="WMT-2025-Q2",
        company_symbol="WMT",
        company_name="Walmart",
        llm_model="fake-model",
        source_digest=source_digest_for_event(
            manifests=[manifest],
            target_phrases=["advertising"],
            prompt_version="event-dossier-v1",
            llm_model="fake-model",
        ),
        topics=["retail media"],
        analyst_questions=["How large is the advertising business?"],
        management_answers=["Management may discuss retail media momentum."],
        synthetic_call_snippets=[],
        target_phrase_variants={"advertising": ["retail media"]},
        source_rationales=[],
    )
    cache_path = tmp_path / "WMT-2025-Q2.json"
    cache_path.write_text(cached.model_dump_json(), encoding="utf-8")

    class FailingGenerator:
        model = "fake-model"

        def generate(self, **_kwargs):  # pragma: no cover - cache should bypass this
            raise AssertionError("generator should not be called")

    catalogs = generate_event_dossiers(
        inventory_rows=[row],
        manifests=[manifest],
        phrase_catalog=phrase_catalog,
        generator=FailingGenerator(),
        cache_dir=tmp_path,
    )

    assert catalogs == [cached]


def test_generate_event_dossiers_filters_manifests_after_evidence_cutoff(
    tmp_path: Path,
):
    row = TranscriptInventoryRow(
        company_symbol="WMT",
        company_name="Walmart",
        fiscal_year=2025,
        fiscal_quarter=2,
        transcript_path=tmp_path / "transcript.txt",
        estimated_call_time=datetime(2025, 5, 10, 12, 0, tzinfo=UTC),
    )
    before = _manifest_with_text(
        tmp_path,
        url="https://example.com/before",
        digest="before",
        published_at=datetime(2025, 5, 10, 11, 0, tzinfo=UTC),
        text="Pre-call traffic evidence.",
    )
    after = _manifest_with_text(
        tmp_path,
        url="https://example.com/after",
        digest="after",
        published_at=datetime(2025, 5, 10, 11, 59, tzinfo=UTC),
        text="Post-cutoff leaked evidence.",
    )
    phrase_catalog = PhraseCatalog(
        entries=[
            PhraseCatalogEntry(
                company_symbol="WMT",
                fiscal_year=2025,
                fiscal_quarter=2,
                phrase="traffic",
                label="present",
            )
        ]
    )

    class FakeGenerator:
        model = "fake-model"

        def generate(self, **kwargs):
            assert kwargs["material_snippets"] == ["Pre-call traffic evidence."]
            return EventScenarioCatalog(
                event_id=kwargs["event_id"],
                company_symbol=kwargs["company_symbol"],
                company_name=kwargs["company_name"],
                llm_model=self.model,
                topics=["traffic"],
                analyst_questions=[],
                management_answers=[],
                synthetic_call_snippets=[],
                target_phrase_variants={"traffic": ["guest traffic"]},
                source_rationales=[],
            )

    catalogs = generate_event_dossiers(
        inventory_rows=[row],
        manifests=[after, before],
        phrase_catalog=phrase_catalog,
        generator=FakeGenerator(),
        cache_dir=tmp_path,
    )

    assert len(catalogs) == 1


def test_generate_event_dossiers_calls_generator_and_writes_cache(tmp_path: Path):
    row = TranscriptInventoryRow(
        company_symbol="WMT",
        company_name="Walmart",
        fiscal_year=2025,
        fiscal_quarter=2,
        transcript_path=tmp_path / "transcript.txt",
    )
    manifest = _manifest(tmp_path, url="https://example.com/a", digest="aaa")
    phrase_catalog = PhraseCatalog(
        entries=[
            PhraseCatalogEntry(
                company_symbol="WMT",
                fiscal_year=2025,
                fiscal_quarter=2,
                phrase="traffic",
                label="present",
            )
        ]
    )

    class FakeGenerator:
        model = "fake-model"

        def generate(self, **kwargs):
            assert kwargs["event_id"] == "WMT-2025-Q2"
            assert kwargs["target_phrases"] == ["traffic"]
            assert kwargs["material_snippets"]
            return EventScenarioCatalog(
                event_id=kwargs["event_id"],
                company_symbol=kwargs["company_symbol"],
                company_name=kwargs["company_name"],
                llm_model=self.model,
                topics=["store traffic"],
                analyst_questions=[],
                management_answers=[],
                synthetic_call_snippets=[],
                target_phrase_variants={"traffic": ["guest traffic"]},
                source_rationales=[],
            )

    catalogs = generate_event_dossiers(
        inventory_rows=[row],
        manifests=[manifest],
        phrase_catalog=phrase_catalog,
        generator=FakeGenerator(),
        cache_dir=tmp_path,
    )

    assert catalogs[0].source_digest
    assert catalogs[0].target_phrase_variants == {"traffic": ["guest traffic"]}
    assert (tmp_path / "WMT-2025-Q2.json").exists()


def test_generate_event_dossiers_reports_progress_with_bounded_workers(tmp_path: Path):
    rows = [
        TranscriptInventoryRow(
            company_symbol="WMT",
            company_name="Walmart",
            fiscal_year=2025,
            fiscal_quarter=quarter,
            transcript_path=tmp_path / f"transcript-{quarter}.txt",
        )
        for quarter in [1, 2]
    ]
    manifests = [
        _manifest_for_period(tmp_path, quarter=1, digest="aaa"),
        _manifest_for_period(tmp_path, quarter=2, digest="bbb"),
    ]
    phrase_catalog = PhraseCatalog(
        entries=[
            PhraseCatalogEntry(
                company_symbol="WMT",
                fiscal_year=2025,
                fiscal_quarter=quarter,
                phrase="traffic",
                label="present",
            )
            for quarter in [1, 2]
        ]
    )

    class FakeGenerator:
        model = "fake-model"

        def generate(self, **kwargs):
            return EventScenarioCatalog(
                event_id=kwargs["event_id"],
                company_symbol=kwargs["company_symbol"],
                company_name=kwargs["company_name"],
                llm_model=self.model,
                topics=[kwargs["event_id"]],
                analyst_questions=[],
                management_answers=[],
                synthetic_call_snippets=[],
                target_phrase_variants={"traffic": ["guest traffic"]},
                source_rationales=[],
            )

    progress = []
    catalogs = generate_event_dossiers(
        inventory_rows=rows,
        manifests=manifests,
        phrase_catalog=phrase_catalog,
        generator=FakeGenerator(),
        cache_dir=tmp_path,
        max_workers=2,
        progress_callback=lambda done, total, event_id, reused: progress.append(
            (done, total, event_id, reused)
        ),
    )

    assert len(catalogs) == 2
    assert progress[-1][0:2] == (2, 2)
    assert {row[2] for row in progress} == {"WMT-2025-Q1", "WMT-2025-Q2"}


def test_phrase_variants_by_event_filters_noisy_generated_variants():
    catalog = EventScenarioCatalog(
        event_id="WMT-2025-Q2",
        company_symbol="WMT",
        company_name="Walmart",
        llm_model="fake-model",
        topics=[],
        analyst_questions=[],
        management_answers=[],
        synthetic_call_snippets=[],
        target_phrase_variants={
            "you": ["your company", "your organization"],
            "s": ["is", "has been"],
            "cloud": ["cloud services", "cloud", " "],
            "advertising": ["retail media", "retail media"],
        },
        source_rationales=[],
    )

    variants = phrase_variants_by_event([catalog])

    assert variants == {
        "WMT-2025-Q2": {
            "advertising": ["retail media"],
            "cloud": ["cloud services", "cloud"],
        }
    }


def _manifest(tmp_path: Path, *, url: str, digest: str) -> PublicDocumentManifest:
    text_path = tmp_path / f"{digest}.txt"
    text_path.write_text("Retail media and store traffic were discussed.", encoding="utf-8")
    return PublicDocumentManifest(
        source_url=url,
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        source_type="news_article_relevant_reliability_095",
        published_at=datetime(2025, 5, 9, tzinfo=UTC),
        fetched_at=datetime(2025, 5, 9, 1, tzinfo=UTC),
        raw_path=str(text_path),
        extracted_text_path=str(text_path),
        content_hash=digest,
        extraction_method="defeatbeta_api",
    )


def _manifest_with_text(
    tmp_path: Path,
    *,
    url: str,
    digest: str,
    published_at: datetime,
    text: str,
) -> PublicDocumentManifest:
    text_path = tmp_path / f"{digest}.txt"
    text_path.write_text(text, encoding="utf-8")
    return PublicDocumentManifest(
        source_url=url,
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        source_type="news_article_relevant_reliability_095",
        published_at=published_at,
        fetched_at=published_at,
        raw_path=str(text_path),
        extracted_text_path=str(text_path),
        content_hash=digest,
        extraction_method="defeatbeta_api",
    )


def _manifest_for_period(
    tmp_path: Path,
    *,
    quarter: int,
    digest: str,
) -> PublicDocumentManifest:
    text_path = tmp_path / f"{digest}.txt"
    text_path.write_text("Store traffic was discussed.", encoding="utf-8")
    return PublicDocumentManifest(
        source_url=f"https://example.com/{digest}",
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=quarter,
        source_type="sec_ex_99_1_supplemental",
        published_at=datetime(2025, quarter, 1, tzinfo=UTC),
        fetched_at=datetime(2025, quarter, 1, 1, tzinfo=UTC),
        raw_path=str(text_path),
        extracted_text_path=str(text_path),
        content_hash=digest,
        extraction_method="html_text",
    )


def test_event_dossier_id_uses_company_and_period():
    assert event_dossier_id("wmt", 2025, 2) == "WMT-2025-Q2"
