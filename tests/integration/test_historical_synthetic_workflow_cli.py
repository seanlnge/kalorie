import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from typer.testing import CliRunner

from kalorie.app.cli import app
from kalorie.clients.defeatbeta import DefeatBetaArticle
from kalorie.data_grepping.event_scenarios import EventScenarioCatalog
from kalorie.io.public_documents import PublicDocumentManifest
from kalorie.ml.datasets import HistoricalTrainingExample
from kalorie.workflows.historical_synthetic import (
    build_historical_synthetic_rows,
    build_transcript_inventory,
)
from kalorie.workflows.models import PhraseCatalog, PhraseCatalogEntry


def test_build_historical_synthetic_rows_uses_phrase_catalog_and_synthetic_markets(
    tmp_path: Path,
):
    company = tmp_path / "Walmart"
    company.mkdir()
    transcript = company / "2025_Q2_wmt_processed.txt"
    transcript.write_text("Traffic improved and OpenAI helped.", encoding="utf-8")
    release = tmp_path / "release.txt"
    release.write_text("Traffic was strong before the call.", encoding="utf-8")
    inventory = build_transcript_inventory(tmp_path)
    manifests = [
        PublicDocumentManifest(
            source_url="https://www.sec.gov/Archives/wmt/ex991.htm",
            company_symbol="WMT",
            fiscal_year=2025,
            fiscal_quarter=2,
            source_type="sec_ex_99_1_supplemental",
            published_at=datetime(2025, 8, 1, tzinfo=UTC),
            fetched_at=datetime(2025, 8, 1, 1, tzinfo=UTC),
            raw_path=str(release),
            raw_original_path=str(tmp_path / "release.htm"),
            raw_original_content_hash="raw",
            extracted_text_path=str(release),
            content_hash="hash",
            extraction_method="html_text",
        )
    ]
    phrase_catalog = PhraseCatalog(
        entries=[
            PhraseCatalogEntry(
                company_symbol="WMT",
                fiscal_year=2025,
                fiscal_quarter=2,
                phrase="traffic",
                label="present",
                match_count=1,
            ),
            PhraseCatalogEntry(
                company_symbol="WMT",
                fiscal_year=2025,
                fiscal_quarter=2,
                phrase="robotaxi",
                label="absent",
                match_count=0,
            ),
        ]
    )

    result = build_historical_synthetic_rows(
        inventory=inventory,
        manifests=manifests,
        phrase_catalog=phrase_catalog,
    )

    by_phrase = {example.target_phrase: example for example in result.examples}
    assert sorted(by_phrase) == ["robotaxi", "traffic"]
    assert by_phrase["traffic"].label == 1
    assert by_phrase["robotaxi"].label == 0
    assert by_phrase["traffic"].market_venue == "synthetic"
    assert by_phrase["traffic"].market_id == "WMT-2025-Q2-traffic"
    assert not result.skipped_records


def test_build_historical_synthetic_rows_uses_event_dossiers_as_derived_features(
    tmp_path: Path,
):
    company = tmp_path / "Walmart"
    company.mkdir()
    transcript = company / "2025_Q2_wmt_processed.txt"
    transcript.write_text("Guest traffic improved.", encoding="utf-8")
    release = tmp_path / "release.txt"
    release.write_text("Guest traffic improved before the call.", encoding="utf-8")
    inventory = build_transcript_inventory(tmp_path)
    manifests = [
        PublicDocumentManifest(
            source_url="https://www.sec.gov/Archives/wmt/ex991.htm",
            company_symbol="WMT",
            fiscal_year=2025,
            fiscal_quarter=2,
            source_type="sec_ex_99_1_supplemental",
            published_at=datetime(2025, 8, 1, tzinfo=UTC),
            fetched_at=datetime(2025, 8, 1, 1, tzinfo=UTC),
            raw_path=str(release),
            raw_original_path=str(tmp_path / "release.htm"),
            raw_original_content_hash="raw",
            extracted_text_path=str(release),
            content_hash="hash",
            extraction_method="html_text",
        )
    ]
    phrase_catalog = PhraseCatalog(
        entries=[
            PhraseCatalogEntry(
                company_symbol="WMT",
                fiscal_year=2025,
                fiscal_quarter=2,
                phrase="traffic",
                label="present",
                match_count=1,
            )
        ]
    )
    dossiers = [
        EventScenarioCatalog(
            event_id="WMT-2025-Q2",
            company_symbol="WMT",
            company_name="Walmart",
            llm_model="fake-model",
            topics=["store traffic"],
            analyst_questions=["How is traffic trending?"],
            management_answers=[],
            synthetic_call_snippets=[],
            target_phrase_variants={"traffic": ["guest traffic"]},
            source_rationales=[],
        )
    ]

    result = build_historical_synthetic_rows(
        inventory=inventory,
        manifests=manifests,
        phrase_catalog=phrase_catalog,
        event_dossiers=dossiers,
        embedding_provider=_SimpleDossierEmbeddingProvider(),
    )

    features = result.examples[0].features
    assert features["template_phrase_count"] == 1.0
    assert features["max_template_embedding_similarity"] == 1.0
    assert features["scenario_text_count"] == 2.0
    assert features["max_scenario_embedding_similarity"] == 1.0


def test_build_historical_synthetic_rows_cli_accepts_event_dossiers(
    tmp_path: Path,
    monkeypatch,
):
    transcript_root, manifests, phrase_catalog, dossiers = _dossier_fixture(tmp_path)
    manifests_path = tmp_path / "manifests.json"
    phrase_catalog_path = tmp_path / "phrase-catalog.json"
    dossiers_path = tmp_path / "event-dossiers.json"
    out_path = tmp_path / "examples.json"
    manifests_path.write_text(
        json.dumps([manifest.model_dump(mode="json") for manifest in manifests]),
        encoding="utf-8",
    )
    phrase_catalog_path.write_text(
        phrase_catalog.model_dump_json(),
        encoding="utf-8",
    )
    dossiers_path.write_text(
        json.dumps([dossier.model_dump(mode="json") for dossier in dossiers]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kalorie.app.cli._load_embedding_provider",
        lambda _message: _SimpleDossierEmbeddingProvider(),
    )

    result = CliRunner().invoke(
        app,
        [
            "build-historical-synthetic-kalshi-rows",
            "--transcript-root",
            str(transcript_root),
            "--manifests",
            str(manifests_path),
            "--phrase-catalog",
            str(phrase_catalog_path),
            "--event-dossiers",
            str(dossiers_path),
            "--record-concurrency",
            "2",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    examples = json.loads(out_path.read_text(encoding="utf-8"))
    assert examples[0]["features"]["template_phrase_count"] == 1.0
    assert examples[0]["features"]["scenario_text_count"] == 2.0


def test_generate_historical_event_dossiers_cli_writes_cache_and_catalog(
    tmp_path: Path,
    monkeypatch,
):
    transcript_root, manifests, phrase_catalog, _ = _dossier_fixture(tmp_path)
    manifests_path = tmp_path / "manifests.json"
    phrase_catalog_path = tmp_path / "phrase-catalog.json"
    cache_dir = tmp_path / "dossier-cache"
    out_path = tmp_path / "event-dossiers.json"
    manifests_path.write_text(
        json.dumps([manifest.model_dump(mode="json") for manifest in manifests]),
        encoding="utf-8",
    )
    phrase_catalog_path.write_text(phrase_catalog.model_dump_json(), encoding="utf-8")

    class FakeSecret:
        def get_secret_value(self):
            return "test-key"

    class FakeSettings:
        openai_api_key = FakeSecret()

    class FakeGenerator:
        def __init__(self, *, api_key: str, model: str):
            assert api_key == "test-key"
            self.model = model

        def generate(self, **kwargs):
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

    monkeypatch.setattr("kalorie.app.cli._load_settings", lambda: FakeSettings())
    monkeypatch.setattr("kalorie.app.cli.OpenAIEventScenarioGenerator", FakeGenerator)

    result = CliRunner().invoke(
        app,
        [
            "generate-historical-event-dossiers",
            "--transcript-root",
            str(transcript_root),
            "--manifests",
            str(manifests_path),
            "--phrase-catalog",
            str(phrase_catalog_path),
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(out_path),
            "--llm-model",
            "fake-model",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = json.loads(out_path.read_text(encoding="utf-8"))
    assert rows[0]["event_id"] == "WMT-2025-Q2"
    assert rows[0]["source_digest"]
    assert (cache_dir / "WMT-2025-Q2.json").exists()


def test_build_historical_synthetic_rows_reports_missing_evidence_and_phrases(
    tmp_path: Path,
):
    company = tmp_path / "Walmart"
    company.mkdir()
    transcript = company / "2025_Q2_wmt_processed.txt"
    transcript.write_text("Traffic improved.", encoding="utf-8")
    inventory = build_transcript_inventory(tmp_path)

    result = build_historical_synthetic_rows(
        inventory=inventory,
        manifests=[],
        phrase_catalog=PhraseCatalog(entries=[]),
    )

    assert result.examples == []
    reasons = {record.reason for record in result.skipped_records}
    assert reasons == {"missing_evidence", "missing_phrases"}


class _SimpleDossierEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "traffic": [1.0, 0.0],
            "guest traffic": [1.0, 0.0],
            "Guest traffic improved before the call.": [1.0, 0.0],
            "store traffic": [1.0, 0.0],
            "How is traffic trending?": [1.0, 0.0],
        }
        return [vectors[text] for text in texts]


def _dossier_fixture(tmp_path: Path):
    company = tmp_path / "Walmart"
    company.mkdir()
    transcript = company / "2025_Q2_wmt_processed.txt"
    transcript.write_text("Guest traffic improved.", encoding="utf-8")
    release = tmp_path / "release.txt"
    release.write_text("Guest traffic improved before the call.", encoding="utf-8")
    manifests = [
        PublicDocumentManifest(
            source_url="https://www.sec.gov/Archives/wmt/ex991.htm",
            company_symbol="WMT",
            fiscal_year=2025,
            fiscal_quarter=2,
            source_type="sec_ex_99_1_supplemental",
            published_at=datetime(2025, 8, 1, tzinfo=UTC),
            fetched_at=datetime(2025, 8, 1, 1, tzinfo=UTC),
            raw_path=str(release),
            raw_original_path=str(tmp_path / "release.htm"),
            raw_original_content_hash="raw",
            extracted_text_path=str(release),
            content_hash="hash",
            extraction_method="html_text",
        )
    ]
    phrase_catalog = PhraseCatalog(
        entries=[
            PhraseCatalogEntry(
                company_symbol="WMT",
                fiscal_year=2025,
                fiscal_quarter=2,
                phrase="traffic",
                label="present",
                match_count=1,
            )
        ]
    )
    dossiers = [
        EventScenarioCatalog(
            event_id="WMT-2025-Q2",
            company_symbol="WMT",
            company_name="Walmart",
            llm_model="fake-model",
            topics=["store traffic"],
            analyst_questions=["How is traffic trending?"],
            management_answers=[],
            synthetic_call_snippets=[],
            target_phrase_variants={"traffic": ["guest traffic"]},
            source_rationales=[],
        )
    ]
    return tmp_path, manifests, phrase_catalog, dossiers


def test_collect_historical_news_manifests_cli_writes_manifest(
    tmp_path: Path,
    monkeypatch,
):
    examples_path = tmp_path / "examples.json"
    manifest_out = tmp_path / "news-manifests.json"
    summary_out = tmp_path / "news-summary.json"
    example = HistoricalTrainingExample(
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        evidence_cutoff=datetime(2025, 5, 10, 12, 0, tzinfo=UTC),
        market_id="SYNTH-WMT-2025Q2-ads",
        target_phrase="advertising",
        label=1,
        features={},
        document_ids=[],
        market_probability=Decimal("0.50"),
        market_venue="synthetic",
    )
    examples_path.write_text(
        json.dumps([example.model_dump(mode="json")]),
        encoding="utf-8",
    )

    class FakeDefeatBetaClient:
        def __init__(self, **kwargs):
            pass

        def search_stock_news(self, **kwargs):
            return [
                DefeatBetaArticle(
                    article_id="news-1",
                    title="Walmart ad business preview",
                    link="https://example.com/news-1",
                    source_name="Reuters",
                    description="Preview",
                    content="Advertising was discussed before the call.",
                    datatype="story",
                    published_at=datetime(2025, 5, 9, 12, 0, tzinfo=UTC),
                    tickers=["WMT"],
                )
            ]

    monkeypatch.setattr("kalorie.app.cli.DefeatBetaNewsClient", FakeDefeatBetaClient)

    result = CliRunner().invoke(
        app,
        [
            "collect-historical-news-manifests",
            "--examples",
            str(examples_path),
            "--raw-dir",
            str(tmp_path / "raw"),
            "--manifest-out",
            str(manifest_out),
            "--summary-out",
            str(summary_out),
            "--no-use-yfinance-fallback",
        ],
    )

    assert result.exit_code == 0, result.output
    manifests = json.loads(manifest_out.read_text(encoding="utf-8"))
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert manifests[0]["source_url"] == "https://example.com/news-1"
    assert summary["manifest_count"] == 1
