import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from typer.testing import CliRunner

from kalorie.app.cli import app
from kalorie.data_grepping.event_scenarios import EventScenarioCatalog
from kalorie.domain.models import MentionMarketContract, TargetPhrase
from kalorie.io.public_documents import PublicDocumentManifest
from kalorie.workflows.real_event_rows import build_real_event_pack_training_rows


def test_build_real_event_pack_training_rows_preserves_real_market_metadata(tmp_path: Path):
    event_dir = tmp_path / "KXEARNINGSMENTIONWMT-26AUG"
    (event_dir / "transcript").mkdir(parents=True)
    transcript_path = event_dir / "transcript" / "transcript.txt"
    transcript_path.write_text("Walmart talked about traffic but not robotaxi.", encoding="utf-8")
    release_path = event_dir / "release.txt"
    release_path.write_text("Traffic was strong before the call.", encoding="utf-8")
    contract = MentionMarketContract(
        venue="kalshi",
        market_id="KXEARNINGSMENTIONWMT-26AUG-TFFIC",
        event_ticker="KXEARNINGSMENTIONWMT-26AUG",
        title='Will WMT mention "traffic"?',
        rules_text='If traffic is said by any Walmart representative.',
        target_phrase=TargetPhrase(phrase="traffic", normalized_phrase="traffic"),
        yes_bid=Decimal("0.40"),
        yes_ask=Decimal("0.99"),
        observed_at=datetime(2026, 8, 21, 15, 50, tzinfo=UTC),
    )
    manifest = PublicDocumentManifest(
        source_url="https://www.sec.gov/Archives/wmt/ex991.htm",
        company_symbol="WMT",
        fiscal_year=2026,
        fiscal_quarter=2,
        source_type="sec_ex_99_1_supplemental",
        published_at=datetime(2026, 8, 21, 12, tzinfo=UTC),
        fetched_at=datetime(2026, 8, 21, 12, 1, tzinfo=UTC),
        raw_path=str(release_path),
        raw_original_path=str(event_dir / "release.htm"),
        raw_original_content_hash="raw",
        extracted_text_path=str(release_path),
        content_hash="hash",
        extraction_method="html_text",
    )
    (event_dir / "event.json").write_text(
        json.dumps(
            {
                "event_ticker": "KXEARNINGSMENTIONWMT-26AUG",
                "company_symbol": "WMT",
                "company_name": "Walmart",
                "fiscal_year": 2026,
                "fiscal_quarter": 2,
                "call_start_at": "2026-08-21T16:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (event_dir / "contracts.json").write_text(
        json.dumps([contract.model_dump(mode="json")]),
        encoding="utf-8",
    )
    (event_dir / "evidence-manifests.json").write_text(
        json.dumps([manifest.model_dump(mode="json")]),
        encoding="utf-8",
    )
    (event_dir / "snapshots.json").write_text(
        json.dumps(
            [
                {
                    "event_ticker": "KXEARNINGSMENTIONWMT-26AUG",
                    "market_id": "KXEARNINGSMENTIONWMT-26AUG-TFFIC",
                    "snapshot_target_time": "2026-08-21T15:50:00+00:00",
                    "candle_end_ts": int(datetime(2026, 8, 21, 15, 49, tzinfo=UTC).timestamp()),
                    "yes_bid": "0.41",
                    "yes_ask": "0.46",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = build_real_event_pack_training_rows(tmp_path)

    assert len(result.examples) == 1
    example = result.examples[0]
    assert example.market_venue == "kalshi"
    assert example.market_id == "KXEARNINGSMENTIONWMT-26AUG-TFFIC"
    assert example.event_ticker == "KXEARNINGSMENTIONWMT-26AUG"
    assert example.market_probability == Decimal("0.46")
    assert example.label == 1
    assert not result.skipped_records


class _SimpleDossierEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "traffic": [1.0, 0.0],
            "guest traffic": [1.0, 0.0],
            "Traffic was strong before the call.": [1.0, 0.0],
            "Traffic was strong before preclose.": [1.0, 0.0],
            "store traffic": [1.0, 0.0],
            "How is traffic trending?": [1.0, 0.0],
        }
        return [vectors[text] for text in texts]


def test_build_real_event_pack_training_rows_uses_event_dossiers(tmp_path: Path):
    event_dir = tmp_path / "KXEARNINGSMENTIONWMT-26AUG"
    (event_dir / "transcript").mkdir(parents=True)
    (event_dir / "transcript" / "transcript.txt").write_text(
        "Walmart talked about guest traffic.",
        encoding="utf-8",
    )
    release_path = event_dir / "release.txt"
    release_path.write_text("Traffic was strong before the call.", encoding="utf-8")
    contract = MentionMarketContract(
        venue="kalshi",
        market_id="KXEARNINGSMENTIONWMT-26AUG-TFFIC",
        event_ticker="KXEARNINGSMENTIONWMT-26AUG",
        title='Will WMT mention "traffic"?',
        rules_text='If traffic is said by any Walmart representative.',
        target_phrase=TargetPhrase(phrase="traffic", normalized_phrase="traffic"),
        yes_bid=Decimal("0.40"),
        yes_ask=Decimal("0.50"),
        observed_at=datetime(2026, 8, 21, 15, 50, tzinfo=UTC),
    )
    manifest = PublicDocumentManifest(
        source_url="https://www.sec.gov/Archives/wmt/ex991.htm",
        company_symbol="WMT",
        fiscal_year=2026,
        fiscal_quarter=2,
        source_type="sec_ex_99_1_supplemental",
        published_at=datetime(2026, 8, 21, 12, tzinfo=UTC),
        fetched_at=datetime(2026, 8, 21, 12, 1, tzinfo=UTC),
        raw_path=str(release_path),
        raw_original_path=str(event_dir / "release.htm"),
        raw_original_content_hash="raw",
        extracted_text_path=str(release_path),
        content_hash="hash",
        extraction_method="html_text",
    )
    (event_dir / "event.json").write_text(
        json.dumps(
            {
                "event_ticker": "KXEARNINGSMENTIONWMT-26AUG",
                "company_symbol": "WMT",
                "company_name": "Walmart",
                "fiscal_year": 2026,
                "fiscal_quarter": 2,
                "call_start_at": "2026-08-21T16:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (event_dir / "contracts.json").write_text(
        json.dumps([contract.model_dump(mode="json")]),
        encoding="utf-8",
    )
    (event_dir / "evidence-manifests.json").write_text(
        json.dumps([manifest.model_dump(mode="json")]),
        encoding="utf-8",
    )
    (event_dir / "snapshots.json").write_text(
        json.dumps(
            [
                {
                    "event_ticker": "KXEARNINGSMENTIONWMT-26AUG",
                    "market_id": "KXEARNINGSMENTIONWMT-26AUG-TFFIC",
                    "snapshot_target_time": "2026-08-21T15:50:00+00:00",
                    "candle_end_ts": int(datetime(2026, 8, 21, 15, 49, tzinfo=UTC).timestamp()),
                    "yes_bid": "0.40",
                    "yes_ask": "0.50",
                }
            ]
        ),
        encoding="utf-8",
    )
    dossiers = [
        EventScenarioCatalog(
            event_id="WMT-2026-Q2",
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

    result = build_real_event_pack_training_rows(
        tmp_path,
        event_dossiers=dossiers,
        embedding_provider=_SimpleDossierEmbeddingProvider(),
    )

    features = result.examples[0].features
    assert features["scenario_text_count"] == 2.0
    assert features["template_phrase_count"] == 1.0


def test_build_kalshi_event_pack_training_rows_cli_accepts_event_dossiers(
    tmp_path: Path,
    monkeypatch,
):
    _write_event_pack_fixture(tmp_path)
    dossiers_path = tmp_path / "event-dossiers.json"
    out_path = tmp_path / "examples.json"
    dossiers_path.write_text(
        json.dumps(
            [
                EventScenarioCatalog(
                    event_id="WMT-2026-Q2",
                    company_symbol="WMT",
                    company_name="Walmart",
                    llm_model="fake-model",
                    topics=["store traffic"],
                    analyst_questions=["How is traffic trending?"],
                    management_answers=[],
                    synthetic_call_snippets=[],
                    target_phrase_variants={"traffic": ["guest traffic"]},
                    source_rationales=[],
                ).model_dump(mode="json")
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kalorie.app.cli._load_embedding_provider",
        lambda _message: _SimpleDossierEmbeddingProvider(),
    )

    result = CliRunner().invoke(
        app,
        [
            "build-kalshi-event-pack-training-rows",
            "--pack-dir",
            str(tmp_path),
            "--event-dossiers",
            str(dossiers_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    rows = json.loads(out_path.read_text(encoding="utf-8"))
    assert rows[0]["features"]["scenario_text_count"] == 2.0
    assert rows[0]["features"]["template_phrase_count"] == 1.0


def test_build_real_event_pack_training_rows_accepts_raw_collected_pack(tmp_path: Path):
    event_dir = tmp_path / "kxearningsmentionwmt-26aug21"
    event_dir.mkdir(parents=True)
    transcript_path = event_dir / "transcript.txt"
    transcript_path.write_text("Walmart talked about guest traffic.", encoding="utf-8")
    before_path = event_dir / "before-release.txt"
    before_path.write_text("Traffic was strong before preclose.", encoding="utf-8")
    after_path = event_dir / "after-release.txt"
    after_path.write_text("Post-call traffic details should not be available.", encoding="utf-8")
    contract = MentionMarketContract(
        venue="kalshi",
        market_id="KXEARNINGSMENTIONWMT-26AUG-TFFIC",
        event_ticker="KXEARNINGSMENTIONWMT-26AUG",
        title='What will Walmart say during their next earnings call?',
        rules_text='If traffic is said by any Walmart representative.',
        target_phrase=TargetPhrase(phrase="traffic", normalized_phrase="traffic"),
        yes_bid=Decimal("0.40"),
        yes_ask=Decimal("0.50"),
        observed_at=datetime(2026, 8, 21, 15, 50, tzinfo=UTC),
    )
    (event_dir / "event.json").write_text(
        json.dumps({"event": {"event_ticker": "KXEARNINGSMENTIONWMT-26AUG"}}),
        encoding="utf-8",
    )
    (event_dir / "contracts-preclose.json").write_text(
        json.dumps([contract.model_dump(mode="json")]),
        encoding="utf-8",
    )
    (event_dir / "transcript_manifest.json").write_text(
        json.dumps(
            PublicDocumentManifest(
                source_url="https://example.com/transcript",
                company_symbol="WMT",
                fiscal_year=2026,
                fiscal_quarter=2,
                source_type="earnings_call_transcript_web",
                published_at=datetime(2026, 8, 21, 16, 30, tzinfo=UTC),
                fetched_at=datetime(2026, 8, 21, 16, 31, tzinfo=UTC),
                raw_path=str(transcript_path),
                raw_original_path=str(transcript_path),
                raw_original_content_hash="transcript-raw",
                extracted_text_path=str(transcript_path),
                content_hash="transcript-hash",
                extraction_method="html_text",
            ).model_dump(mode="json")
        ),
        encoding="utf-8",
    )
    before_manifest = PublicDocumentManifest(
        source_url="https://example.com/before",
        company_symbol="WMT",
        fiscal_year=2026,
        fiscal_quarter=2,
        source_type="sec_ex_99_1_press_release",
        published_at=datetime(2026, 8, 21, 15, 49, tzinfo=UTC),
        fetched_at=datetime(2026, 8, 21, 15, 49, tzinfo=UTC),
        raw_path=str(before_path),
        raw_original_path=str(before_path),
        raw_original_content_hash="before-raw",
        extracted_text_path=str(before_path),
        content_hash="before-hash",
        extraction_method="html_text",
    )
    after_manifest = before_manifest.model_copy(
        update={
            "source_url": "https://example.com/after",
            "published_at": datetime(2026, 8, 21, 15, 51, tzinfo=UTC),
            "raw_path": str(after_path),
            "extracted_text_path": str(after_path),
            "content_hash": "after-hash",
        }
    )
    (event_dir / "sec_manifests.json").write_text(
        json.dumps([before_manifest.model_dump(mode="json"), after_manifest.model_dump(mode="json")]),
        encoding="utf-8",
    )
    (event_dir / "preclose_snapshots.json").write_text(
        json.dumps(
            [
                {
                    "event_ticker": "KXEARNINGSMENTIONWMT-26AUG",
                    "market_id": "KXEARNINGSMENTIONWMT-26AUG-TFFIC",
                    "snapshot_target_time": "2026-08-21T15:50:00+00:00",
                    "candle_end_ts": int(datetime(2026, 8, 21, 15, 49, tzinfo=UTC).timestamp()),
                    "preclose_yes_bid": "0.40",
                    "preclose_yes_ask": "0.50",
                }
            ]
        ),
        encoding="utf-8",
    )
    dossiers = [
        EventScenarioCatalog(
            event_id="WMT-2026-Q2",
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

    result = build_real_event_pack_training_rows(
        tmp_path,
        event_dossiers=dossiers,
        embedding_provider=_SimpleDossierEmbeddingProvider(),
    )

    assert len(result.examples) == 1
    example = result.examples[0]
    assert example.features["scenario_text_count"] == 2.0
    assert example.features["template_phrase_count"] == 1.0
    assert len(example.document_ids) == 1


def _write_event_pack_fixture(tmp_path: Path) -> None:
    event_dir = tmp_path / "KXEARNINGSMENTIONWMT-26AUG"
    (event_dir / "transcript").mkdir(parents=True)
    (event_dir / "transcript" / "transcript.txt").write_text(
        "Walmart talked about guest traffic.",
        encoding="utf-8",
    )
    release_path = event_dir / "release.txt"
    release_path.write_text("Traffic was strong before the call.", encoding="utf-8")
    contract = MentionMarketContract(
        venue="kalshi",
        market_id="KXEARNINGSMENTIONWMT-26AUG-TFFIC",
        event_ticker="KXEARNINGSMENTIONWMT-26AUG",
        title='Will WMT mention "traffic"?',
        rules_text='If traffic is said by any Walmart representative.',
        target_phrase=TargetPhrase(phrase="traffic", normalized_phrase="traffic"),
        yes_bid=Decimal("0.40"),
        yes_ask=Decimal("0.50"),
        observed_at=datetime(2026, 8, 21, 15, 50, tzinfo=UTC),
    )
    manifest = PublicDocumentManifest(
        source_url="https://www.sec.gov/Archives/wmt/ex991.htm",
        company_symbol="WMT",
        fiscal_year=2026,
        fiscal_quarter=2,
        source_type="sec_ex_99_1_supplemental",
        published_at=datetime(2026, 8, 21, 12, tzinfo=UTC),
        fetched_at=datetime(2026, 8, 21, 12, 1, tzinfo=UTC),
        raw_path=str(release_path),
        raw_original_path=str(event_dir / "release.htm"),
        raw_original_content_hash="raw",
        extracted_text_path=str(release_path),
        content_hash="hash",
        extraction_method="html_text",
    )
    (event_dir / "event.json").write_text(
        json.dumps(
            {
                "event_ticker": "KXEARNINGSMENTIONWMT-26AUG",
                "company_symbol": "WMT",
                "company_name": "Walmart",
                "fiscal_year": 2026,
                "fiscal_quarter": 2,
                "call_start_at": "2026-08-21T16:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (event_dir / "contracts.json").write_text(
        json.dumps([contract.model_dump(mode="json")]),
        encoding="utf-8",
    )
    (event_dir / "evidence-manifests.json").write_text(
        json.dumps([manifest.model_dump(mode="json")]),
        encoding="utf-8",
    )
    (event_dir / "snapshots.json").write_text(
        json.dumps(
            [
                {
                    "event_ticker": "KXEARNINGSMENTIONWMT-26AUG",
                    "market_id": "KXEARNINGSMENTIONWMT-26AUG-TFFIC",
                    "snapshot_target_time": "2026-08-21T15:50:00+00:00",
                    "candle_end_ts": int(datetime(2026, 8, 21, 15, 49, tzinfo=UTC).timestamp()),
                    "yes_bid": "0.40",
                    "yes_ask": "0.50",
                }
            ]
        ),
        encoding="utf-8",
    )
