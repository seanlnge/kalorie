import json
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import httpx
from typer.testing import CliRunner

from kalorie.app.cli import app
from kalorie.clients.financial_modeling_prep import FmpTranscriptReference
from kalorie.clients.sec_api import SecApiFiling, SecApiRateLimitError, SecCompanyMapping
from kalorie.domain.config import Settings
from kalorie.domain.models import FeatureVector, MentionMarketContract, TargetPhrase
from kalorie.io.public_documents import PublicDocumentManifest, collect_public_document
from kalorie.ml.datasets import HistoricalTrainingExample


def _example(company: str, year: int, quarter: int, phrase: str, label: int):
    return HistoricalTrainingExample(
        company_symbol=company,
        fiscal_year=year,
        fiscal_quarter=quarter,
        evidence_cutoff=datetime(year, quarter, 1, tzinfo=UTC),
        market_id=f"{company}-{year}-Q{quarter}-{phrase}",
        target_phrase=phrase,
        label=label,
        features={
            "exact_match_count": float(label),
            "max_tfidf_similarity": 0.8 if label else 0.0,
            "appears_in_headline_or_first_chunk": float(label),
        },
        document_ids=[f"{company}-{year}-Q{quarter}-press"],
        market_probability=Decimal("0.50"),
    )


def test_train_model_cli_outputs_brier_and_log_loss(tmp_path):
    examples = [
        _example("CAVA", 2025, 1, "traffic", 1),
        _example("CAVA", 2025, 2, "robotaxi", 0),
        _example("NVDA", 2025, 1, "ai", 1),
        _example("NVDA", 2025, 2, "tariffs", 0),
        _example("CAVA", 2026, 1, "traffic", 1),
        _example("NVDA", 2026, 1, "robotaxi", 0),
    ]
    examples_path = tmp_path / "examples.json"
    out_path = tmp_path / "report.json"
    examples_path.write_text(
        json.dumps([example.model_dump(mode="json") for example in examples]),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["train-model", "--examples", str(examples_path), "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    assert "Global Brier score:" in result.output
    assert "MSE:" not in result.output
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert "global_mean_squared_error" not in report
    assert report["sample_count"] == 2


def test_train_and_predict_model1_cli(tmp_path):
    examples = [
        _example("AAPL", 2024, 1, "ai", 1),
        _example("AAPL", 2024, 2, "ai", 0),
        _example("AAPL", 2024, 3, "ai", 1),
        _example("AAPL", 2024, 4, "ai", 0),
        _example("MSFT", 2024, 1, "cloud", 1),
        _example("MSFT", 2024, 2, "cloud", 0),
        _example("MSFT", 2024, 3, "cloud", 1),
        _example("MSFT", 2024, 4, "cloud", 0),
    ]
    examples_path = tmp_path / "examples.json"
    model_path = tmp_path / "model1.json"
    features_path = tmp_path / "features.json"
    prediction_path = tmp_path / "predictions.json"

    examples_path.write_text(
        json.dumps([example.model_dump(mode="json") for example in examples]),
        encoding="utf-8",
    )
    features_path.write_text(
        json.dumps(
            [
                FeatureVector(
                    target_phrase="ai",
                    features={
                        "exact_match_count": 1.0,
                        "lexical_match_count": 0.0,
                        "max_tfidf_similarity": 0.8,
                        "appears_in_headline_or_first_chunk": 1.0,
                    },
                ).model_dump(mode="json")
            ]
        ),
        encoding="utf-8",
    )

    train_result = CliRunner().invoke(
        app,
        [
            "train-model1",
            "--examples",
            str(examples_path),
            "--out",
            str(model_path),
            "--min-company-rows",
            "4",
        ],
    )
    assert train_result.exit_code == 0, train_result.output

    predict_result = CliRunner().invoke(
        app,
        [
            "predict-model1",
            "--model",
            str(model_path),
            "--features",
            str(features_path),
            "--company-symbol",
            "AAPL",
            "--out",
            str(prediction_path),
        ],
    )
    assert predict_result.exit_code == 0, predict_result.output
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert len(predictions) == 1
    assert predictions[0]["model_version"] == "mention-base-company-v1"


def test_train_model1_cli_can_fail_strict_pretrain_gate(tmp_path):
    examples = [_example("AAPL", 2024, 1, "ai", 1)]
    examples_path = tmp_path / "examples.json"
    model_path = tmp_path / "model1.json"
    gate_path = tmp_path / "gate.json"
    examples_path.write_text(
        json.dumps([example.model_dump(mode="json") for example in examples]),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "train-model1",
            "--examples",
            str(examples_path),
            "--out",
            str(model_path),
            "--enforce-pretrain-gate",
            "--pretrain-gate-report-out",
            str(gate_path),
        ],
    )

    assert result.exit_code != 0
    assert "pre-train gate failed" in result.output
    gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate_payload["passed"] is False
    assert gate_payload["metrics"]["sample_count"] == 1


def test_train_and_predict_model1_company_cli(tmp_path):
    examples = [
        _example("AAPL", 2024, 1, "ai", 1),
        _example("AAPL", 2024, 2, "ai", 0),
        _example("AAPL", 2024, 3, "ai", 1),
        _example("AAPL", 2024, 4, "ai", 0),
        _example("MSFT", 2024, 1, "cloud", 1),
        _example("MSFT", 2024, 2, "cloud", 0),
    ]
    examples_path = tmp_path / "examples.json"
    model_path = tmp_path / "model1-company-aapl.json"
    features_path = tmp_path / "features.json"
    prediction_path = tmp_path / "predictions.json"

    examples_path.write_text(
        json.dumps([example.model_dump(mode="json") for example in examples]),
        encoding="utf-8",
    )
    features_path.write_text(
        json.dumps(
            [
                FeatureVector(
                    target_phrase="ai",
                    features={
                        "exact_match_count": 1.0,
                        "lexical_match_count": 0.0,
                        "max_tfidf_similarity": 0.8,
                        "appears_in_headline_or_first_chunk": 1.0,
                    },
                ).model_dump(mode="json")
            ]
        ),
        encoding="utf-8",
    )

    train_result = CliRunner().invoke(
        app,
        [
            "train-model1-company",
            "--examples",
            str(examples_path),
            "--company-symbol",
            "AAPL",
            "--out",
            str(model_path),
            "--min-company-rows",
            "4",
        ],
    )
    assert train_result.exit_code == 0, train_result.output

    predict_result = CliRunner().invoke(
        app,
        [
            "predict-model1-company",
            "--model",
            str(model_path),
            "--features",
            str(features_path),
            "--out",
            str(prediction_path),
        ],
    )
    assert predict_result.exit_code == 0, predict_result.output
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert len(predictions) == 1
    assert predictions[0]["model_version"] == "mention-company-retrained-v1"


def test_train_model1_optimized_and_evaluate_cli(tmp_path):
    examples = []
    for year, quarter in [(2023, 1), (2023, 2), (2023, 3), (2023, 4), (2024, 1), (2024, 2)]:
        examples.extend(
            [
                _example("AAPL", year, quarter, "ai", 1 if quarter % 2 else 0),
                _example("MSFT", year, quarter, "cloud", 1 if quarter in {1, 3} else 0),
                _example("NVDA", year, quarter, "gpu", 1 if quarter in {2, 4} else 0),
            ]
        )
    examples_path = tmp_path / "examples.json"
    model_path = tmp_path / "model1-optimized.json"
    report_path = tmp_path / "optimization-report.json"
    eval_path = tmp_path / "model1-eval.json"
    examples_path.write_text(
        json.dumps([example.model_dump(mode="json") for example in examples]),
        encoding="utf-8",
    )

    optimize_result = CliRunner().invoke(
        app,
        [
            "train-model1-optimized",
            "--examples",
            str(examples_path),
            "--out",
            str(model_path),
            "--report-out",
            str(report_path),
            "--regularization-grid",
            "0.1,1.0",
            "--min-company-rows-grid",
            "4,8",
            "--blend-weight-grid",
            "0.25,0.5",
            "--class-weight-balanced-values",
            "false,true",
            "--include-target-indicator-values",
            "false",
            "--no-enable-isotonic-calibration",
        ],
    )
    assert optimize_result.exit_code == 0, optimize_result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["best_trial"]["holdout_brier_score"] >= 0
    model = json.loads(model_path.read_text(encoding="utf-8"))
    assert model["model_version"] == "mention-base-company-v2"

    eval_result = CliRunner().invoke(
        app,
        [
            "evaluate-model1",
            "--model",
            str(model_path),
            "--examples",
            str(examples_path),
            "--out",
            str(eval_path),
        ],
    )
    assert eval_result.exit_code == 0, eval_result.output
    eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    assert "brier_score" in eval_payload
    assert "expected_calibration_error" in eval_payload
    assert eval_payload["sample_count"] == len(examples)


def test_build_synthetic_phrase_dataset_cli_does_not_require_kalshi_contracts(tmp_path):
    transcript_root = tmp_path / "transcripts"
    walmart = transcript_root / "Walmart"
    walmart.mkdir(parents=True)
    (walmart / "2025_Q2_wmt_processed.txt").write_text(
        "Traffic improved and automation helped.",
        encoding="utf-8",
    )
    release_path = tmp_path / "release.txt"
    release_path.write_text("Automation investment improved operations.", encoding="utf-8")
    manifest = PublicDocumentManifest(
        source_url="https://www.sec.gov/wmt-release.htm",
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        source_type="sec_ex_99_1_supplemental",
        published_at=datetime(2025, 8, 21, tzinfo=UTC),
        fetched_at=datetime(2025, 8, 21, tzinfo=UTC),
        raw_path=str(release_path),
        content_hash="abc",
        extraction_method="html_text",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([manifest.model_dump(mode="json")]), encoding="utf-8")
    out_path = tmp_path / "synthetic-examples.json"

    result = CliRunner().invoke(
        app,
        [
            "build-synthetic-phrase-dataset",
            "--transcript-root",
            str(transcript_root),
            "--manifests",
            str(manifest_path),
            "--target-phrases",
            "traffic,automation,robotaxi",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    rows = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(rows) == 3
    assert {row["company_symbol"] for row in rows} == {"WMT"}
    assert {row["target_phrase"] for row in rows} == {"traffic", "automation", "robotaxi"}


def test_build_synthetic_phrase_dataset_with_template_catalog_adds_template_features(
    tmp_path,
    monkeypatch,
):
    transcript_root = tmp_path / "transcripts"
    walmart = transcript_root / "Walmart"
    walmart.mkdir(parents=True)
    (walmart / "2025_Q2_wmt_processed.txt").write_text(
        "Guest traffic improved and automation helped operations.",
        encoding="utf-8",
    )
    release_path = tmp_path / "release.txt"
    release_path.write_text("Automation investment improved operations.", encoding="utf-8")
    manifest = PublicDocumentManifest(
        source_url="https://www.sec.gov/wmt-release.htm",
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        source_type="sec_ex_99_1_supplemental",
        published_at=datetime(2025, 8, 21, tzinfo=UTC),
        fetched_at=datetime(2025, 8, 21, tzinfo=UTC),
        raw_path=str(release_path),
        content_hash="abc",
        extraction_method="html_text",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([manifest.model_dump(mode="json")]), encoding="utf-8")
    template_catalog_path = tmp_path / "template-catalog.json"
    template_catalog_path.write_text(
        json.dumps(
            {
                "llm_model": "fake",
                "phrase_variants": {"traffic": ["traffic growth", "guest traffic momentum"]},
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "synthetic-examples.json"

    class _TemplateEmbeddingProvider:
        def embed(self, texts: list[str]) -> list[list[float]]:
            vectors = {
                "traffic growth": [1.0, 0.0],
                "guest traffic momentum": [1.0, 0.0],
                "Automation investment improved operations.": [0.0, 1.0],
            }
            return [vectors[text] for text in texts]

    monkeypatch.setattr(
        "kalorie.app.cli._load_template_features_context",
        lambda _catalog: (
            {"traffic": ["traffic growth", "guest traffic momentum"]},
            _TemplateEmbeddingProvider(),
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "build-synthetic-phrase-dataset",
            "--transcript-root",
            str(transcript_root),
            "--manifests",
            str(manifest_path),
            "--template-catalog",
            str(template_catalog_path),
            "--target-phrases",
            "traffic",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    rows = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["features"]["template_phrase_count"] == 2.0
    assert "max_template_embedding_similarity" in rows[0]["features"]


def test_build_synthetic_phrase_dataset_expands_global_targets_from_market_contracts(tmp_path):
    transcript_root = tmp_path / "transcripts"
    walmart = transcript_root / "Walmart"
    costco = transcript_root / "Costco"
    walmart.mkdir(parents=True)
    costco.mkdir(parents=True)
    (walmart / "2025_Q2_wmt_processed.txt").write_text(
        "Traffic and omnichannel trends improved.",
        encoding="utf-8",
    )
    (costco / "2025_Q2_cost_processed.txt").write_text(
        "Traffic and omnichannel trends improved.",
        encoding="utf-8",
    )

    wmt_release = tmp_path / "wmt-release.txt"
    wmt_release.write_text("Omnichannel investments improved operations.", encoding="utf-8")
    cost_release = tmp_path / "cost-release.txt"
    cost_release.write_text("Omnichannel investments improved operations.", encoding="utf-8")

    manifest_rows = [
        PublicDocumentManifest(
            source_url="https://example.com/wmt-release.htm",
            company_symbol="WMT",
            fiscal_year=2025,
            fiscal_quarter=2,
            source_type="sec_ex_99_1_supplemental",
            published_at=datetime(2025, 8, 21, tzinfo=UTC),
            fetched_at=datetime(2025, 8, 21, tzinfo=UTC),
            raw_path=str(wmt_release),
            content_hash="wmt",
            extraction_method="html_text",
        ),
        PublicDocumentManifest(
            source_url="https://example.com/cost-release.htm",
            company_symbol="COST",
            fiscal_year=2025,
            fiscal_quarter=2,
            source_type="sec_ex_99_1_supplemental",
            published_at=datetime(2025, 8, 21, tzinfo=UTC),
            fetched_at=datetime(2025, 8, 21, tzinfo=UTC),
            raw_path=str(cost_release),
            content_hash="cost",
            extraction_method="html_text",
        ),
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([row.model_dump(mode="json") for row in manifest_rows]))

    contracts = [
        MentionMarketContract(
            venue="kalshi",
            market_id="KXEARNINGSMENTIONWMT-26MAY15-OMNI",
            event_ticker="KXEARNINGSMENTIONWMT-26MAY15",
            title="What will Walmart say during their next earnings call?",
            rules_text="If omnichannel is said by any Walmart representative.",
            target_phrase=TargetPhrase(phrase="omnichannel", normalized_phrase="omnichannel"),
            yes_bid=Decimal("0.40"),
            yes_ask=Decimal("0.45"),
            observed_at=datetime(2026, 5, 15, tzinfo=UTC),
        )
    ]
    contracts_path = tmp_path / "contracts.json"
    contracts_path.write_text(
        json.dumps([contract.model_dump(mode="json") for contract in contracts]),
        encoding="utf-8",
    )
    out_path = tmp_path / "synthetic-examples.json"

    result = CliRunner().invoke(
        app,
        [
            "build-synthetic-phrase-dataset",
            "--transcript-root",
            str(transcript_root),
            "--manifests",
            str(manifest_path),
            "--target-phrases",
            "traffic",
            "--market-contracts",
            str(contracts_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    rows = json.loads(out_path.read_text(encoding="utf-8"))
    targets_by_company = {
        symbol: sorted({row["target_phrase"] for row in rows if row["company_symbol"] == symbol})
        for symbol in {"WMT", "COST"}
    }
    assert targets_by_company["WMT"] == ["omnichannel", "traffic"]
    assert targets_by_company["COST"] == ["omnichannel", "traffic"]


def test_generate_template_phrases_shows_parallel_progress(tmp_path, monkeypatch):
    raw_path = tmp_path / "release.txt"
    raw_path.write_text("Traffic and margin were discussed.", encoding="utf-8")
    manifest = PublicDocumentManifest(
        source_url="https://www.sec.gov/wmt-release.htm",
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        source_type="sec_ex_99_1_supplemental",
        published_at=datetime(2025, 8, 21, tzinfo=UTC),
        fetched_at=datetime(2025, 8, 21, tzinfo=UTC),
        raw_path=str(raw_path),
        content_hash="abc",
        extraction_method="html_text",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([manifest.model_dump(mode="json")]), encoding="utf-8")
    out_path = tmp_path / "template-catalog.json"

    class _FakeGenerator:
        def __init__(self, *, api_key=None, model="fake", client=None):
            self._model = model

        def generate(self, *, target_phrase: str, material_snippets, max_variants: int = 12):
            return [f"{target_phrase} template"]

        @property
        def model(self):
            return self._model

    monkeypatch.setattr("kalorie.app.cli._load_settings", lambda: Settings(openai_api_key="x"))
    monkeypatch.setattr("kalorie.app.cli.OpenAITemplatePhraseGenerator", _FakeGenerator)

    result = CliRunner().invoke(
        app,
        [
            "generate-template-phrases",
            "--manifests",
            str(manifest_path),
            "--target-phrases",
            "traffic,margin,ai",
            "--max-concurrency",
            "3",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Generating template variants for 3 targets" in result.output
    assert "[1/3] queued 'traffic'" in result.output
    assert "done 'traffic'" in result.output
    catalog = json.loads(out_path.read_text(encoding="utf-8"))
    assert sorted(catalog["phrase_variants"]) == ["ai", "margin", "traffic"]


def test_run_base_ablation_harness_writes_variant_summary(tmp_path, monkeypatch):
    from kalorie.app import cli as cli_module

    transcript_root = tmp_path / "transcripts"
    walmart = transcript_root / "Walmart"
    walmart.mkdir(parents=True)
    (walmart / "2025_Q2_wmt_processed.txt").write_text("Traffic improved.", encoding="utf-8")
    sec_release = tmp_path / "sec-release.txt"
    sec_release.write_text("Traffic improved.", encoding="utf-8")
    news_release = tmp_path / "news-release.txt"
    news_release.write_text("Traffic improved.", encoding="utf-8")

    sec_manifest = PublicDocumentManifest(
        source_url="https://example.com/wmt-sec.htm",
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        source_type="sec_ex_99_1_supplemental",
        published_at=datetime(2025, 8, 21, tzinfo=UTC),
        fetched_at=datetime(2025, 8, 21, tzinfo=UTC),
        raw_path=str(sec_release),
        content_hash="sec",
        extraction_method="html_text",
    )
    news_manifest = PublicDocumentManifest(
        source_url="https://example.com/wmt-news",
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        source_type="news_article_relevant_reliability_070",
        published_at=datetime(2025, 8, 20, tzinfo=UTC),
        fetched_at=datetime(2025, 8, 20, tzinfo=UTC),
        raw_path=str(news_release),
        extracted_text_path=str(news_release),
        content_hash="news",
        extraction_method="newsdata_api",
    )
    sec_manifest_path = tmp_path / "sec-manifests.json"
    sec_manifest_path.write_text(json.dumps([sec_manifest.model_dump(mode="json")]), encoding="utf-8")
    news_manifest_path = tmp_path / "news-manifests.json"
    news_manifest_path.write_text(
        json.dumps([news_manifest.model_dump(mode="json")]),
        encoding="utf-8",
    )
    contracts = [
        MentionMarketContract(
            venue="kalshi",
            market_id="KXEARNINGSMENTIONWMT-26MAY15-OMNI",
            event_ticker="KXEARNINGSMENTIONWMT-26MAY15",
            title="What will Walmart say during their next earnings call?",
            rules_text="If omnichannel is said by any Walmart representative.",
            target_phrase=TargetPhrase(phrase="omnichannel", normalized_phrase="omnichannel"),
            yes_bid=Decimal("0.40"),
            yes_ask=Decimal("0.45"),
            observed_at=datetime(2026, 5, 15, tzinfo=UTC),
        )
    ]
    contracts_path = tmp_path / "contracts.json"
    contracts_path.write_text(
        json.dumps([contract.model_dump(mode="json") for contract in contracts]),
        encoding="utf-8",
    )

    examples = []
    for year, quarter in [(2023, 1), (2023, 2), (2023, 3), (2023, 4), (2024, 1), (2024, 2)]:
        examples.append(_example("WMT", year, quarter, "traffic", 1 if quarter % 2 else 0))
        examples.append(_example("NVDA", year, quarter, "ai", 1 if quarter in {1, 4} else 0))

    calls: list[dict] = []

    def fake_build_synthetic_phrase_examples_from_transcript_records(**kwargs):
        calls.append(kwargs)
        return examples

    monkeypatch.setattr(
        cli_module,
        "build_synthetic_phrase_examples_from_transcript_records",
        fake_build_synthetic_phrase_examples_from_transcript_records,
    )

    out_dir = tmp_path / "ablation"
    result = CliRunner().invoke(
        app,
        [
            "run-base-ablation-harness",
            "--transcript-root",
            str(transcript_root),
            "--sec-manifests",
            str(sec_manifest_path),
            "--news-manifests",
            str(news_manifest_path),
            "--market-contracts",
            str(contracts_path),
            "--target-phrases",
            "traffic,ai",
            "--out-dir",
            str(out_dir),
            "--regularization-grid",
            "1.0",
            "--min-company-rows-grid",
            "4",
            "--blend-weight-grid",
            "0.35",
            "--class-weight-balanced-values",
            "false",
            "--include-target-indicator-values",
            "false",
        ],
    )

    assert result.exit_code == 0, result.output
    summary_path = out_dir / "eval" / "base-ablation-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    variant_names = {row["variant"] for row in summary["variants"]}
    assert {
        "sec_only",
        "sec_plus_market_phrases",
        "sec_plus_news",
        "sec_plus_news_plus_market_phrases",
    }.issubset(variant_names)
    assert any("omnichannel" in call["target_phrases"] for call in calls[1:])


def test_write_json_creates_manifest_parent_directory(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<h1>Release</h1>")

    manifest = collect_public_document(
        url="https://example.com/release",
        company_symbol="TEST",
        fiscal_year=2026,
        fiscal_quarter=1,
        source_type="press_release",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        fetched_at=datetime(2026, 1, 2, tzinfo=UTC),
        raw_dir=tmp_path / "raw",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    out = tmp_path / "manifests" / "release.json"

    result = CliRunner().invoke(
        app,
        [
            "build-historical-dataset",
            "--examples",
            _write_examples(tmp_path),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    assert manifest.content_hash


def test_collect_kalshi_historical_markets_writes_raw_and_mentions(tmp_path, monkeypatch):
    from kalorie.app import cli

    class FakeClient:
        def __init__(self, timeout: int):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeKalshiPublicClient:
        def __init__(self, http_client, base_url: str):
            self.http_client = http_client
            self.base_url = base_url

        def get_historical_markets(self, *, status, search, limit, cursor=None):
            return {
                "cursor": "next",
                "markets": [
                    {
                        "ticker": "CAVA-26Q1-TRAFFIC",
                        "event_ticker": "CAVA-26Q1",
                        "title": "Will CAVA mention traffic during earnings?",
                        "rules_primary": "If traffic is said by any CAVA representative.",
                        "yes_bid": 38,
                        "yes_ask": 45,
                    }
                ],
            }

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr(cli, "KalshiPublicClient", FakeKalshiPublicClient)
    raw_out = tmp_path / "kalshi" / "raw.json"
    mentions_out = tmp_path / "kalshi" / "mentions.json"

    result = CliRunner().invoke(
        app,
        [
            "collect-kalshi-historical-markets",
            "--search",
            "CAVA",
            "--raw-out",
            str(raw_out),
            "--mentions-out",
            str(mentions_out),
            "--limit",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(raw_out.read_text(encoding="utf-8"))["cursor"] == "next"
    mentions = json.loads(mentions_out.read_text(encoding="utf-8"))
    assert mentions[0]["target_phrase"]["normalized_phrase"] == "traffic"


def test_collect_fmp_transcripts_writes_symbol_company_files(tmp_path, monkeypatch):
    from kalorie.app import cli

    transcript_root = tmp_path / "transcripts"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeFmpClient:
        def __init__(self, api_key, http_client, base_url: str):
            self.api_key = api_key
            self.http_client = http_client
            self.base_url = base_url

        def get_transcript_dates(self, symbol: str):
            if symbol == "CAVA":
                return [
                    FmpTranscriptReference(
                        symbol="CAVA",
                        fiscal_year=2026,
                        fiscal_quarter=1,
                        published_at=datetime(2026, 5, 28, tzinfo=UTC),
                    )
                ]
            return []

        def get_transcript_text(self, *, symbol: str, fiscal_year: int, fiscal_quarter: int):
            return ("Operator: Welcome to CAVA earnings.", datetime(2026, 5, 28, tzinfo=UTC))

    monkeypatch.setattr(
        cli,
        "_load_settings",
        lambda: Settings(financial_modeling_prep_api_key="fmp-secret"),
    )
    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr(cli, "FinancialModelingPrepClient", FakeFmpClient)

    result = CliRunner().invoke(
        app,
        [
            "collect-fmp-transcripts",
            "--symbols",
            "CAVA",
            "--transcript-root",
            str(transcript_root),
            "--symbol-company-names",
            "CAVA:CAVA Group, Inc.",
            "--start-year",
            "2025",
            "--end-year",
            "2026",
        ],
    )

    assert result.exit_code == 0, result.output
    transcript_path = transcript_root / "CAVA Group, Inc." / "2026_Q1_cava_processed.txt"
    assert transcript_path.exists()
    assert "Operator: Welcome" in transcript_path.read_text(encoding="utf-8")


def test_collect_newsdata_company_articles_writes_news_manifests(tmp_path, monkeypatch):
    from kalorie.app import cli

    raw_dir = tmp_path / "raw"
    manifest_out = tmp_path / "manifests" / "newsdata.json"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeNewsDataClient:
        def __init__(self, api_key, http_client, base_url: str):
            self.api_key = api_key
            self.http_client = http_client
            self.base_url = base_url

        def search_archive(
            self,
            *,
            query: str,
            from_date: str,
            to_date: str,
            language: str,
            size: int,
            max_articles: int,
        ):
            return [
                SimpleNamespace(
                    article_id="a1",
                    title="NVIDIA earnings opinion",
                    link="https://example.com/nvda-opinion",
                    source_name="Example News",
                    source_priority=2,
                    datatype="opinion",
                    description="Opinion article",
                    content="Detailed content",
                    published_at=datetime(2026, 5, 19, 15, 0, tzinfo=UTC),
                )
            ]

    monkeypatch.setattr(
        cli,
        "_load_settings",
        lambda: Settings(newsdata_api_key="news-secret"),
    )
    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr(cli, "NewsDataClient", FakeNewsDataClient)

    result = CliRunner().invoke(
        app,
        [
            "collect-newsdata-company-articles",
            "--company-symbol",
            "NVDA",
            "--company-name",
            "NVIDIA Corporation",
            "--from-date",
            "2026-05-01",
            "--to-date",
            "2026-05-20",
            "--raw-dir",
            str(raw_dir),
            "--manifest-out",
            str(manifest_out),
            "--max-articles",
            "5",
            "--size",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    manifests = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert len(manifests) == 1
    assert manifests[0]["company_symbol"] == "NVDA"
    assert manifests[0]["source_type"].startswith("news_article_opinion_reliability_")


def test_collect_tiingo_company_articles_writes_historical_manifests(tmp_path, monkeypatch):
    from kalorie.app import cli

    raw_dir = tmp_path / "raw"
    manifest_out = tmp_path / "manifests" / "tiingo.json"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeTiingoClient:
        def __init__(self, *, api_key, http_client, base_url: str):
            self.api_key = api_key
            self.http_client = http_client
            self.base_url = base_url

        def search_news(self, *, ticker: str, start_date: str, end_date: str, limit: int):
            return [
                SimpleNamespace(
                    article_id="t1",
                    title="Walmart omnichannel analysis",
                    link="https://example.com/wmt-omni",
                    source_name="Reuters",
                    source_priority=None,
                    datatype="analysis",
                    description="Historical analysis",
                    content=None,
                    published_at=datetime(2024, 5, 15, 12, 0, tzinfo=UTC),
                )
            ]

    monkeypatch.setattr(
        cli,
        "_load_settings",
        lambda: Settings(tiingo_api_key="tiingo-secret"),
    )
    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr(cli, "TiingoNewsClient", FakeTiingoClient)

    result = CliRunner().invoke(
        app,
        [
            "collect-tiingo-company-articles",
            "--company-symbol",
            "WMT",
            "--company-name",
            "Walmart",
            "--from-date",
            "2024-01-01",
            "--to-date",
            "2024-12-31",
            "--raw-dir",
            str(raw_dir),
            "--manifest-out",
            str(manifest_out),
            "--max-articles",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    manifests = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert len(manifests) == 1
    assert manifests[0]["company_symbol"] == "WMT"
    assert manifests[0]["fiscal_year"] == 2024
    assert manifests[0]["source_type"].startswith("news_article_opinion_reliability_")


def test_collect_defeatbeta_pre_earnings_week_articles_writes_manifests(tmp_path, monkeypatch):
    from kalorie.app import cli

    transcript_root = tmp_path / "transcripts"
    walmart = transcript_root / "Walmart"
    walmart.mkdir(parents=True)
    (walmart / "2025_Q1_wmt_processed.txt").write_text("Traffic commentary.", encoding="utf-8")
    (walmart / "2025_Q2_wmt_processed.txt").write_text("Omnichannel commentary.", encoding="utf-8")
    raw_dir = tmp_path / "raw"
    manifest_out = tmp_path / "manifests" / "defeatbeta.json"

    class FakeDefeatBetaClient:
        def __init__(self, *, dataset_url: str | None = None):
            self.dataset_url = dataset_url

        def search_stock_news(self, *, symbol: str, start_date: str, end_date: str, max_rows: int):
            first = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
            second = datetime.fromisoformat(end_date).replace(tzinfo=UTC)
            return [
                SimpleNamespace(
                    article_id=f"{symbol}-{start_date}",
                    title=f"{symbol} pre-earnings outlook",
                    link="https://example.com/defeatbeta",
                    source_name="Reuters",
                    source_priority=None,
                    datatype="analysis",
                    description="DefeatBeta historical article",
                    content="Paragraph one.",
                    published_at=first,
                    tickers=[symbol],
                    tags=[],
                ),
                SimpleNamespace(
                    article_id=f"{symbol}-{end_date}",
                    title=f"{symbol} earnings-week recap",
                    link="https://example.com/defeatbeta-2",
                    source_name="Reuters",
                    source_priority=None,
                    datatype="analysis",
                    description="DefeatBeta historical article",
                    content="Paragraph two.",
                    published_at=second,
                    tickers=[symbol],
                    tags=[],
                )
            ]

    monkeypatch.setattr(cli, "DefeatBetaNewsClient", FakeDefeatBetaClient)

    result = CliRunner().invoke(
        app,
        [
            "collect-defeatbeta-pre-earnings-week-articles",
            "--company-symbol",
            "WMT",
            "--company-name",
            "Walmart",
            "--transcript-root",
            str(transcript_root),
            "--raw-dir",
            str(raw_dir),
            "--manifest-out",
            str(manifest_out),
            "--days-before-call",
            "7",
            "--max-articles-per-event",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    manifests = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert len(manifests) == 2
    assert {row["fiscal_quarter"] for row in manifests} == {1, 2}
    assert all(row["company_symbol"] == "WMT" for row in manifests)


def test_collect_sec_press_releases_for_corpus_resolves_mapping_and_writes_manifest(
    tmp_path,
    monkeypatch,
):
    from kalorie.app import cli

    transcript_root = tmp_path / "transcripts"
    microsoft = transcript_root / "Microsoft"
    microsoft.mkdir(parents=True)
    (microsoft / "2025_Q1_msft_processed.txt").write_text("AI and cloud.", encoding="utf-8")
    company_map = tmp_path / "sec" / "company_to_cik.json"
    manifest_out = tmp_path / "manifests" / "sec.json"
    raw_dir = tmp_path / "raw"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeSecApiClient:
        def __init__(self, api_key, http_client, base_url: str):
            self.api_key = api_key
            self.http_client = http_client
            self.base_url = base_url

        def resolve_mapping(self, *, resolve_by: str, value: str):
            assert resolve_by == "name"
            assert value == "Microsoft"
            return [
                SecCompanyMapping(
                    name="MICROSOFT CORP",
                    ticker="MSFT",
                    cik="789019",
                    category="Domestic Common Stock",
                )
            ]

        def query_ex99_1_filings(self, *, query: str, size: int):
            assert 'cik:789019' in query
            return [
                SecApiFiling(
                    ticker="MSFT",
                    cik="789019",
                    filed_at=datetime(2025, 1, 30, tzinfo=UTC),
                    exhibit_url="https://www.sec.gov/msft-release.htm",
                    exhibits=[
                        {
                            "document_type": "EX-99.1",
                            "description": "PRESS RELEASE",
                            "document_url": "https://www.sec.gov/msft-release.htm",
                        },
                        {
                            "document_type": "EX-99.2",
                            "description": "EARNINGS PRESENTATION",
                            "document_url": "https://www.sec.gov/msft-slides.htm",
                        },
                    ],
                )
            ]

    def fake_collect_public_document(**kwargs):
        path = raw_dir / "MSFT-2025-Q1-release.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Microsoft cloud release.", encoding="utf-8")
        return PublicDocumentManifest(
            source_url=kwargs["url"],
            company_symbol=kwargs["company_symbol"],
            fiscal_year=kwargs["fiscal_year"],
            fiscal_quarter=kwargs["fiscal_quarter"],
            source_type=kwargs["source_type"],
            published_at=kwargs["published_at"],
            fetched_at=kwargs["fetched_at"],
            raw_path=str(path),
            content_hash="abc",
            extraction_method="html_text",
        )

    monkeypatch.setattr(cli, "_load_settings", lambda: Settings(sec_api_key="sec-secret"))
    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr(cli, "SecApiClient", FakeSecApiClient)
    monkeypatch.setattr(cli, "collect_public_document", fake_collect_public_document)

    result = CliRunner().invoke(
        app,
        [
            "collect-sec-press-releases-for-corpus",
            "--transcript-root",
            str(transcript_root),
            "--company-map",
            str(company_map),
            "--raw-dir",
            str(raw_dir),
            "--manifest-out",
            str(manifest_out),
            "--max-companies",
            "1",
            "--filings-per-company",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(company_map.read_text(encoding="utf-8")) == {"MSFT": "789019"}
    manifests = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert len(manifests) == 2
    assert manifests[0]["company_symbol"] == "MSFT"
    assert [manifest["source_type"] for manifest in manifests] == [
        "sec_ex_99_1_supplemental",
        "sec_ex_99_2_supplemental",
    ]


def test_collect_sec_press_releases_for_corpus_skips_far_year_filings(
    tmp_path,
    monkeypatch,
):
    from kalorie.app import cli

    transcript_root = tmp_path / "transcripts"
    accenture = transcript_root / "Accenture plc"
    accenture.mkdir(parents=True)
    (accenture / "2024_Q2_acn_processed.txt").write_text(
        "Bookings and consulting revenue.",
        encoding="utf-8",
    )
    company_map = tmp_path / "sec" / "company_to_cik.json"
    company_map.parent.mkdir(parents=True)
    company_map.write_text(json.dumps({"ACN": "1467373"}), encoding="utf-8")
    manifest_out = tmp_path / "manifests" / "sec.json"
    raw_dir = tmp_path / "raw"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeSecApiClient:
        def __init__(self, api_key, http_client, base_url: str):
            pass

        def query_ex99_1_filings(self, *, query: str, size: int):
            return [
                SecApiFiling(
                    ticker="ACN",
                    cik="1467373",
                    filed_at=datetime(2015, 5, 4, 8, 5, tzinfo=UTC),
                    exhibit_url="https://www.sec.gov/acn-2015-q2.htm",
                    exhibits=[
                        {
                            "document_type": "EX-99.1",
                            "description": "OLD PRESS RELEASE",
                            "document_url": "https://www.sec.gov/acn-2015-q2.htm",
                        }
                    ],
                ),
                SecApiFiling(
                    ticker="ACN",
                    cik="1467373",
                    filed_at=datetime(2024, 3, 21, 8, 0, tzinfo=UTC),
                    exhibit_url="https://www.sec.gov/acn-2024-q2.htm",
                    exhibits=[
                        {
                            "document_type": "EX-99.1",
                            "description": "Q2 2024 PRESS RELEASE",
                            "document_url": "https://www.sec.gov/acn-2024-q2.htm",
                        }
                    ],
                ),
            ]

    collected_urls = []

    def fake_collect_public_document(**kwargs):
        collected_urls.append(kwargs["url"])
        path = raw_dir / "ACN-2024-Q2-release.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Accenture Q2 release.", encoding="utf-8")
        return PublicDocumentManifest(
            source_url=kwargs["url"],
            company_symbol=kwargs["company_symbol"],
            fiscal_year=kwargs["fiscal_year"],
            fiscal_quarter=kwargs["fiscal_quarter"],
            source_type=kwargs["source_type"],
            published_at=kwargs["published_at"],
            fetched_at=kwargs["fetched_at"],
            raw_path=str(path),
            content_hash="abc",
            extraction_method="html_text",
        )

    monkeypatch.setattr(cli, "_load_settings", lambda: Settings(sec_api_key="sec-secret"))
    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr(cli, "SecApiClient", FakeSecApiClient)
    monkeypatch.setattr(cli, "collect_public_document", fake_collect_public_document)

    result = CliRunner().invoke(
        app,
        [
            "collect-sec-press-releases-for-corpus",
            "--transcript-root",
            str(transcript_root),
            "--company-map",
            str(company_map),
            "--raw-dir",
            str(raw_dir),
            "--manifest-out",
            str(manifest_out),
            "--max-companies",
            "1",
            "--filings-per-company",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert collected_urls == ["https://www.sec.gov/acn-2024-q2.htm"]
    manifests = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert manifests[0]["published_at"].startswith("2024-03-21")


def test_collect_sec_press_releases_for_corpus_does_not_clobber_existing_manifest(
    tmp_path,
    monkeypatch,
):
    from kalorie.app import cli

    transcript_root = tmp_path / "transcripts"
    microsoft = transcript_root / "Microsoft"
    microsoft.mkdir(parents=True)
    (microsoft / "2025_Q1_msft_processed.txt").write_text("AI and cloud.", encoding="utf-8")
    company_map = tmp_path / "sec" / "company_to_cik.json"
    company_map.parent.mkdir(parents=True)
    company_map.write_text(json.dumps({"MSFT": "789019"}), encoding="utf-8")
    manifest_out = tmp_path / "manifests" / "sec.json"
    manifest_out.parent.mkdir(parents=True)
    existing_manifest = [{"source_url": "https://www.sec.gov/existing.htm"}]
    manifest_out.write_text(json.dumps(existing_manifest), encoding="utf-8")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class RateLimitedSecApiClient:
        calls = 0

        def __init__(self, api_key, http_client, base_url: str):
            pass

        def query_ex99_1_filings(self, *, query: str, size: int):
            RateLimitedSecApiClient.calls += 1
            raise SecApiRateLimitError("SEC API rate limit exceeded")

    monkeypatch.setattr(cli, "_load_settings", lambda: Settings(sec_api_key="sec-secret"))
    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr(cli, "SecApiClient", RateLimitedSecApiClient)

    result = CliRunner().invoke(
        app,
        [
            "collect-sec-press-releases-for-corpus",
            "--transcript-root",
            str(transcript_root),
            "--company-map",
            str(company_map),
            "--raw-dir",
            str(tmp_path / "raw"),
            "--manifest-out",
            str(manifest_out),
            "--max-companies",
            "1",
            "--filings-per-company",
            "1",
        ],
    )

    assert result.exit_code != 0
    assert RateLimitedSecApiClient.calls == 1
    assert json.loads(manifest_out.read_text(encoding="utf-8")) == existing_manifest


def _write_examples(tmp_path) -> str:
    examples_path = tmp_path / "examples.json"
    examples_path.write_text(
        json.dumps(
            [
                _example("CAVA", 2025, 1, "traffic", 1).model_dump(mode="json"),
                _example("CAVA", 2025, 2, "robotaxi", 0).model_dump(mode="json"),
                _example("NVDA", 2025, 1, "ai", 1).model_dump(mode="json"),
                _example("NVDA", 2026, 1, "robotaxi", 0).model_dump(mode="json"),
            ]
        ),
        encoding="utf-8",
    )
    return str(examples_path)
