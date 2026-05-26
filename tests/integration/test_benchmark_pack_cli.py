import json
from datetime import UTC, datetime
from decimal import Decimal

from typer.testing import CliRunner

from kalorie.app.cli import app
from kalorie.benchmarking.packs import (
    BenchmarkEvent,
    BenchmarkEvidenceDocument,
    BenchmarkMarket,
    BenchmarkSnapshot,
)
from kalorie.ml.datasets import HistoricalTrainingExample
from kalorie.ml.model1 import MentionModelArtifact


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _inputs(tmp_path):
    event = BenchmarkEvent(
        event_ticker="KXEARNINGSMENTIONTGT-26MAY20",
        company_symbol="TGT",
        company_name="Target Corporation",
        fiscal_year=2026,
        fiscal_quarter=1,
        call_start_at=datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
        evidence_cutoff_at=datetime(2026, 5, 20, 11, 50, tzinfo=UTC),
    )
    market = BenchmarkMarket(
        event_ticker=event.event_ticker,
        market_id="KXEARNINGSMENTIONTGT-26MAY20-BEAU",
        target_phrase="beauty",
        title="What will Target Corporation say during their next earnings call?",
        result="yes",
    )
    evidence = BenchmarkEvidenceDocument(
        event_ticker=event.event_ticker,
        source_id="TGT-2026-Q1-release",
        company_symbol="TGT",
        document_type="sec_ex_99_1_supplemental",
        source_path="artifacts/benchmarks/tgt/release.txt",
        published_at=datetime(2026, 5, 20, 11, 45, tzinfo=UTC),
        content_hash="abc123",
        cutoff_eligible=True,
    )
    example = HistoricalTrainingExample(
        company_symbol="TGT",
        fiscal_year=2026,
        fiscal_quarter=1,
        evidence_cutoff=datetime(2026, 5, 20, 11, 50, tzinfo=UTC),
        market_id=market.market_id,
        target_phrase="beauty",
        label=1,
        features={"exact_match_count": 1.0},
        document_ids=[evidence.source_id],
        market_probability=Decimal("0.95"),
        market_venue="kalshi",
    )
    snapshot = BenchmarkSnapshot(
        event_ticker=event.event_ticker,
        market_id=market.market_id,
        preclose_yes_bid=Decimal("0.93"),
        preclose_yes_ask=Decimal("0.95"),
        snapshot_target_time=datetime(2026, 5, 20, 11, 50, tzinfo=UTC),
        candle_end_ts=1779277800,
    )
    event_path = tmp_path / "events.json"
    contracts_path = tmp_path / "contracts.json"
    evidence_path = tmp_path / "evidence.json"
    examples_path = tmp_path / "examples.json"
    snapshots_path = tmp_path / "snapshots.json"
    _write_json(event_path, [event.model_dump(mode="json")])
    _write_json(contracts_path, [market.model_dump(mode="json")])
    _write_json(evidence_path, [evidence.model_dump(mode="json")])
    _write_json(examples_path, [example.model_dump(mode="json")])
    _write_json(snapshots_path, [snapshot.model_dump(mode="json")])
    return event_path, contracts_path, evidence_path, examples_path, snapshots_path


def test_build_benchmark_pack_cli_writes_structured_pack(tmp_path):
    event_path, contracts_path, evidence_path, examples_path, snapshots_path = _inputs(tmp_path)
    out_dir = tmp_path / "pack"

    result = CliRunner().invoke(
        app,
        [
            "build-benchmark-pack",
            "--event-config",
            str(event_path),
            "--contracts",
            str(contracts_path),
            "--examples",
            str(examples_path),
            "--evidence-manifests",
            str(evidence_path),
            "--snapshots",
            str(snapshots_path),
            "--split",
            "validation",
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "events.json").exists()
    assert (out_dir / "markets.json").exists()
    assert (out_dir / "snapshots.json").exists()
    assert (out_dir / "evidence.json").exists()
    assert (out_dir / "examples.json").exists()
    assert (out_dir / "pack.json").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    pack = json.loads((out_dir / "pack.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "validation"
    assert pack["manifest"]["pack_id"] == "pack"
    assert pack["markets"][0]["result"] == "yes"


def test_build_benchmark_pack_cli_fails_when_required_snapshot_is_missing(tmp_path):
    event_path, contracts_path, evidence_path, examples_path, snapshots_path = _inputs(tmp_path)
    _write_json(snapshots_path, [])

    result = CliRunner().invoke(
        app,
        [
            "build-benchmark-pack",
            "--event-config",
            str(event_path),
            "--contracts",
            str(contracts_path),
            "--examples",
            str(examples_path),
            "--evidence-manifests",
            str(evidence_path),
            "--snapshots",
            str(snapshots_path),
            "--split",
            "validation",
            "--out",
            str(tmp_path / "pack"),
        ],
    )

    assert result.exit_code != 0
    assert "missing snapshots" in result.output


def test_run_benchmark_pack_cli_requires_model_family(tmp_path):
    model_path = tmp_path / "model.json"
    pack_path = tmp_path / "pack.json"
    model_path.write_text("{}", encoding="utf-8")
    pack_path.write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "run-benchmark-pack",
            "--pack",
            str(pack_path),
            "--model",
            str(model_path),
            "--out",
            str(tmp_path / "report.json"),
        ],
    )

    assert result.exit_code != 0
    assert "model-family" in result.output


def test_run_benchmark_pack_cli_writes_model_family_report(tmp_path):
    event_path, contracts_path, evidence_path, examples_path, snapshots_path = _inputs(tmp_path)
    pack_dir = tmp_path / "pack"
    build_result = CliRunner().invoke(
        app,
        [
            "build-benchmark-pack",
            "--event-config",
            str(event_path),
            "--contracts",
            str(contracts_path),
            "--examples",
            str(examples_path),
            "--evidence-manifests",
            str(evidence_path),
            "--snapshots",
            str(snapshots_path),
            "--split",
            "validation",
            "--out",
            str(pack_dir),
        ],
    )
    assert build_result.exit_code == 0, build_result.output
    model = MentionModelArtifact(
        model_version="unit-test-model",
        feature_columns=["exact_match_count"],
        base_intercept=0.0,
        base_coefficients={"exact_match_count": 0.0},
        blend_weight=0.0,
        min_company_rows=9999,
    )
    model_path = tmp_path / "model.json"
    _write_json(model_path, model.model_dump(mode="json"))
    out_path = tmp_path / "report.json"

    result = CliRunner().invoke(
        app,
        [
            "run-benchmark-pack",
            "--pack",
            str(pack_dir / "pack.json"),
            "--model",
            str(model_path),
            "--model-family",
            "global_base",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["model_family"] == "global_base"
    assert report["rows"][0]["prediction_source"] == "global_base:unit-test-model"
    assert (tmp_path / "report.table.md").exists()
    assert (tmp_path / "report.log").exists()
    assert (tmp_path / "report.diagnostics.md").exists()
