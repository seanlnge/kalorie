from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from kalorie.benchmarking.packs import (
    BenchmarkEvent,
    BenchmarkEvidenceDocument,
    BenchmarkMarket,
    BenchmarkPack,
    BenchmarkPackManifest,
    BenchmarkRunMetadata,
    BenchmarkSnapshot,
    validate_benchmark_pack,
)
from kalorie.ml.datasets import HistoricalTrainingExample


def _example(*, market_id: str = "KXEARNINGSMENTIONTGT-26MAY20-BEAU") -> HistoricalTrainingExample:
    return HistoricalTrainingExample(
        company_symbol="TGT",
        fiscal_year=2026,
        fiscal_quarter=1,
        evidence_cutoff=datetime(2026, 5, 20, 11, 50, tzinfo=UTC),
        market_id=market_id,
        target_phrase="beauty",
        label=1,
        features={"exact_match_count": 1.0},
        document_ids=["TGT-2026-Q1-release"],
        market_probability=Decimal("0.95"),
        market_venue="kalshi",
    )


def _pack(
    *,
    snapshot_market_id: str = "KXEARNINGSMENTIONTGT-26MAY20-BEAU",
    example_market_id: str = "KXEARNINGSMENTIONTGT-26MAY20-BEAU",
    candle_end_ts: int = 1779277800,
) -> BenchmarkPack:
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
    snapshot = BenchmarkSnapshot(
        event_ticker=event.event_ticker,
        market_id=snapshot_market_id,
        preclose_yes_bid=Decimal("0.93"),
        preclose_yes_ask=Decimal("0.95"),
        snapshot_target_time=datetime(2026, 5, 20, 11, 50, tzinfo=UTC),
        candle_end_ts=candle_end_ts,
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
    manifest = BenchmarkPackManifest(
        pack_id="tgt-validation",
        split="validation",
        created_at=datetime(2026, 5, 21, 13, 0, tzinfo=UTC),
        description="Target validation event",
    )
    return BenchmarkPack(
        manifest=manifest,
        events=[event],
        markets=[market],
        snapshots=[snapshot],
        evidence=[evidence],
        examples=[_example(market_id=example_market_id)],
    )


def test_benchmark_event_requires_timezone_aware_times():
    with pytest.raises(ValidationError):
        BenchmarkEvent(
            event_ticker="KXEARNINGSMENTIONTGT-26MAY20",
            company_symbol="TGT",
            company_name="Target Corporation",
            fiscal_year=2026,
            fiscal_quarter=1,
            call_start_at=datetime(2026, 5, 20, 12, 0),
            evidence_cutoff_at=datetime(2026, 5, 20, 11, 50, tzinfo=UTC),
        )


def test_benchmark_snapshot_rejects_candle_after_snapshot_target():
    with pytest.raises(ValidationError, match="candle_end_ts"):
        BenchmarkSnapshot(
            event_ticker="KXEARNINGSMENTIONTGT-26MAY20",
            market_id="KXEARNINGSMENTIONTGT-26MAY20-BEAU",
            preclose_yes_bid=Decimal("0.93"),
            preclose_yes_ask=Decimal("0.95"),
            snapshot_target_time=datetime(2026, 5, 20, 11, 50, tzinfo=UTC),
            candle_end_ts=int(
                (datetime(2026, 5, 20, 11, 50, tzinfo=UTC) + timedelta(minutes=1)).timestamp()
            ),
        )


def test_validate_benchmark_pack_accepts_matching_market_example_and_snapshot_ids():
    validate_benchmark_pack(_pack())


def test_validate_benchmark_pack_rejects_missing_snapshot_for_example_market():
    pack = _pack(snapshot_market_id="KXEARNINGSMENTIONTGT-26MAY20-OTHER")

    with pytest.raises(ValueError, match="missing snapshots"):
        validate_benchmark_pack(pack)


def test_validate_benchmark_pack_rejects_example_without_market_metadata():
    pack = _pack(example_market_id="KXEARNINGSMENTIONTGT-26MAY20-OTHER")

    with pytest.raises(ValueError, match="missing market metadata"):
        validate_benchmark_pack(pack)


def test_benchmark_manifest_rejects_unknown_split():
    with pytest.raises(ValidationError):
        BenchmarkPackManifest(
            pack_id="bad",
            split="test",
            created_at=datetime(2026, 5, 21, 13, 0, tzinfo=UTC),
            description="bad split",
        )


def test_benchmark_run_metadata_requires_known_model_family():
    metadata = BenchmarkRunMetadata(
        model_family="global_base",
        model_path="artifacts/model1/models/model.json",
        pack_path="artifacts/benchmarks/tgt-validation",
    )

    assert metadata.model_family == "global_base"

    with pytest.raises(ValidationError):
        BenchmarkRunMetadata(
            model_family="secret_blend",
            model_path="artifacts/model1/models/model.json",
            pack_path="artifacts/benchmarks/tgt-validation",
        )
