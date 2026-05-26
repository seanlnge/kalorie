from datetime import UTC, datetime
from decimal import Decimal

import pytest

from kalorie.benchmarking.packs import (
    BenchmarkEvent,
    BenchmarkMarket,
    BenchmarkPack,
    BenchmarkPackManifest,
    BenchmarkRunMetadata,
    BenchmarkSnapshot,
)
from kalorie.benchmarking.runner import (
    run_market_residual_pack_benchmark,
    run_model1_pack_benchmark,
)
from kalorie.ml.datasets import HistoricalTrainingExample
from kalorie.ml.market_residual import train_market_residual
from kalorie.ml.model1 import MentionModelArtifact


def _pack() -> BenchmarkPack:
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
        market_id=market.market_id,
        preclose_yes_bid=Decimal("0.93"),
        preclose_yes_ask=Decimal("0.95"),
        snapshot_target_time=datetime(2026, 5, 20, 11, 50, tzinfo=UTC),
        candle_end_ts=1779277800,
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
        document_ids=[],
        market_probability=Decimal("0.95"),
        market_venue="kalshi",
    )
    return BenchmarkPack(
        manifest=BenchmarkPackManifest(
            pack_id="tgt-validation",
            split="validation",
            created_at=datetime(2026, 5, 21, 13, 0, tzinfo=UTC),
            description="Target validation",
        ),
        events=[event],
        markets=[market],
        snapshots=[snapshot],
        examples=[example],
    )


def _model() -> MentionModelArtifact:
    return MentionModelArtifact(
        model_version="unit-test-model",
        feature_columns=["exact_match_count"],
        base_intercept=0.0,
        base_coefficients={"exact_match_count": 0.0},
        blend_weight=0.0,
        min_company_rows=9999,
    )


def test_run_model1_pack_benchmark_records_model_family_and_prediction_source():
    metadata = BenchmarkRunMetadata(
        model_family="global_base",
        model_path="artifacts/model1/models/model.json",
        pack_path="artifacts/benchmarks/tgt-validation",
        excluded_events=["KXEARNINGSMENTIONAAPL-26APR30"],
        calibration="none",
    )

    report = run_model1_pack_benchmark(_pack(), _model(), metadata)

    assert report["model_family"] == "global_base"
    assert report["model_path"] == "artifacts/model1/models/model.json"
    assert report["pack_split"] == "validation"
    assert report["excluded_events"] == ["KXEARNINGSMENTIONAAPL-26APR30"]
    assert report["sample_count"] == 1
    assert report["rows"][0]["prediction_source"] == "global_base:unit-test-model"
    assert report["rows"][0]["model_probability"] == 0.5
    assert report["skip_summary"]["skipped_rows"] == 0


def test_run_model1_pack_benchmark_rejects_company_niched_without_model_map():
    metadata = BenchmarkRunMetadata(
        model_family="company_niched",
        model_path="artifacts/model1/models/model.json",
        pack_path="artifacts/benchmarks/tgt-validation",
    )

    with pytest.raises(ValueError, match="company_model_map"):
        run_model1_pack_benchmark(_pack(), _model(), metadata)


def test_run_market_residual_pack_benchmark_uses_pack_market_anchor():
    pack = _pack()
    training_examples = [
        pack.examples[0],
        pack.examples[0].model_copy(
            update={
                "market_id": "negative-high-market",
                "target_phrase": "tariff",
                "label": 0,
                "market_probability": Decimal("0.90"),
                "features": {"exact_match_count": 0.0, "semantic_signal_max_tfidf": 0.0},
            }
        ),
        pack.examples[0].model_copy(
            update={
                "market_id": "positive-low-market",
                "target_phrase": "beauty",
                "label": 1,
                "market_probability": Decimal("0.20"),
                "features": {"exact_match_count": 1.0, "semantic_signal_max_tfidf": 1.0},
            }
        ),
        pack.examples[0].model_copy(
            update={
                "market_id": "negative-low-evidence",
                "target_phrase": "beauty",
                "label": 0,
                "market_probability": Decimal("0.80"),
                "features": {"exact_match_count": 0.0, "semantic_signal_max_tfidf": 0.0},
            }
        ),
    ]
    artifact = train_market_residual(training_examples)
    metadata = BenchmarkRunMetadata(
        model_family="ensemble_research",
        model_path="artifacts/model1/models/market-residual.pkl",
        pack_path="artifacts/benchmarks/tgt-validation",
    )

    report = run_market_residual_pack_benchmark(pack, artifact, metadata)

    assert report["model_family"] == "ensemble_research"
    assert report["rows"][0]["prediction_source"] == "ensemble_research:market-residual-v1"
    assert "market_residual" in report["rows"][0]["prediction_reasons"]
