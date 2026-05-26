from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from kalorie2.prediction_types import (
    ArtifactRetentionPolicy,
    MarketSnapshotFeatures,
    PredictionInputRow,
    PredictionRecord,
    PredictionRunConfig,
)


def _historical_row() -> dict:
    return {
        "market_ticker": "KXEARNINGSMENTIONDE-26MAY21-TARI",
        "event_ticker": "KXEARNINGSMENTIONDE-26MAY21",
        "series_ticker": "KXEARNINGSMENTIONDE",
        "market_category": "earnings",
        "event_phrase": "What will John Deere say during their next earnings call?",
        "market_name": "What will John Deere say during their next earnings call? - Tariff",
        "word_said": "Tariff",
        "normalized_word_said": "tariff",
        "final_outcome": "yes",
        "status": None,
        "close_time": datetime(2026, 5, 21, 15, 41, 50, tzinfo=UTC),
        "snapshot_target_time": datetime(2026, 5, 21, 7, 41, 50, tzinfo=UTC),
        "preclose_yes_bid": Decimal("0.95"),
        "preclose_yes_ask": Decimal("0.97"),
        "preclose_yes_mid": Decimal("0.96"),
        "candle_end_ts": 1779349020,
        "snapshot_staleness_seconds": 290,
        "settlement_ts": None,
        "source": "kalshi_search_series",
    }


def test_prediction_input_row_omits_label_from_inference_payload():
    row = PredictionInputRow.model_validate(_historical_row())

    payload = row.to_inference_payload()

    assert row.outcome_label == 1
    assert payload["market_ticker"] == "KXEARNINGSMENTIONDE-26MAY21-TARI"
    assert payload["normalized_word_said"] == "tariff"
    assert "final_outcome" not in payload
    assert "outcome_label" not in payload
    assert "settlement_ts" not in payload


def test_prediction_run_config_requires_explicit_core_fields():
    with pytest.raises(ValidationError):
        PredictionRunConfig.model_validate(
            {
                "run_id": "missing-decision-time-and-policy",
            }
        )

    config = PredictionRunConfig(
        run_id="unit-test-run",
        decision_time_column="snapshot_target_time",
        artifact_retention_policy=ArtifactRetentionPolicy(
            canonical_source_files={
                "mention-markets-historical-20260523.csv",
                "mention-markets-historical-20260523.json",
            }
        ),
    )

    assert config.run_id == "unit-test-run"
    assert config.decision_time_column == "snapshot_target_time"


def test_artifact_policy_rejects_non_source_writes_to_full_directory():
    config = PredictionRunConfig(
        run_id="unit-test-run",
        decision_time_column="snapshot_target_time",
        artifact_retention_policy=ArtifactRetentionPolicy(
            canonical_source_files={
                "mention-markets-historical-20260523.csv",
                "mention-markets-historical-20260523.json",
            }
        ),
    )

    with pytest.raises(ValueError, match="artifacts/full"):
        config.validate_output_path(
            Path("artifacts/full/predictions.csv"),
            artifact_kind="temporary_prediction",
        )
    with pytest.raises(ValueError, match="artifacts/full"):
        config.validate_output_path(
            Path("artifacts/prediction-engine/../full/predictions.csv"),
            artifact_kind="temporary_prediction",
        )

    canonical_path = config.validate_output_path(
        Path("artifacts/full/mention-markets-historical-20260523.csv"),
        artifact_kind="canonical_source",
    )
    run_path = config.validate_output_path(
        Path("artifacts/prediction-engine/unit-test-run/predictions.csv"),
        artifact_kind="temporary_prediction",
    )

    assert canonical_path == Path("artifacts/full/mention-markets-historical-20260523.csv")
    assert run_path == Path("artifacts/prediction-engine/unit-test-run/predictions.csv")


def test_market_snapshot_features_and_prediction_record_validate_contracts():
    features = MarketSnapshotFeatures.from_row(PredictionInputRow.model_validate(_historical_row()))

    assert features.yes_bid == Decimal("0.95")
    assert features.yes_ask == Decimal("0.97")
    assert features.yes_mid == Decimal("0.96")
    assert features.spread == Decimal("0.02")
    assert features.snapshot_staleness_seconds == 290

    record = PredictionRecord(
        market_ticker="KXEARNINGSMENTIONDE-26MAY21-TARI",
        event_ticker="KXEARNINGSMENTIONDE-26MAY21",
        probability=Decimal("0.93"),
        market_probability=Decimal("0.96"),
        feature_values={"market_spread": 0.02, "phrase_is_macro": 1.0},
        reasons=["market_anchor", "task1_contract_test"],
    )

    assert record.feature_values["market_spread"] == 0.02

    with pytest.raises(ValidationError):
        PredictionRecord(
            market_ticker="bad",
            event_ticker="bad-event",
            probability=Decimal("1.20"),
            market_probability=Decimal("0.50"),
            feature_values={"final_outcome": 1.0},
            reasons=["label_leak"],
        )
