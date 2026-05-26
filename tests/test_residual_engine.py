from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from kalorie2.prediction_types import PredictionInputRow
from kalorie2.residual_engine import (
    LinearResidualModel,
    apply_residual,
    fit_linear_residual_model,
    walk_forward_predictions,
)


def _row(
    event_ticker: str,
    *,
    close_offset_days: int,
    outcome: str,
    market_mid: str = "0.50",
) -> PredictionInputRow:
    close_time = datetime(2026, 1, 1, 20, tzinfo=UTC) + timedelta(days=close_offset_days)
    mid = Decimal(market_mid)
    return PredictionInputRow.model_validate(
        {
            "market_ticker": f"{event_ticker}-TARI",
            "event_ticker": event_ticker,
            "series_ticker": "KXEARNINGSMENTIONDE",
            "market_category": "earnings",
            "event_phrase": "What will John Deere say during their next earnings call?",
            "market_name": "What will John Deere say during their next earnings call? - Tariff",
            "word_said": "Tariff",
            "normalized_word_said": "tariff",
            "final_outcome": outcome,
            "status": None,
            "close_time": close_time,
            "snapshot_target_time": close_time - timedelta(hours=8),
            "preclose_yes_bid": max(Decimal("0.00"), mid - Decimal("0.01")),
            "preclose_yes_ask": min(Decimal("1.00"), mid + Decimal("0.01")),
            "preclose_yes_mid": mid,
            "candle_end_ts": int(close_time.timestamp()) - 8 * 3600,
            "snapshot_staleness_seconds": 0,
            "settlement_ts": None,
            "source": "kalshi_search_series",
        }
    )


def test_apply_residual_anchors_on_market_logit():
    assert apply_residual(0.40, 0.0) == pytest.approx(0.40)
    assert apply_residual(0.40, 0.75) > 0.40
    assert apply_residual(0.40, -0.75) < 0.40
    assert 0.0 < apply_residual(0.999999, 20.0) < 1.0
    assert 0.0 < apply_residual(0.000001, -20.0) < 1.0
    assert 0.0 < apply_residual(0.50, -1000.0) < 1.0


def test_linear_residual_model_emits_reasons_for_weighted_features():
    model = LinearResidualModel(weights={"support": 0.8, "drag": -0.4}, intercept=0.0)

    prediction = model.predict(
        market_probability=0.50,
        feature_values={"support": 2.0, "drag": 1.0, "unused": 10.0},
    )

    assert prediction.probability > Decimal("0.50")
    assert "market_anchor" in prediction.reasons
    assert "linear_residual" in prediction.reasons
    assert "positive:support" in prediction.reasons
    assert "negative:drag" in prediction.reasons


def test_linear_residual_model_clips_extreme_residual_deltas():
    model = LinearResidualModel(weights={"support": 100.0}, residual_clip=0.5)

    prediction = model.predict(
        market_probability=0.50,
        feature_values={"support": 10.0},
    )

    assert prediction.residual_delta == 0.5
    assert prediction.probability == Decimal("0.622459")


def test_fit_linear_residual_model_uses_configurable_residual_clip():
    rows = [
        _row("EVENT1", close_offset_days=0, outcome="yes"),
        _row("EVENT2", close_offset_days=1, outcome="yes"),
    ]

    model = fit_linear_residual_model(
        rows,
        [{"support": 1.0}, {"support": 1.0}],
        epochs=5,
        learning_rate=0.05,
        residual_clip=0.25,
    )

    assert model.residual_clip == 0.25


def test_walk_forward_predictions_train_only_on_prior_events():
    rows = [
        _row("EVENT1", close_offset_days=0, outcome="no"),
        _row("EVENT2", close_offset_days=1, outcome="yes"),
        _row("EVENT3", close_offset_days=2, outcome="no"),
    ]
    feature_rows = [
        {"support": 1.0},
        {"support": -1.0},
        {"support": 1.0},
    ]

    predictions = walk_forward_predictions(
        rows,
        feature_rows,
        min_training_events=1,
        epochs=5,
        learning_rate=0.05,
    )

    assert [prediction.event_ticker for prediction in predictions] == ["EVENT2", "EVENT3"]
    assert predictions[0].training_event_tickers == ["EVENT1"]
    assert predictions[1].training_event_tickers == ["EVENT1", "EVENT2"]
    assert all(
        prediction.event_ticker not in prediction.training_event_tickers
        for prediction in predictions
    )


def test_training_handles_large_unscaled_feature_values_without_overflow():
    rows = [
        _row("EVENT1", close_offset_days=0, outcome="yes"),
        _row("EVENT2", close_offset_days=1, outcome="no"),
        _row("EVENT3", close_offset_days=2, outcome="yes"),
    ]

    model = fit_linear_residual_model(
        rows,
        [
            {"snapshot_staleness_seconds": 30_000.0},
            {"snapshot_staleness_seconds": 1.0},
            {"snapshot_staleness_seconds": 15_000.0},
        ],
        epochs=25,
        learning_rate=0.05,
    )
    prediction = model.predict(
        market_probability=0.50,
        feature_values={"snapshot_staleness_seconds": 20_000.0},
    )

    assert "snapshot_staleness_seconds" in model.feature_means
    assert "snapshot_staleness_seconds" in model.feature_scales
    assert Decimal("0") < prediction.probability < Decimal("1")


def test_no_side_residual_training_optimizes_no_probability_without_yes_dilution():
    rows = [
        _row("EVENT1", close_offset_days=0, outcome="no", market_mid="0.80"),
        _row("EVENT2", close_offset_days=1, outcome="no", market_mid="0.80"),
        _row("EVENT3", close_offset_days=2, outcome="yes", market_mid="0.80"),
    ]

    no_model = fit_linear_residual_model(
        rows,
        [{"no_signal": 1.0}, {"no_signal": 1.0}, {"no_signal": -1.0}],
        epochs=50,
        learning_rate=0.1,
        target_side="no",
        positive_label_weight=2.0,
    )
    yes_model = fit_linear_residual_model(
        rows,
        [{"no_signal": 1.0}, {"no_signal": 1.0}, {"no_signal": -1.0}],
        epochs=50,
        learning_rate=0.1,
        target_side="yes",
    )

    no_prediction = no_model.predict(
        market_probability=0.80,
        feature_values={"no_signal": 1.0},
    )
    yes_prediction = yes_model.predict(
        market_probability=0.80,
        feature_values={"no_signal": 1.0},
    )

    assert no_prediction.probability < Decimal("0.80")
    assert "target_side:no" in no_prediction.reasons
    assert no_model.target_side == "no"
    assert no_model.positive_label_weight == 2.0
    assert yes_model.target_side == "yes"
    assert no_model.weights["no_signal"] > 0
    assert yes_model.weights["no_signal"] < 0
    assert "target_side:yes" in yes_prediction.reasons
