from datetime import UTC, datetime
from decimal import Decimal

from kalorie.domain.models import MarketSnapshot, Prediction
from kalorie.market.paper import (
    compare_prediction_to_market,
    implied_no_probability,
    implied_yes_probability,
)


def _snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        venue="kalshi",
        market_id="CAVA-TRAFFIC",
        title="Will CAVA mention traffic during earnings?",
        yes_bid=Decimal("0.38"),
        yes_ask=Decimal("0.45"),
        observed_at=datetime(2026, 5, 19, tzinfo=UTC),
    )


def test_implied_yes_no_probabilities_and_spread():
    snapshot = _snapshot()

    assert implied_yes_probability(snapshot) == Decimal("0.45")
    assert implied_no_probability(snapshot) == Decimal("0.62")


def test_compare_prediction_selects_yes_side_when_edge_exceeds_threshold():
    comparison = compare_prediction_to_market(
        Prediction(
            target_phrase="traffic",
            model_version="rule-based-v0",
            probability=0.62,
            reasons=["exact_match"],
        ),
        _snapshot(),
        min_edge=Decimal("0.05"),
    )

    assert comparison.side == "yes"
    assert comparison.market_probability == Decimal("0.45")
    assert comparison.edge == Decimal("0.17")
    assert comparison.spread == Decimal("0.07")
    assert "yes_edge" in comparison.reasons


def test_compare_prediction_selects_no_side_when_no_edge_exceeds_threshold():
    comparison = compare_prediction_to_market(
        Prediction(
            target_phrase="traffic",
            model_version="rule-based-v0",
            probability=0.25,
            reasons=["weak_evidence"],
        ),
        _snapshot(),
        min_edge=Decimal("0.05"),
    )

    assert comparison.side == "no"
    assert comparison.market_probability == Decimal("0.62")
    assert comparison.edge == Decimal("0.13")
    assert "no_edge" in comparison.reasons


def test_compare_prediction_skips_when_edge_below_threshold():
    comparison = compare_prediction_to_market(
        Prediction(
            target_phrase="traffic",
            model_version="rule-based-v0",
            probability=0.48,
            reasons=["base_rate"],
        ),
        _snapshot(),
        min_edge=Decimal("0.05"),
    )

    assert comparison.side == "skip"
    assert comparison.edge == Decimal("0.03")
    assert "edge_below_threshold" in comparison.reasons
