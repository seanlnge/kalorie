from datetime import UTC, datetime
from decimal import Decimal

from kalorie.domain.models import FeatureVector
from kalorie.ml.datasets import HistoricalTrainingExample
from kalorie.ml.market_residual import (
    market_microstructure_features,
    predict_market_residual,
    train_market_residual,
)


def _example(
    phrase: str,
    label: int,
    market_probability: Decimal,
    evidence: float,
) -> HistoricalTrainingExample:
    return HistoricalTrainingExample(
        company_symbol="TGT",
        fiscal_year=2026,
        fiscal_quarter=1,
        evidence_cutoff=datetime(2026, 5, 20, 11, 50, tzinfo=UTC),
        market_id=f"TGT-{phrase}-{market_probability}",
        target_phrase=phrase,
        label=label,
        features={
            "exact_match_count": evidence,
            "semantic_signal_max_tfidf": evidence,
            "phrase_category_generic": 1.0,
        },
        document_ids=[],
        market_probability=market_probability,
        market_venue="kalshi",
    )


def test_market_microstructure_features_include_spread_and_illiquidity():
    features = market_microstructure_features(
        yes_bid=Decimal("0.20"),
        yes_ask=Decimal("0.80"),
    )

    assert features["market_mid_probability"] == 0.5
    assert features["market_spread"] == 0.6
    assert features["market_wide_spread_binary"] == 1.0
    assert features["market_illiquidity_score"] == 0.6


def test_market_residual_prediction_combines_market_anchor_and_evidence_residual():
    examples = [
        _example("beauty", 1, Decimal("0.20"), 1.0),
        _example("beauty", 1, Decimal("0.30"), 1.0),
        _example("beauty", 0, Decimal("0.80"), 0.0),
        _example("beauty", 0, Decimal("0.70"), 0.0),
        _example("tariff", 1, Decimal("0.25"), 1.0),
        _example("tariff", 0, Decimal("0.75"), 0.0),
    ]

    artifact = train_market_residual(examples)
    high_evidence = predict_market_residual(
        artifact,
        company_symbol="TGT",
        feature_vector=FeatureVector(
            target_phrase="beauty",
            features={"exact_match_count": 1.0, "semantic_signal_max_tfidf": 1.0},
        ),
        market_probability=0.20,
    )
    low_evidence = predict_market_residual(
        artifact,
        company_symbol="TGT",
        feature_vector=FeatureVector(
            target_phrase="beauty",
            features={"exact_match_count": 0.0, "semantic_signal_max_tfidf": 0.0},
        ),
        market_probability=0.80,
    )

    assert "market_residual" in high_evidence.reasons
    assert "grouped_calibration" in high_evidence.reasons
    assert high_evidence.probability > 0.20
    assert low_evidence.probability < 0.80
