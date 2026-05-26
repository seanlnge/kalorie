from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from kalorie.domain.models import (
    Company,
    DocumentChunk,
    EarningsEvent,
    FeatureVector,
    MarketSnapshot,
    MatchSpan,
    MentionLabel,
    PaperTradeComparison,
    Prediction,
    SourceDocument,
    TargetPhrase,
)


def test_core_models_round_trip_through_json():
    observed_at = datetime(2026, 5, 19, 14, 30, tzinfo=UTC)
    published_at = datetime(2026, 5, 19, 12, 0, tzinfo=UTC)

    models = [
        Company(symbol="CAVA", name="CAVA Group, Inc."),
        EarningsEvent(
            company_symbol="CAVA",
            fiscal_year=2026,
            fiscal_quarter=1,
            event_date=date(2026, 5, 19),
        ),
        SourceDocument(
            source_id="doc-cava-q1-2026",
            company_symbol="CAVA",
            document_type="earnings_press_release",
            source_path="Earnings-Release-2026-Q1.pdf",
            published_at=published_at,
            content_hash="abc123",
        ),
        DocumentChunk(
            document_id="doc-cava-q1-2026",
            chunk_index=0,
            text="CAVA same restaurant sales increased.",
            section="headline",
            token_start=0,
            token_end=5,
        ),
        TargetPhrase(
            phrase="Same Restaurant Sales",
            normalized_phrase="same restaurant sales",
            aliases=["same-restaurant sales"],
        ),
        MarketSnapshot(
            venue="kalshi",
            market_id="CAVA-TRAFFIC",
            title="Will CAVA mention traffic during earnings?",
            yes_bid=Decimal("0.38"),
            yes_ask=Decimal("0.45"),
            observed_at=observed_at,
        ),
        MentionLabel(
            target_phrase="traffic",
            exact_mentioned=True,
            lexical_mentioned=True,
            match_spans=[MatchSpan(start=5, end=12, text="traffic", match_type="exact")],
        ),
        FeatureVector(
            target_phrase="traffic",
            features={"exact_match_count": 1.0, "max_tfidf_similarity": 0.42},
        ),
        Prediction(
            target_phrase="traffic",
            model_version="rule-based-v0",
            probability=0.72,
            reasons=["exact_match"],
        ),
        PaperTradeComparison(
            target_phrase="traffic",
            model_probability=Decimal("0.72"),
            market_probability=Decimal("0.45"),
            edge=Decimal("0.27"),
            side="yes",
            reasons=["yes_edge"],
        ),
    ]

    for model in models:
        restored = type(model).model_validate_json(model.model_dump_json())
        assert restored == model


def test_market_prices_and_probabilities_are_bounded():
    observed_at = datetime(2026, 5, 19, 14, 30, tzinfo=UTC)

    with pytest.raises(ValidationError):
        MarketSnapshot(
            venue="kalshi",
            market_id="bad",
            title="bad",
            yes_bid=Decimal("1.20"),
            yes_ask=Decimal("0.45"),
            observed_at=observed_at,
        )

    with pytest.raises(ValidationError):
        Prediction(
            target_phrase="traffic",
            model_version="rule-based-v0",
            probability=1.5,
            reasons=[],
        )


def test_datetime_fields_require_timezone():
    with pytest.raises(ValidationError, match="timezone-aware"):
        SourceDocument(
            source_id="doc",
            company_symbol="CAVA",
            document_type="earnings_press_release",
            source_path="release.pdf",
            published_at=datetime(2026, 5, 19, 12, 0),
            content_hash="abc123",
        )
