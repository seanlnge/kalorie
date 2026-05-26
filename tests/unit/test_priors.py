from datetime import UTC, datetime
from decimal import Decimal

from kalorie.ml.datasets import HistoricalTrainingExample
from kalorie.ml.priors import (
    add_historical_prior_features,
    phrase_category,
    phrase_category_features,
)


def _example(
    company: str,
    year: int,
    quarter: int,
    phrase: str,
    label: int,
) -> HistoricalTrainingExample:
    return HistoricalTrainingExample(
        company_symbol=company,
        fiscal_year=year,
        fiscal_quarter=quarter,
        evidence_cutoff=datetime(year, quarter, 1, tzinfo=UTC),
        market_id=f"{company}-{year}-Q{quarter}-{phrase}",
        target_phrase=phrase,
        label=label,
        features={},
        document_ids=[],
        market_probability=Decimal("0.50"),
        market_venue="synthetic",
    )


def test_phrase_category_features_identify_known_categories():
    assert phrase_category("tariff") == "macro"
    assert phrase_category("openai") == "competitor"
    assert phrase_category("nano banana") == "codename_or_product"
    assert phrase_category("live sports") == "multiword"
    assert phrase_category("margin") == "generic"

    features = phrase_category_features("tariff")

    assert features["phrase_category_macro"] == 1.0
    assert features["phrase_category_generic"] == 0.0


def test_historical_prior_features_use_only_prior_events():
    examples = [
        _example("WMT", 2024, 1, "tariff", 1),
        _example("WMT", 2024, 2, "tariff", 0),
        _example("WMT", 2024, 3, "tariff", 1),
    ]

    enriched = add_historical_prior_features(examples)

    assert enriched[0].features["prior_target_global_log_count"] == 0.0
    assert enriched[1].features["prior_target_global_rate"] == 1.0
    assert enriched[1].features["prior_target_global_log_count"] > 0.0
    assert enriched[2].features["prior_target_global_rate"] == 0.5
    assert enriched[2].features["prior_company_target_rate"] == 0.5
