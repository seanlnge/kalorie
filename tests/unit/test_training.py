from datetime import UTC, datetime
from decimal import Decimal

import pytest

from kalorie.ml.datasets import HistoricalTrainingExample
from kalorie.ml.training import _time_split, train_and_evaluate


def _example(
    company: str,
    year: int,
    quarter: int,
    target: str,
    label: int,
    similarity: float,
) -> HistoricalTrainingExample:
    return HistoricalTrainingExample(
        company_symbol=company,
        fiscal_year=year,
        fiscal_quarter=quarter,
        evidence_cutoff=datetime(year, quarter, 1, tzinfo=UTC),
        market_id=f"{company}-{year}-Q{quarter}-{target}",
        target_phrase=target,
        label=label,
        features={
            "exact_match_count": float(label),
            "lexical_match_count": 0.0,
            "max_tfidf_similarity": similarity,
            "appears_in_headline_or_first_chunk": float(label),
        },
        document_ids=[f"{company}-{year}-Q{quarter}-press"],
        market_probability=Decimal("0.50"),
    )


def test_train_and_evaluate_reports_global_and_company_adapted_metrics():
    examples = [
        _example("CAVA", 2025, 1, "traffic", 1, 0.9),
        _example("CAVA", 2025, 2, "robotaxi", 0, 0.0),
        _example("NVDA", 2025, 1, "ai", 1, 0.8),
        _example("NVDA", 2025, 2, "tariffs", 0, 0.1),
        _example("CAVA", 2026, 1, "traffic", 1, 0.7),
        _example("NVDA", 2026, 1, "robotaxi", 0, 0.0),
    ]

    report = train_and_evaluate(examples, test_fraction=0.34, company_prior_strength=2.0)

    assert report.sample_count == 2
    assert report.split_strategy == "time"
    assert report.global_model_version == "historical-logistic-v0"
    assert report.company_adapted_model_version == "company-shrinkage-v0"
    assert 0 <= report.global_brier_score <= 1
    assert not hasattr(report, "global_mean_squared_error")
    assert 0 <= report.company_adapted_brier_score <= 1
    assert not hasattr(report, "company_adapted_mean_squared_error")
    assert report.global_log_loss >= 0
    assert isinstance(report.company_adaptation_improved_brier, bool)


def test_train_and_evaluate_rejects_obvious_company_market_mismatch():
    examples = [_example("ACN", 2025, 1, "traffic", 1, 0.9)]
    examples[0] = examples[0].model_copy(
        update={"market_id": "KXEARNINGSMENTIONCAVA-26MAY19-TRAF"}
    )

    with pytest.raises(ValueError, match="market company CAVA does not match example company ACN"):
        train_and_evaluate(examples)


def test_time_split_keeps_phrase_rows_from_same_event_together():
    examples = [
        _example("CAVA", 2025, 1, "traffic", 1, 0.9),
        _example("CAVA", 2025, 1, "automation", 0, 0.1),
        _example("CAVA", 2025, 2, "traffic", 1, 0.8),
        _example("CAVA", 2025, 2, "automation", 0, 0.2),
        _example("NVDA", 2026, 1, "ai", 1, 0.7),
        _example("NVDA", 2026, 1, "cloud", 1, 0.6),
    ]

    train_examples, test_examples = _time_split(examples, test_fraction=0.5)
    train_events = {
        (example.company_symbol, example.fiscal_year, example.fiscal_quarter)
        for example in train_examples
    }
    test_events = {
        (example.company_symbol, example.fiscal_year, example.fiscal_quarter)
        for example in test_examples
    }

    assert train_events.isdisjoint(test_events)
