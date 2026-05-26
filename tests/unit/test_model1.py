from datetime import UTC, datetime
from decimal import Decimal

from kalorie.domain.models import FeatureVector
from kalorie.ml.datasets import HistoricalTrainingExample
from kalorie.ml.model1 import (
    TemperatureCalibrationModel,
    _time_split_examples,
    optimize_model1_for_brier,
    predict_company_model1,
    predict_model1,
    train_company_model1,
    train_model1,
)


def _example(
    company: str,
    year: int,
    quarter: int,
    phrase: str,
    label: int,
    similarity: float,
    exact: float,
    market_probability: Decimal = Decimal("0.50"),
) -> HistoricalTrainingExample:
    return HistoricalTrainingExample(
        company_symbol=company,
        fiscal_year=year,
        fiscal_quarter=quarter,
        evidence_cutoff=datetime(year, quarter, 1, tzinfo=UTC),
        market_id=f"{company}-{year}-Q{quarter}-{phrase}",
        target_phrase=phrase,
        label=label,
        features={
            "exact_match_count": exact,
            "lexical_match_count": 0.0,
            "max_tfidf_similarity": similarity,
            "appears_in_headline_or_first_chunk": exact,
        },
        document_ids=[f"{company}-{year}-Q{quarter}-press"],
        market_probability=market_probability,
        market_venue="synthetic",
    )


def test_train_model1_builds_global_and_company_overrides():
    examples = [
        _example("AAPL", 2024, 1, "ai", 1, 0.9, 1.0),
        _example("AAPL", 2024, 2, "ai", 0, 0.1, 0.0),
        _example("AAPL", 2024, 3, "ai", 1, 0.8, 1.0),
        _example("AAPL", 2024, 4, "ai", 0, 0.2, 0.0),
        _example("MSFT", 2024, 1, "cloud", 1, 0.85, 1.0),
        _example("MSFT", 2024, 2, "cloud", 0, 0.15, 0.0),
        _example("MSFT", 2024, 3, "cloud", 1, 0.82, 1.0),
        _example("MSFT", 2024, 4, "cloud", 0, 0.18, 0.0),
    ]

    artifact = train_model1(examples, min_company_rows=4, blend_weight=0.35)

    assert artifact.model_version == "mention-base-company-v1"
    assert "exact_match_count" in artifact.feature_columns
    assert "AAPL" in artifact.company_overrides
    assert "MSFT" in artifact.company_overrides
    assert artifact.company_overrides["AAPL"].training_rows == 4


def test_train_model1_adds_category_overrides_for_phrase_mixture():
    examples = [
        _example("AAPL", 2024, 1, "tariff", 1, 0.8, 0.0),
        _example("AAPL", 2024, 2, "tariff", 0, 0.2, 0.0),
        _example("MSFT", 2024, 1, "inflation", 1, 0.7, 0.0),
        _example("MSFT", 2024, 2, "inflation", 0, 0.3, 0.0),
        _example("NVDA", 2024, 1, "margin", 1, 0.8, 1.0),
        _example("NVDA", 2024, 2, "margin", 0, 0.2, 0.0),
    ]

    artifact = train_model1(examples, min_company_rows=99, min_category_rows=4)
    prediction = predict_model1(
        artifact,
        company_symbol="COST",
        feature_vector=FeatureVector(
            target_phrase="tariff",
            features={
                "exact_match_count": 0.0,
                "lexical_match_count": 0.0,
                "max_tfidf_similarity": 0.7,
                "appears_in_headline_or_first_chunk": 0.0,
            },
        ),
    )

    assert "macro" in artifact.category_overrides
    assert "category_finetune" in prediction.reasons


def test_predict_model1_falls_back_without_company_override():
    examples = [
        _example("AAPL", 2024, 1, "ai", 1, 0.9, 1.0),
        _example("AAPL", 2024, 2, "ai", 0, 0.1, 0.0),
        _example("MSFT", 2024, 1, "cloud", 1, 0.85, 1.0),
        _example("MSFT", 2024, 2, "cloud", 0, 0.15, 0.0),
    ]
    artifact = train_model1(examples, min_company_rows=10, blend_weight=0.35)
    feature = FeatureVector(
        target_phrase="ai",
        features={
            "exact_match_count": 1.0,
            "lexical_match_count": 0.0,
            "max_tfidf_similarity": 0.8,
            "appears_in_headline_or_first_chunk": 1.0,
        },
    )

    prediction = predict_model1(artifact, company_symbol="NVDA", feature_vector=feature)

    assert 0.01 <= prediction.probability <= 0.99
    assert "base_logistic" in prediction.reasons
    assert "company_finetune" not in prediction.reasons


def test_model1_uses_market_prior_features_at_prediction_time():
    examples = [
        _example("AAPL", 2024, 1, "ai", 1, 0.5, 0.0, Decimal("0.90")),
        _example("AAPL", 2024, 2, "ai", 0, 0.5, 0.0, Decimal("0.10")),
        _example("MSFT", 2024, 1, "cloud", 1, 0.5, 0.0, Decimal("0.85")),
        _example("MSFT", 2024, 2, "cloud", 0, 0.5, 0.0, Decimal("0.15")),
    ]
    artifact = train_model1(examples, min_company_rows=99, include_market_features=True)
    feature = FeatureVector(
        target_phrase="ai",
        features={
            "exact_match_count": 0.0,
            "lexical_match_count": 0.0,
            "max_tfidf_similarity": 0.5,
            "appears_in_headline_or_first_chunk": 0.0,
        },
    )

    high_market = predict_model1(
        artifact,
        company_symbol="AAPL",
        feature_vector=feature,
        market_probability=0.90,
    )
    low_market = predict_model1(
        artifact,
        company_symbol="AAPL",
        feature_vector=feature,
        market_probability=0.10,
    )

    assert "market_prior" in high_market.reasons
    assert high_market.probability > low_market.probability


def test_model1_adds_hierarchical_prior_features_without_target_indicators():
    examples = [
        _example("AAPL", 2024, 1, "tariff", 1, 0.8, 0.0),
        _example("AAPL", 2024, 2, "tariff", 1, 0.8, 0.0),
        _example("MSFT", 2024, 1, "tariff", 0, 0.2, 0.0),
        _example("MSFT", 2024, 2, "tariff", 0, 0.2, 0.0),
    ]

    artifact = train_model1(
        examples,
        min_company_rows=99,
        include_target_indicator=False,
        include_hierarchical_priors=True,
    )
    prediction = predict_model1(
        artifact,
        company_symbol="AAPL",
        feature_vector=FeatureVector(
            target_phrase="tariff",
            features={
                "exact_match_count": 0.0,
                "lexical_match_count": 0.0,
                "max_tfidf_similarity": 0.5,
                "appears_in_headline_or_first_chunk": 0.0,
            },
        ),
    )

    assert artifact.hierarchical_priors is not None
    assert "prior_target_global_rate" in artifact.feature_columns
    assert not any(column.startswith("target_indicator__") for column in artifact.feature_columns)
    assert "hierarchical_priors" in prediction.reasons


def test_train_company_model1_builds_company_only_artifact():
    examples = [
        _example("AAPL", 2024, 1, "ai", 1, 0.9, 1.0),
        _example("AAPL", 2024, 2, "ai", 0, 0.1, 0.0),
        _example("AAPL", 2024, 3, "ai", 1, 0.8, 1.0),
        _example("AAPL", 2024, 4, "ai", 0, 0.2, 0.0),
        _example("MSFT", 2024, 1, "cloud", 1, 0.85, 1.0),
        _example("MSFT", 2024, 2, "cloud", 0, 0.15, 0.0),
    ]
    artifact = train_company_model1(
        examples,
        company_symbol="AAPL",
        min_company_rows=4,
        source_global_model_version="mention-base-company-v1",
    )

    assert artifact.company_symbol == "AAPL"
    assert artifact.training_rows == 4
    assert artifact.model_version == "mention-company-retrained-v1"
    assert artifact.source_global_model_version == "mention-base-company-v1"


def test_predict_company_model1_uses_company_retrained_coefficients():
    examples = [
        _example("AAPL", 2024, 1, "ai", 1, 0.9, 1.0),
        _example("AAPL", 2024, 2, "ai", 0, 0.1, 0.0),
        _example("AAPL", 2024, 3, "ai", 1, 0.8, 1.0),
        _example("AAPL", 2024, 4, "ai", 0, 0.2, 0.0),
    ]
    artifact = train_company_model1(examples, company_symbol="AAPL", min_company_rows=4)
    feature = FeatureVector(
        target_phrase="ai",
        features={
            "exact_match_count": 1.0,
            "lexical_match_count": 0.0,
            "max_tfidf_similarity": 0.8,
            "appears_in_headline_or_first_chunk": 1.0,
        },
    )

    prediction = predict_company_model1(artifact, feature_vector=feature)

    assert 0.01 <= prediction.probability <= 0.99
    assert "company_retrained_logistic" in prediction.reasons


def test_train_company_model1_supports_recency_ema_weighting():
    examples = [
        _example("NVDA", 2021, 1, "ai", 1, 0.55, 1.0),
        _example("NVDA", 2021, 2, "ai", 0, 0.45, 0.0),
        _example("NVDA", 2025, 3, "ai", 1, 0.95, 1.0),
        _example("NVDA", 2025, 4, "ai", 1, 0.97, 1.0),
    ]
    artifact = train_company_model1(
        examples,
        company_symbol="NVDA",
        min_company_rows=4,
        recency_ema_half_life_quarters=2.0,
    )
    assert artifact.company_symbol == "NVDA"
    assert artifact.recency_ema_half_life_quarters == 2.0


def test_optimize_model1_for_brier_returns_best_trial_and_artifact():
    examples = []
    for year, quarter in [(2023, 1), (2023, 2), (2023, 3), (2023, 4), (2024, 1), (2024, 2)]:
        examples.extend(
            [
                _example("AAPL", year, quarter, "ai", 1 if quarter % 2 else 0, 0.85, 1.0),
                _example("MSFT", year, quarter, "cloud", 1 if quarter in {1, 3} else 0, 0.8, 1.0),
                _example("NVDA", year, quarter, "gpu", 1 if quarter in {2, 4} else 0, 0.82, 1.0),
            ]
        )

    artifact, report = optimize_model1_for_brier(
        examples,
        test_fraction=0.25,
        regularization_values=[0.1, 1.0],
        min_company_rows_values=[4, 8],
        blend_weight_values=[0.25, 0.5],
        class_weight_balanced_values=[False, True],
        include_target_indicator_values=[False],
        enable_isotonic_calibration=False,
    )

    assert artifact.model_version == "mention-base-company-v2"
    assert report.holdout_sample_count > 0
    assert report.holdout_brier_score == report.best_trial.holdout_brier_score
    assert len(report.trials) == 16


def test_predict_model1_uses_unseen_target_fallback_with_target_indicators():
    examples = [
        _example("AAPL", 2024, 1, "ai", 1, 0.9, 1.0),
        _example("AAPL", 2024, 2, "ai", 0, 0.1, 0.0),
        _example("MSFT", 2024, 1, "cloud", 1, 0.85, 1.0),
        _example("MSFT", 2024, 2, "cloud", 0, 0.15, 0.0),
    ]
    artifact = train_model1(
        examples,
        min_company_rows=2,
        blend_weight=0.35,
        include_target_indicator=True,
        enable_isotonic_calibration=True,
    )
    feature = FeatureVector(
        target_phrase="sweet potato",
        features={
            "exact_match_count": 0.0,
            "lexical_match_count": 0.0,
            "max_tfidf_similarity": 0.1,
            "appears_in_headline_or_first_chunk": 0.0,
        },
    )
    prediction = predict_model1(artifact, company_symbol="CAVA", feature_vector=feature)

    assert 0.01 <= prediction.probability <= 0.99
    assert "unseen_target_fallback" in prediction.reasons


def test_temperature_calibration_softens_extreme_probabilities():
    calibrator = TemperatureCalibrationModel(temperature=2.0)

    assert 0.5 < calibrator.calibrate(0.99) < 0.99
    assert 0.01 < calibrator.calibrate(0.01) < 0.5


def test_train_model1_can_attach_temperature_calibration():
    examples = []
    for year in [2020, 2021, 2022]:
        for quarter in [1, 2, 3, 4]:
            examples.extend(
                [
                    _example("AAPL", year, quarter, "ai", 1, 0.9, 1.0),
                    _example("AAPL", year, quarter, "margin", 0, 0.2, 0.0),
                    _example(
                        "MSFT",
                        year,
                        quarter,
                        "cloud",
                        quarter % 2,
                        0.6,
                        float(quarter % 2),
                    ),
                ]
            )

    artifact = train_model1(
        examples,
        min_company_rows=999,
        enable_temperature_calibration=True,
        calibration_fraction=0.5,
    )
    prediction = predict_model1(
        artifact,
        company_symbol="AAPL",
        feature_vector=FeatureVector(
            target_phrase="ai",
            features={
                "exact_match_count": 1.0,
                "lexical_match_count": 0.0,
                "max_tfidf_similarity": 0.8,
                "appears_in_headline_or_first_chunk": 1.0,
            },
        ),
    )

    assert artifact.temperature_calibration is not None
    assert "temperature_calibration" in prediction.reasons


def test_time_split_examples_keeps_event_rows_together():
    examples = []
    # Each event has multiple phrase rows; split must keep them in one side.
    for year, quarter in [(2023, 1), (2023, 2), (2023, 3), (2023, 4)]:
        examples.extend(
            [
                _example("AAPL", year, quarter, "ai", 1 if quarter % 2 else 0, 0.8, 1.0),
                _example("AAPL", year, quarter, "margin", 0 if quarter % 2 else 1, 0.4, 0.0),
                _example(
                    "AAPL",
                    year,
                    quarter,
                    "guidance",
                    1 if quarter in {1, 4} else 0,
                    0.6,
                    0.0,
                ),
            ]
        )

    train, test = _time_split_examples(examples, test_fraction=0.25)
    train_events = {(row.company_symbol, row.fiscal_year, row.fiscal_quarter) for row in train}
    test_events = {(row.company_symbol, row.fiscal_year, row.fiscal_quarter) for row in test}

    assert train_events
    assert test_events
    assert train_events.isdisjoint(test_events)


def test_time_split_examples_orders_events_by_evidence_cutoff_not_fiscal_period():
    early_fiscal_late_call = _example("AAPL", 2024, 4, "ai", 1, 0.8, 1.0)
    early_fiscal_late_call.evidence_cutoff = datetime(2024, 6, 1, tzinfo=UTC)
    middle_call = _example("MSFT", 2024, 1, "cloud", 0, 0.2, 0.0)
    middle_call.evidence_cutoff = datetime(2024, 2, 1, tzinfo=UTC)
    latest_call = _example("NVDA", 2024, 2, "gpu", 1, 0.9, 1.0)
    latest_call.evidence_cutoff = datetime(2024, 8, 1, tzinfo=UTC)

    train, test = _time_split_examples(
        [early_fiscal_late_call, middle_call, latest_call],
        test_fraction=0.34,
    )

    assert {(row.company_symbol, row.fiscal_quarter) for row in train} == {
        ("MSFT", 1),
        ("AAPL", 4),
    }
    assert {(row.company_symbol, row.fiscal_quarter) for row in test} == {("NVDA", 2)}
