import pandas as pd

from kalorie.domain.models import FeatureVector
from kalorie.ml.modeling import LogisticBaseline, RuleBasedBaseline


def test_rule_based_baseline_is_bounded_and_explainable():
    model = RuleBasedBaseline(base_rate=0.25)
    no_evidence = FeatureVector(
        target_phrase="robotaxi",
        features={
            "exact_match_count": 0.0,
            "lexical_match_count": 0.0,
            "max_tfidf_similarity": 0.0,
            "appears_in_headline_or_first_chunk": 0.0,
        },
    )
    exact_evidence = FeatureVector(
        target_phrase="traffic",
        features={
            "exact_match_count": 1.0,
            "lexical_match_count": 0.0,
            "max_tfidf_similarity": 0.8,
            "appears_in_headline_or_first_chunk": 1.0,
        },
    )

    low = model.predict_proba(no_evidence)
    high = model.predict_proba(exact_evidence)

    assert 0.01 <= low.probability <= 0.99
    assert 0.01 <= high.probability <= 0.99
    assert low.probability == 0.25
    assert high.probability > low.probability
    assert "exact_match" in high.reasons
    assert "tfidf_similarity" in high.reasons


def test_logistic_baseline_trains_and_predicts_deterministically():
    training = pd.DataFrame(
        [
            {"exact_match_count": 0, "max_tfidf_similarity": 0.0, "label": 0},
            {"exact_match_count": 0, "max_tfidf_similarity": 0.1, "label": 0},
            {"exact_match_count": 1, "max_tfidf_similarity": 0.7, "label": 1},
            {"exact_match_count": 1, "max_tfidf_similarity": 0.9, "label": 1},
        ]
    )
    model = LogisticBaseline(min_training_rows=4).fit(training, label_column="label")
    feature_vector = FeatureVector(
        target_phrase="traffic",
        features={"exact_match_count": 1.0, "max_tfidf_similarity": 0.8},
    )

    prediction = model.predict_proba(feature_vector)

    assert 0.01 <= prediction.probability <= 0.99
    assert prediction.model_version == "logistic-baseline-v0"
    assert "logistic_regression" in prediction.reasons
