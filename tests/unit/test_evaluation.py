from kalorie.domain.models import MentionLabel, Prediction
from kalorie.ml.evaluation import evaluate_binary_predictions


def test_brier_score_is_primary_binary_probability_metric():
    predictions = [
        Prediction(
            target_phrase="traffic",
            model_version="rule-based-v0",
            probability=0.75,
            reasons=["exact_match"],
        ),
        Prediction(
            target_phrase="digital revenue",
            model_version="rule-based-v0",
            probability=0.25,
            reasons=["base_rate"],
        ),
    ]
    labels = [
        MentionLabel(target_phrase="traffic", exact_mentioned=True, lexical_mentioned=False),
        MentionLabel(
            target_phrase="digital revenue",
            exact_mentioned=False,
            lexical_mentioned=False,
        ),
    ]

    report = evaluate_binary_predictions(predictions, labels, evaluation_kind="smoke")

    assert report.sample_count == 2
    assert report.brier_score == 0.0625
    assert report.expected_calibration_error == 0.25
    assert not hasattr(report, "mean_squared_error")
    assert report.evaluation_kind == "smoke"
