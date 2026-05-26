from kalorie.ml.grouped_calibration import (
    GroupedCalibrationExample,
    evidence_strength_bucket,
    fit_grouped_temperature_calibration,
)


def test_evidence_strength_bucket_uses_exact_alias_and_semantic_support():
    assert evidence_strength_bucket({"exact_match_count": 1.0}) == "strong"
    assert evidence_strength_bucket({"alias_lexical_signal_binary": 1.0}) == "strong"
    assert evidence_strength_bucket({"semantic_signal_max_tfidf": 0.5}) == "medium"
    assert evidence_strength_bucket({"semantic_signal_max_tfidf": 0.1}) == "weak"


def test_grouped_temperature_calibration_routes_by_category_and_evidence_bucket():
    examples = [
        GroupedCalibrationExample("macro", "strong", 0.90, 0),
        GroupedCalibrationExample("macro", "strong", 0.85, 0),
        GroupedCalibrationExample("macro", "strong", 0.80, 0),
        GroupedCalibrationExample("macro", "strong", 0.75, 0),
        GroupedCalibrationExample("generic", "weak", 0.80, 1),
        GroupedCalibrationExample("generic", "weak", 0.85, 1),
        GroupedCalibrationExample("generic", "weak", 0.90, 1),
        GroupedCalibrationExample("generic", "weak", 0.95, 1),
    ]

    model = fit_grouped_temperature_calibration(
        examples,
        min_group_rows=4,
        shrinkage=2.0,
    )
    macro_calibrated = model.calibrate(
        0.90,
        category="macro",
        evidence_bucket="strong",
    )
    fallback_calibrated = model.calibrate(
        0.90,
        category="macro",
        evidence_bucket="unknown",
    )
    generic_calibrated = model.calibrate(
        0.90,
        category="generic",
        evidence_bucket="weak",
    )

    assert "macro::strong" in model.group_temperatures
    assert macro_calibrated < 0.90
    assert generic_calibrated > fallback_calibrated
