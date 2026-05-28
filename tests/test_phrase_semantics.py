from kalorie2.phrase_semantics import antonym_axis_features


def test_antonym_axis_features_score_phrase_by_relative_embedding_distance():
    embeddings = {
        "tariff": [0.9, 0.1],
        "risk": [1.0, 0.0],
        "opportunity": [0.0, 1.0],
    }

    features = antonym_axis_features(
        "tariff",
        embeddings=embeddings,
        axes={"risk_opportunity": ("risk", "opportunity")},
    )

    assert features["phrase_semantic_embedding_available"] == 1.0
    assert features["phrase_semantic_axis_risk_opportunity"] > 0.8


def test_antonym_axis_features_are_missing_safe():
    features = antonym_axis_features(
        "unknown",
        embeddings={"risk": [1.0, 0.0], "opportunity": [0.0, 1.0]},
        axes={"risk_opportunity": ("risk", "opportunity")},
    )

    assert features["phrase_semantic_embedding_available"] == 0.0
    assert features["phrase_semantic_axis_risk_opportunity"] == 0.0

