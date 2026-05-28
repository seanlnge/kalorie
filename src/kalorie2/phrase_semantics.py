import json
from functools import lru_cache
from pathlib import Path

from kalorie2.embedding_patterns import cosine_similarity

DEFAULT_PHRASE_SEMANTICS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "phrase_semantic_embeddings.json"
)

DEFAULT_ANTONYM_AXES = {
    "risk_opportunity": ("risk", "opportunity"),
    "cost_pressure_margin_expansion": ("cost pressure", "margin expansion"),
    "macro_product": ("macro", "product"),
    "specific_entity_generic_concept": ("specific entity", "generic concept"),
}


def antonym_axis_features(
    phrase: str,
    *,
    embeddings: dict[str, list[float]],
    axes: dict[str, tuple[str, str]] = DEFAULT_ANTONYM_AXES,
) -> dict[str, float]:
    normalized_phrase = phrase.strip().lower()
    features = _empty_axis_features(axes)
    phrase_embedding = embeddings.get(normalized_phrase)
    if phrase_embedding is None:
        return features

    features["phrase_semantic_embedding_available"] = 1.0
    for axis_name, (positive_anchor, negative_anchor) in axes.items():
        positive = embeddings.get(positive_anchor)
        negative = embeddings.get(negative_anchor)
        if positive is None or negative is None:
            continue
        positive_distance = 1.0 - cosine_similarity(phrase_embedding, positive)
        negative_distance = 1.0 - cosine_similarity(phrase_embedding, negative)
        denominator = positive_distance + negative_distance
        features[f"phrase_semantic_axis_{axis_name}"] = (
            (negative_distance - positive_distance) / denominator
            if denominator > 0.0
            else 0.0
        )
    return features


def default_antonym_axis_features(phrase: str) -> dict[str, float]:
    return antonym_axis_features(phrase, embeddings=default_phrase_semantic_embeddings())


@lru_cache(maxsize=1)
def default_phrase_semantic_embeddings() -> dict[str, list[float]]:
    if not DEFAULT_PHRASE_SEMANTICS_PATH.exists():
        return {}
    payload = json.loads(DEFAULT_PHRASE_SEMANTICS_PATH.read_text(encoding="utf-8"))
    return {
        key.strip().lower(): value
        for key, value in payload.get("embeddings", {}).items()
    }


def _empty_axis_features(axes: dict[str, tuple[str, str]]) -> dict[str, float]:
    features = {"phrase_semantic_embedding_available": 0.0}
    features.update({f"phrase_semantic_axis_{axis_name}": 0.0 for axis_name in axes})
    return features

