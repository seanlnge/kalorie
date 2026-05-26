from kalorie.domain.models import DocumentChunk, MentionLabel, TargetPhrase
from kalorie.ml.embeddings import CachedEmbeddingProvider
from kalorie.ml.features import (
    extract_alias_embedding_feature_vectors,
    extract_embedding_feature_vectors,
    extract_feature_vectors,
    extract_scenario_embedding_feature_vectors,
    extract_template_embedding_feature_vectors,
    extract_transcript_recurrence_feature_vectors,
)
from kalorie.ml.labeling import label_document_chunks


def test_extract_feature_vectors_contains_expected_keys_and_counts():
    chunks = [
        DocumentChunk(
            document_id="doc",
            chunk_index=0,
            text="CAVA reports traffic growth and same restaurant sales strength.",
            section="headline",
            token_start=0,
            token_end=9,
        ),
        DocumentChunk(
            document_id="doc",
            chunk_index=1,
            text="Restaurant-level profit margin expanded.",
            section=None,
            token_start=9,
            token_end=13,
        ),
    ]
    targets = [
        TargetPhrase(phrase="traffic", normalized_phrase="traffic"),
        TargetPhrase(
            phrase="margin",
            normalized_phrase="margin",
            aliases=["restaurant-level profit margin"],
        ),
    ]
    labels = label_document_chunks(chunks, targets)

    vectors = {fv.target_phrase: fv for fv in extract_feature_vectors(chunks, targets, labels)}
    traffic_features = vectors["traffic"].features
    margin_features = vectors["margin"].features

    assert set(traffic_features) == {
        "exact_match_count",
        "lexical_match_count",
        "alias_count",
        "alias_exact_match_count",
        "alias_lexical_signal_binary",
        "alias_max_tfidf_similarity",
        "alias_appears_in_headline_or_first_chunk",
        "exact_signal_binary",
        "chunk_count",
        "target_token_count",
        "semantic_signal_max_tfidf",
        "semantic_signal_mean_top3_tfidf",
        "semantic_without_exact_high",
        "semantic_exact_gap",
        "hard_negative_neighbor_count",
        "hard_negative_neighbor_max_similarity",
        "hard_negative_neighbor_present",
        "max_tfidf_similarity",
        "mean_top3_tfidf_similarity",
        "chunks_above_0_20_similarity",
        "appears_in_headline_or_first_chunk",
    }
    assert traffic_features["exact_match_count"] == 1.0
    assert traffic_features["exact_signal_binary"] == 1.0
    assert traffic_features["chunk_count"] == 2.0
    assert traffic_features["target_token_count"] == 1.0
    assert traffic_features["max_tfidf_similarity"] > 0
    assert traffic_features["semantic_signal_max_tfidf"] == traffic_features["max_tfidf_similarity"]
    assert traffic_features["appears_in_headline_or_first_chunk"] == 1.0
    assert margin_features["lexical_match_count"] == 1.0
    assert margin_features["alias_count"] == 1.0
    assert margin_features["alias_exact_match_count"] == 1.0
    assert margin_features["alias_lexical_signal_binary"] == 1.0
    assert margin_features["appears_in_headline_or_first_chunk"] == 0.0


def test_alias_features_use_alias_text_without_changing_exact_signal():
    chunks = [
        DocumentChunk(
            document_id="doc",
            chunk_index=0,
            text="Nano Banana image generation was highlighted in the release.",
            section="headline",
            token_start=0,
            token_end=9,
        )
    ]
    targets = [
        TargetPhrase(
            phrase="gemini image model",
            normalized_phrase="gemini image model",
            aliases=["nano banana"],
        )
    ]
    labels = label_document_chunks(chunks, targets)

    features = extract_feature_vectors(chunks, targets, labels)[0].features

    assert features["exact_match_count"] == 0.0
    assert features["exact_signal_binary"] == 0.0
    assert features["lexical_match_count"] == 1.0
    assert features["alias_count"] == 1.0
    assert features["alias_exact_match_count"] == 1.0
    assert features["alias_lexical_signal_binary"] == 1.0
    assert features["alias_max_tfidf_similarity"] > features["max_tfidf_similarity"]
    assert features["alias_appears_in_headline_or_first_chunk"] == 1.0


def test_tfidf_similarity_is_higher_for_related_chunks_than_unrelated_chunks():
    chunks = [
        DocumentChunk(
            document_id="doc",
            chunk_index=0,
            text="Digital revenue and digital orders grew.",
            section=None,
            token_start=0,
            token_end=6,
        ),
        DocumentChunk(
            document_id="doc",
            chunk_index=1,
            text="The company opened new restaurants.",
            section=None,
            token_start=6,
            token_end=11,
        ),
    ]
    targets = [TargetPhrase(phrase="digital revenue", normalized_phrase="digital revenue")]
    labels = label_document_chunks(chunks, targets)

    features = extract_feature_vectors(chunks, targets, labels)[0].features

    assert features["max_tfidf_similarity"] > features["mean_top3_tfidf_similarity"] / 2
    assert features["chunks_above_0_20_similarity"] >= 1.0


def test_hard_negative_features_activate_for_similar_unmentioned_target():
    chunks = [
        DocumentChunk(
            document_id="doc",
            chunk_index=0,
            text="Gross margin expanded significantly this quarter.",
            section=None,
            token_start=0,
            token_end=7,
        ),
    ]
    targets = [
        TargetPhrase(phrase="margin", normalized_phrase="margin"),
        TargetPhrase(phrase="profit margin", normalized_phrase="profit margin"),
    ]
    labels = label_document_chunks(chunks, targets)
    vectors = {fv.target_phrase: fv for fv in extract_feature_vectors(chunks, targets, labels)}

    assert vectors["margin"].features["hard_negative_neighbor_present"] == 0.0
    assert vectors["profit margin"].features["hard_negative_neighbor_present"] == 1.0
    assert vectors["profit margin"].features["hard_negative_neighbor_max_similarity"] >= 0.35


def test_transcript_recurrence_features_are_zero_without_prior_calls():
    targets = [TargetPhrase(phrase="traffic", normalized_phrase="traffic")]

    features = extract_transcript_recurrence_feature_vectors(
        targets=targets,
        prior_label_sets=[],
    )[0].features

    assert features == {
        "prior_call_count": 0.0,
        "prior_mention_count": 0.0,
        "prior_mention_rate": 0.0,
        "prior_recent_mention_binary": 0.0,
        "prior_mention_streak": 0.0,
    }


def test_transcript_recurrence_features_count_recent_prior_mentions():
    targets = [
        TargetPhrase(phrase="traffic", normalized_phrase="traffic"),
        TargetPhrase(phrase="automation", normalized_phrase="automation"),
    ]
    prior_label_sets = [
        [
            MentionLabel(target_phrase="traffic", exact_mentioned=False, lexical_mentioned=False),
            MentionLabel(
                target_phrase="automation",
                exact_mentioned=True,
                lexical_mentioned=False,
            ),
        ],
        [
            MentionLabel(target_phrase="traffic", exact_mentioned=True, lexical_mentioned=False),
            MentionLabel(
                target_phrase="automation",
                exact_mentioned=False,
                lexical_mentioned=False,
            ),
        ],
        [
            MentionLabel(target_phrase="traffic", exact_mentioned=True, lexical_mentioned=False),
            MentionLabel(
                target_phrase="automation",
                exact_mentioned=False,
                lexical_mentioned=False,
            ),
        ],
    ]

    vectors = {
        vector.target_phrase: vector
        for vector in extract_transcript_recurrence_feature_vectors(
            targets=targets,
            prior_label_sets=prior_label_sets,
        )
    }

    assert vectors["traffic"].features["prior_call_count"] == 3.0
    assert vectors["traffic"].features["prior_mention_count"] == 2.0
    assert vectors["traffic"].features["prior_mention_rate"] == 0.666667
    assert vectors["traffic"].features["prior_recent_mention_binary"] == 1.0
    assert vectors["traffic"].features["prior_mention_streak"] == 2.0
    assert vectors["automation"].features["prior_mention_count"] == 1.0
    assert vectors["automation"].features["prior_recent_mention_binary"] == 0.0
    assert vectors["automation"].features["prior_mention_streak"] == 0.0


class SimpleEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "traffic": [1.0, 0.0],
            "Guest traffic improved in the quarter.": [1.0, 0.0],
            "Unrelated text about stores.": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]


def test_extract_embedding_feature_vectors_uses_provider_similarity():
    chunks = [
        DocumentChunk(
            document_id="doc",
            chunk_index=0,
            text="Guest traffic improved in the quarter.",
            section="headline",
            token_start=0,
            token_end=6,
        ),
        DocumentChunk(
            document_id="doc",
            chunk_index=1,
            text="Unrelated text about stores.",
            section=None,
            token_start=6,
            token_end=10,
        ),
    ]
    targets = [TargetPhrase(phrase="traffic", normalized_phrase="traffic")]

    features = extract_embedding_feature_vectors(chunks, targets, SimpleEmbeddingProvider())

    assert features[0].features["max_embedding_similarity"] == 1.0
    assert features[0].features["mean_top3_embedding_similarity"] == 0.5
    assert features[0].features["chunks_above_0_80_embedding_similarity"] == 1.0


class AliasEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "gemini image model": [0.0, 1.0],
            "nano banana": [1.0, 0.0],
            "Nano Banana was highlighted in the release.": [1.0, 0.0],
        }
        return [vectors[text] for text in texts]


def test_extract_alias_embedding_feature_vectors_uses_alias_expanded_retrieval():
    chunks = [
        DocumentChunk(
            document_id="doc",
            chunk_index=0,
            text="Nano Banana was highlighted in the release.",
            section=None,
            token_start=0,
            token_end=7,
        )
    ]
    targets = [
        TargetPhrase(
            phrase="gemini image model",
            normalized_phrase="gemini image model",
            aliases=["nano banana"],
        )
    ]

    features = extract_alias_embedding_feature_vectors(
        chunks,
        targets,
        AliasEmbeddingProvider(),
    )[0].features

    assert features["alias_max_embedding_similarity"] == 1.0
    assert features["alias_mean_top3_embedding_similarity"] == 1.0
    assert features["alias_chunks_above_0_80_embedding_similarity"] == 1.0
    assert features["alias_embedding_gap_vs_primary"] == 1.0


class TemplateEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "traffic growth": [1.0, 0.0],
            "guest traffic improved": [0.95, 0.05],
            "margin expansion": [0.0, 1.0],
            "Guest traffic improved in the quarter.": [1.0, 0.0],
            "Unrelated text about stores.": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]


def test_extract_template_embedding_feature_vectors_uses_template_phrase_similarity():
    chunks = [
        DocumentChunk(
            document_id="doc",
            chunk_index=0,
            text="Guest traffic improved in the quarter.",
            section=None,
            token_start=0,
            token_end=6,
        ),
        DocumentChunk(
            document_id="doc",
            chunk_index=1,
            text="Unrelated text about stores.",
            section=None,
            token_start=6,
            token_end=10,
        ),
    ]
    targets = [TargetPhrase(phrase="traffic", normalized_phrase="traffic")]
    template_phrases = {
        "traffic": ["traffic growth", "guest traffic improved"],
    }

    features = extract_template_embedding_feature_vectors(
        chunks,
        targets,
        template_phrases,
        TemplateEmbeddingProvider(),
    )[0].features

    assert features["template_phrase_count"] == 2.0
    assert features["max_template_embedding_similarity"] == 1.0
    assert features["mean_top5_template_embedding_similarity"] > 0.4
    assert features["template_pairs_above_0_80_similarity"] >= 1.0


class ScenarioEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "blackwell": [1.0, 0.0],
            "data center": [0.0, 1.0],
            "Blackwell demand remains very strong.": [1.0, 0.0],
            "Analysts may ask about Blackwell supply.": [0.9, 0.1],
            "Pre-call evidence emphasizes Blackwell demand.": [1.0, 0.0],
        }
        return [vectors[text] for text in texts]


def test_extract_scenario_embedding_feature_vectors_uses_event_catalog_texts():
    chunks = [
        DocumentChunk(
            document_id="doc",
            chunk_index=0,
            text="Pre-call evidence emphasizes Blackwell demand.",
            section=None,
            token_start=0,
            token_end=6,
        ),
    ]
    targets = [
        TargetPhrase(phrase="blackwell", normalized_phrase="blackwell"),
        TargetPhrase(phrase="data center", normalized_phrase="data center"),
    ]

    vectors = {
        vector.target_phrase: vector
        for vector in extract_scenario_embedding_feature_vectors(
            chunks=chunks,
            targets=targets,
            scenario_texts=[
                "Blackwell demand remains very strong.",
                "Analysts may ask about Blackwell supply.",
            ],
            provider=ScenarioEmbeddingProvider(),
        )
    }

    blackwell = vectors["blackwell"].features
    data_center = vectors["data center"].features
    assert blackwell["scenario_text_count"] == 2.0
    assert blackwell["max_scenario_embedding_similarity"] == 1.0
    assert blackwell["scenario_pairs_above_0_80_similarity"] == 2.0
    assert blackwell["scenario_evidence_support_max_similarity"] == 1.0
    assert data_center["max_scenario_embedding_similarity"] < 0.2


def test_cached_embedding_provider_recovers_from_corrupt_cache(tmp_path):
    cache_path = tmp_path / "embeddings.json"
    cache_path.write_text("", encoding="utf-8")

    provider = CachedEmbeddingProvider(
        provider=TemplateEmbeddingProvider(),
        cache_path=cache_path,
    )

    vectors = provider.embed(["traffic growth"])

    assert vectors == [[1.0, 0.0]]
    assert list(tmp_path.glob("embeddings.json.*.corrupt"))
