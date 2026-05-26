from statistics import mean

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from kalorie.domain.models import DocumentChunk, FeatureVector, MentionLabel, TargetPhrase
from kalorie.ml.embeddings import EmbeddingProvider
from kalorie.ml.labeling import find_exact_mentions, find_lexical_mentions

FEATURE_KEYS = [
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
]

TEMPLATE_FEATURE_KEYS = [
    "template_phrase_count",
    "max_template_embedding_similarity",
    "mean_top5_template_embedding_similarity",
    "template_pairs_above_0_80_similarity",
]
SCENARIO_FEATURE_KEYS = [
    "scenario_text_count",
    "max_scenario_embedding_similarity",
    "mean_top5_scenario_embedding_similarity",
    "scenario_pairs_above_0_80_similarity",
    "scenario_evidence_support_max_similarity",
]
ALIAS_EMBEDDING_FEATURE_KEYS = [
    "alias_max_embedding_similarity",
    "alias_mean_top3_embedding_similarity",
    "alias_chunks_above_0_80_embedding_similarity",
    "alias_embedding_gap_vs_primary",
]
RECURRENCE_FEATURE_KEYS = [
    "prior_call_count",
    "prior_mention_count",
    "prior_mention_rate",
    "prior_recent_mention_binary",
    "prior_mention_streak",
]


def extract_feature_vectors(
    chunks: list[DocumentChunk],
    targets: list[TargetPhrase],
    labels: list[MentionLabel],
) -> list[FeatureVector]:
    labels_by_target = {label.target_phrase: label for label in labels}
    index_by_target_phrase = {
        target.normalized_phrase: index for index, target in enumerate(targets)
    }
    mentioned_targets = {
        label.target_phrase for label in labels if label.exact_mentioned
    }
    chunk_texts = [chunk.text for chunk in chunks]
    tfidf_context = _build_tfidf_context(chunk_texts)
    target_similarity_matrix = _target_similarity_matrix(targets)
    return [
        FeatureVector(
            target_phrase=target.normalized_phrase,
            features=_extract_features_for_target(
                chunks,
                target,
                labels_by_target,
                index_by_target_phrase=index_by_target_phrase,
                mentioned_targets=mentioned_targets,
                chunk_texts=chunk_texts,
                tfidf_context=tfidf_context,
                target_similarity_matrix=target_similarity_matrix,
            ),
        )
        for target in targets
    ]


def extract_transcript_recurrence_feature_vectors(
    *,
    targets: list[TargetPhrase],
    prior_label_sets: list[list[MentionLabel]],
) -> list[FeatureVector]:
    labels_by_call = [
        {label.target_phrase: label for label in label_set}
        for label_set in prior_label_sets
    ]
    return [
        FeatureVector(
            target_phrase=target.normalized_phrase,
            features=_transcript_recurrence_features_for_target(
                target.normalized_phrase,
                labels_by_call,
            ),
        )
        for target in targets
    ]


def _extract_features_for_target(
    chunks: list[DocumentChunk],
    target: TargetPhrase,
    labels_by_target: dict[str, MentionLabel],
    *,
    index_by_target_phrase: dict[str, int],
    mentioned_targets: set[str],
    chunk_texts: list[str],
    tfidf_context: tuple[TfidfVectorizer, object] | None,
    target_similarity_matrix: list[list[float]],
) -> dict[str, float]:
    label = labels_by_target.get(target.normalized_phrase)
    similarities = _tfidf_similarities(
        target.normalized_phrase,
        chunk_texts,
        tfidf_context=tfidf_context,
    )
    alias_similarities = _alias_tfidf_similarities(
        target.aliases,
        chunk_texts,
        tfidf_context=tfidf_context,
    )
    top3 = sorted(similarities, reverse=True)[:3]
    first_chunk = chunks[0] if chunks else None
    appears_early = False
    alias_appears_early = False
    if first_chunk:
        appears_early = bool(
            find_exact_mentions(first_chunk.text, target.normalized_phrase)
            or find_lexical_mentions(first_chunk.text, target)
            or first_chunk.section == "headline"
            and find_exact_mentions(first_chunk.text, target.normalized_phrase)
        )
        alias_appears_early = bool(find_lexical_mentions(first_chunk.text, target))

    exact_count = _match_count(label, "exact")
    lexical_count = _match_count(label, "lexical")
    exact_signal = 1.0 if exact_count > 0 else 0.0
    max_tfidf = float(max(similarities, default=0.0))
    alias_max_tfidf = float(max(alias_similarities, default=0.0))
    mean_top3 = float(mean(top3) if top3 else 0.0)
    semantic_without_exact_high = 1.0 if exact_signal == 0.0 and max_tfidf >= 0.35 else 0.0
    (
        hard_negative_neighbor_count,
        hard_negative_neighbor_max_similarity,
    ) = _hard_negative_neighbor_stats(
        target_phrase=target.normalized_phrase,
        exact_signal=exact_signal,
        index_by_target_phrase=index_by_target_phrase,
        mentioned_targets=mentioned_targets,
        target_similarity_matrix=target_similarity_matrix,
    )
    return {
        "exact_match_count": float(exact_count),
        "lexical_match_count": float(lexical_count),
        "alias_count": float(len([alias for alias in target.aliases if alias.strip()])),
        "alias_exact_match_count": float(lexical_count),
        "alias_lexical_signal_binary": 1.0 if lexical_count > 0 else 0.0,
        "alias_max_tfidf_similarity": alias_max_tfidf,
        "alias_appears_in_headline_or_first_chunk": 1.0 if alias_appears_early else 0.0,
        "exact_signal_binary": exact_signal,
        "chunk_count": float(len(chunks)),
        "target_token_count": float(len(target.normalized_phrase.split())),
        "semantic_signal_max_tfidf": max_tfidf,
        "semantic_signal_mean_top3_tfidf": mean_top3,
        "semantic_without_exact_high": semantic_without_exact_high,
        "semantic_exact_gap": max_tfidf - exact_signal,
        "hard_negative_neighbor_count": hard_negative_neighbor_count,
        "hard_negative_neighbor_max_similarity": hard_negative_neighbor_max_similarity,
        "hard_negative_neighbor_present": 1.0 if hard_negative_neighbor_count > 0 else 0.0,
        "max_tfidf_similarity": max_tfidf,
        "mean_top3_tfidf_similarity": mean_top3,
        "chunks_above_0_20_similarity": float(sum(score >= 0.20 for score in similarities)),
        "appears_in_headline_or_first_chunk": 1.0 if appears_early else 0.0,
    }


def _transcript_recurrence_features_for_target(
    target_phrase: str,
    labels_by_call: list[dict[str, MentionLabel]],
) -> dict[str, float]:
    mentioned_by_call = [
        bool(labels.get(target_phrase) and labels[target_phrase].exact_mentioned)
        for labels in labels_by_call
    ]
    prior_call_count = len(mentioned_by_call)
    prior_mention_count = sum(mentioned_by_call)
    streak = 0
    for mentioned in reversed(mentioned_by_call):
        if not mentioned:
            break
        streak += 1
    return {
        "prior_call_count": float(prior_call_count),
        "prior_mention_count": float(prior_mention_count),
        "prior_mention_rate": round(prior_mention_count / prior_call_count, 6)
        if prior_call_count
        else 0.0,
        "prior_recent_mention_binary": 1.0
        if mentioned_by_call and mentioned_by_call[-1]
        else 0.0,
        "prior_mention_streak": float(streak),
    }


def _target_similarity_matrix(targets: list[TargetPhrase]) -> list[list[float]]:
    phrases = [target.normalized_phrase for target in targets]
    if not phrases:
        return []
    if len(phrases) < 2:
        return [[1.0]]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform(phrases)
    except ValueError:
        return [[1.0 if i == j else 0.0 for j in range(len(phrases))] for i in range(len(phrases))]
    similarities = cosine_similarity(matrix, matrix)
    return [
        [float(value) for value in row]
        for row in similarities
    ]


def _hard_negative_neighbor_stats(
    *,
    target_phrase: str,
    exact_signal: float,
    index_by_target_phrase: dict[str, int],
    mentioned_targets: set[str],
    target_similarity_matrix: list[list[float]],
    threshold: float = 0.35,
) -> tuple[float, float]:
    if exact_signal > 0:
        return 0.0, 0.0
    target_index = index_by_target_phrase.get(target_phrase)
    if target_index is None:
        return 0.0, 0.0
    neighbor_similarities = []
    for mentioned in mentioned_targets:
        neighbor_index = index_by_target_phrase.get(mentioned)
        if neighbor_index is None or neighbor_index == target_index:
            continue
        if target_index >= len(target_similarity_matrix):
            continue
        if neighbor_index >= len(target_similarity_matrix[target_index]):
            continue
        similarity = target_similarity_matrix[target_index][neighbor_index]
        if similarity >= threshold:
            neighbor_similarities.append(similarity)
    if not neighbor_similarities:
        return 0.0, 0.0
    return float(len(neighbor_similarities)), float(max(neighbor_similarities))


def _match_count(label: MentionLabel | None, match_type: str) -> int:
    if label is None:
        return 0
    return sum(span.match_type == match_type for span in label.match_spans)


def _build_tfidf_context(chunk_texts: list[str]) -> tuple[TfidfVectorizer, object] | None:
    if not chunk_texts:
        return None
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    try:
        chunk_matrix = vectorizer.fit_transform(chunk_texts)
    except ValueError:
        # All chunks were effectively empty after tokenization/stopwording.
        return None
    return vectorizer, chunk_matrix


def _tfidf_similarities(
    target_phrase: str,
    chunk_texts: list[str],
    *,
    tfidf_context: tuple[TfidfVectorizer, object] | None = None,
) -> list[float]:
    if not chunk_texts:
        return []
    context = tfidf_context or _build_tfidf_context(chunk_texts)
    if context is None:
        return [0.0 for _ in chunk_texts]
    vectorizer, chunk_matrix = context
    target_vector = vectorizer.transform([target_phrase])
    scores = cosine_similarity(target_vector, chunk_matrix).ravel()
    return [float(score) for score in scores]


def _alias_tfidf_similarities(
    aliases: list[str],
    chunk_texts: list[str],
    *,
    tfidf_context: tuple[TfidfVectorizer, object] | None = None,
) -> list[float]:
    alias_scores: list[float] = []
    for alias in dict.fromkeys(alias.strip() for alias in aliases if alias.strip()):
        alias_scores.extend(
            _tfidf_similarities(
                alias,
                chunk_texts,
                tfidf_context=tfidf_context,
            )
        )
    return alias_scores


def extract_embedding_feature_vectors(
    chunks: list[DocumentChunk],
    targets: list[TargetPhrase],
    provider: EmbeddingProvider,
) -> list[FeatureVector]:
    chunk_texts = [chunk.text for chunk in chunks]
    return [
        FeatureVector(
            target_phrase=target.normalized_phrase,
            features=_embedding_features_for_target(
                target.normalized_phrase,
                chunk_texts,
                provider,
            ),
        )
        for target in targets
    ]


def extract_alias_embedding_feature_vectors(
    chunks: list[DocumentChunk],
    targets: list[TargetPhrase],
    provider: EmbeddingProvider,
) -> list[FeatureVector]:
    chunk_texts = [chunk.text for chunk in chunks]
    return [
        FeatureVector(
            target_phrase=target.normalized_phrase,
            features=_alias_embedding_features_for_target(
                target,
                chunk_texts,
                provider,
            ),
        )
        for target in targets
    ]


def extract_template_embedding_feature_vectors(
    chunks: list[DocumentChunk],
    targets: list[TargetPhrase],
    template_phrases_by_target: dict[str, list[str]],
    provider: EmbeddingProvider,
) -> list[FeatureVector]:
    chunk_texts = [chunk.text for chunk in chunks]
    if not chunk_texts:
        return [
            FeatureVector(
                target_phrase=target.normalized_phrase,
                features=_template_embedding_features_for_target(
                    target.normalized_phrase,
                    chunk_texts,
                    template_phrases_by_target,
                    provider,
                ),
            )
            for target in targets
        ]
    unique_templates = list(
        dict.fromkeys(
            template
            for target in targets
            for template in template_phrases_by_target.get(target.normalized_phrase, [])
        )
    )
    if not unique_templates:
        return [
            FeatureVector(
                target_phrase=target.normalized_phrase,
                features={
                    "template_phrase_count": 0.0,
                    "max_template_embedding_similarity": 0.0,
                    "mean_top5_template_embedding_similarity": 0.0,
                    "template_pairs_above_0_80_similarity": 0.0,
                },
            )
            for target in targets
        ]
    vectors = provider.embed([*unique_templates, *chunk_texts])
    template_vectors = {
        template: vector
        for template, vector in zip(unique_templates, vectors[: len(unique_templates)], strict=True)
    }
    chunk_vectors = vectors[len(unique_templates) :]
    return [
        FeatureVector(
            target_phrase=target.normalized_phrase,
            features=_template_embedding_features_for_target_vectors(
                templates=template_phrases_by_target.get(target.normalized_phrase, []),
                template_vectors=template_vectors,
                chunk_vectors=chunk_vectors,
            ),
        )
        for target in targets
    ]


def extract_scenario_embedding_feature_vectors(
    *,
    chunks: list[DocumentChunk],
    targets: list[TargetPhrase],
    scenario_texts: list[str],
    provider: EmbeddingProvider,
) -> list[FeatureVector]:
    chunk_texts = [chunk.text for chunk in chunks]
    unique_scenario_texts = list(dict.fromkeys(text for text in scenario_texts if text.strip()))
    if not unique_scenario_texts:
        return [
            FeatureVector(
                target_phrase=target.normalized_phrase,
                features=_empty_scenario_features(),
            )
            for target in targets
        ]
    embed_texts = [
        *(target.normalized_phrase for target in targets),
        *unique_scenario_texts,
        *chunk_texts,
    ]
    vectors = provider.embed(embed_texts)
    target_vectors = vectors[: len(targets)]
    scenario_vectors = vectors[len(targets) : len(targets) + len(unique_scenario_texts)]
    chunk_vectors = vectors[len(targets) + len(unique_scenario_texts) :]
    evidence_support = _scenario_evidence_support(
        scenario_vectors=scenario_vectors,
        chunk_vectors=chunk_vectors,
    )
    return [
        FeatureVector(
            target_phrase=target.normalized_phrase,
            features=_scenario_embedding_features_for_target_vectors(
                scenario_vectors=scenario_vectors,
                target_vector=target_vector,
                evidence_support=evidence_support,
            ),
        )
        for target, target_vector in zip(targets, target_vectors, strict=True)
    ]


def _embedding_features_for_target(
    target_phrase: str,
    chunk_texts: list[str],
    provider: EmbeddingProvider,
) -> dict[str, float]:
    if not chunk_texts:
        return {
            "max_embedding_similarity": 0.0,
            "mean_top3_embedding_similarity": 0.0,
            "chunks_above_0_80_embedding_similarity": 0.0,
        }
    vectors = provider.embed([target_phrase, *chunk_texts])
    target_vector = vectors[0]
    similarities = [_cosine(target_vector, vector) for vector in vectors[1:]]
    top3 = sorted(similarities, reverse=True)[:3]
    return {
        "max_embedding_similarity": round(max(similarities, default=0.0), 6),
        "mean_top3_embedding_similarity": round(mean(top3) if top3 else 0.0, 6),
        "chunks_above_0_80_embedding_similarity": float(
            sum(score >= 0.80 for score in similarities)
        ),
    }


def _alias_embedding_features_for_target(
    target: TargetPhrase,
    chunk_texts: list[str],
    provider: EmbeddingProvider,
) -> dict[str, float]:
    aliases = list(dict.fromkeys(alias.strip() for alias in target.aliases if alias.strip()))
    if not aliases or not chunk_texts:
        return {
            "alias_max_embedding_similarity": 0.0,
            "alias_mean_top3_embedding_similarity": 0.0,
            "alias_chunks_above_0_80_embedding_similarity": 0.0,
            "alias_embedding_gap_vs_primary": 0.0,
        }
    vectors = provider.embed([target.normalized_phrase, *aliases, *chunk_texts])
    primary_vector = vectors[0]
    alias_vectors = vectors[1 : 1 + len(aliases)]
    chunk_vectors = vectors[1 + len(aliases) :]
    alias_scores = [
        _cosine(alias_vector, chunk_vector)
        for alias_vector in alias_vectors
        for chunk_vector in chunk_vectors
    ]
    primary_scores = [_cosine(primary_vector, chunk_vector) for chunk_vector in chunk_vectors]
    top3 = sorted(alias_scores, reverse=True)[:3]
    alias_max = max(alias_scores, default=0.0)
    primary_max = max(primary_scores, default=0.0)
    return {
        "alias_max_embedding_similarity": round(alias_max, 6),
        "alias_mean_top3_embedding_similarity": round(mean(top3) if top3 else 0.0, 6),
        "alias_chunks_above_0_80_embedding_similarity": float(
            sum(score >= 0.80 for score in alias_scores)
        ),
        "alias_embedding_gap_vs_primary": round(alias_max - primary_max, 6),
    }


def _template_embedding_features_for_target(
    target_phrase: str,
    chunk_texts: list[str],
    template_phrases_by_target: dict[str, list[str]],
    provider: EmbeddingProvider,
) -> dict[str, float]:
    templates = template_phrases_by_target.get(target_phrase, [])
    if not templates or not chunk_texts:
        return {
            "template_phrase_count": float(len(templates)),
            "max_template_embedding_similarity": 0.0,
            "mean_top5_template_embedding_similarity": 0.0,
            "template_pairs_above_0_80_similarity": 0.0,
        }
    vectors = provider.embed([*templates, *chunk_texts])
    template_vectors = vectors[: len(templates)]
    chunk_vectors = vectors[len(templates) :]
    pair_scores = [
        _cosine(template_vector, chunk_vector)
        for template_vector in template_vectors
        for chunk_vector in chunk_vectors
    ]
    top5 = sorted(pair_scores, reverse=True)[:5]
    return {
        "template_phrase_count": float(len(templates)),
        "max_template_embedding_similarity": round(max(pair_scores, default=0.0), 6),
        "mean_top5_template_embedding_similarity": round(mean(top5) if top5 else 0.0, 6),
        "template_pairs_above_0_80_similarity": float(sum(score >= 0.80 for score in pair_scores)),
    }


def _scenario_embedding_features_for_target_vectors(
    *,
    scenario_vectors: list[list[float]],
    target_vector: list[float],
    evidence_support: float,
) -> dict[str, float]:
    if not scenario_vectors:
        return _empty_scenario_features()
    similarities = [_cosine(target_vector, vector) for vector in scenario_vectors]
    top5 = sorted(similarities, reverse=True)[:5]
    return {
        "scenario_text_count": float(len(scenario_vectors)),
        "max_scenario_embedding_similarity": round(max(similarities, default=0.0), 6),
        "mean_top5_scenario_embedding_similarity": round(mean(top5) if top5 else 0.0, 6),
        "scenario_pairs_above_0_80_similarity": float(sum(score >= 0.80 for score in similarities)),
        "scenario_evidence_support_max_similarity": round(evidence_support, 6),
    }


def _scenario_evidence_support(
    *,
    scenario_vectors: list[list[float]],
    chunk_vectors: list[list[float]],
) -> float:
    if not scenario_vectors or not chunk_vectors:
        return 0.0
    return max(
        _cosine(scenario_vector, chunk_vector)
        for scenario_vector in scenario_vectors
        for chunk_vector in chunk_vectors
    )


def _empty_scenario_features() -> dict[str, float]:
    return {
        "scenario_text_count": 0.0,
        "max_scenario_embedding_similarity": 0.0,
        "mean_top5_scenario_embedding_similarity": 0.0,
        "scenario_pairs_above_0_80_similarity": 0.0,
        "scenario_evidence_support_max_similarity": 0.0,
    }


def _template_embedding_features_for_target_vectors(
    *,
    templates: list[str],
    template_vectors: dict[str, list[float]],
    chunk_vectors: list[list[float]],
) -> dict[str, float]:
    if not templates or not chunk_vectors:
        return {
            "template_phrase_count": float(len(templates)),
            "max_template_embedding_similarity": 0.0,
            "mean_top5_template_embedding_similarity": 0.0,
            "template_pairs_above_0_80_similarity": 0.0,
        }
    pair_scores = [
        _cosine(template_vectors[template], chunk_vector)
        for template in templates
        if template in template_vectors
        for chunk_vector in chunk_vectors
    ]
    top5 = sorted(pair_scores, reverse=True)[:5]
    return {
        "template_phrase_count": float(len(templates)),
        "max_template_embedding_similarity": round(max(pair_scores, default=0.0), 6),
        "mean_top5_template_embedding_similarity": round(mean(top5) if top5 else 0.0, 6),
        "template_pairs_above_0_80_similarity": float(sum(score >= 0.80 for score in pair_scores)),
    }


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
