import math
import re

from kalorie2.company_metadata import default_company_metadata_features
from kalorie2.event_scenarios import EventScenarioCatalog, TargetPhraseContext
from kalorie2.phrase_semantics import default_antonym_axis_features
from kalorie2.prediction_types import PredictionInputRow
from kalorie2.transcript_model import _split_min_count, _word_options, _word_tokens
from kalorie2.web_evidence import WebEvidencePacket

_LABEL_ONLY_FIELDS = frozenset({"final_outcome", "outcome_label", "settlement_ts", "label"})

_MACRO_TERMS = frozenset(
    {
        "currency",
        "fx",
        "gas",
        "gasoline",
        "headwind",
        "inflation",
        "interest",
        "oil",
        "rate",
        "rates",
        "tariff",
        "tailwind",
    }
)

_GENERIC_BUSINESS_TERMS = frozenset(
    {
        "buyback",
        "cost",
        "costs",
        "dividend",
        "efficiency",
        "growth",
        "inventory",
        "margin",
        "margins",
        "manufacturing",
        "revenue",
        "sales",
    }
)

_SEMANTIC_BUCKETS = {
    "macro": (
        "consumer demand",
        "currency",
        "fx",
        "inflation",
        "interest rates",
        "macro",
        "macroeconomics",
        "oil",
        "tariff",
    ),
    "regulatory": (
        "compliance",
        "policy",
        "regulation",
        "regulatory",
        "tariff",
        "tax",
        "trade policy",
    ),
    "operations": (
        "inventory",
        "logistics",
        "membership",
        "membership fee",
        "operations",
        "stores",
        "supply chain",
    ),
    "product": (
        "brand",
        "customer",
        "launch",
        "product",
        "service",
    ),
    "labor": (
        "compensation",
        "employee",
        "hiring",
        "labor",
        "wage",
        "workforce",
    ),
    "technology": (
        "ai",
        "artificial intelligence",
        "automation",
        "cloud",
        "data",
        "software",
        "technology",
    ),
    "finance": (
        "buyback",
        "cash flow",
        "debt",
        "dividend",
        "finance",
        "margin",
        "revenue",
    ),
    "generic_business": tuple(sorted(_GENERIC_BUSINESS_TERMS)),
}


def extract_market_features(row: PredictionInputRow) -> dict[str, float]:
    mid = float(row.preclose_yes_mid)
    bid = float(row.preclose_yes_bid)
    ask = float(row.preclose_yes_ask)
    spread = max(0.0, ask - bid)
    bid_size = float(row.preclose_yes_bid_size)
    ask_size = float(row.preclose_yes_ask_size)
    total_size = bid_size + ask_size
    hours_before_close = max(
        0.0,
        (row.close_time - row.snapshot_target_time).total_seconds() / 3600.0,
    )
    return {
        "market_yes_bid": bid,
        "market_yes_ask": ask,
        "market_yes_mid": mid,
        "market_mid_logit": _safe_logit(mid),
        "market_spread": spread,
        "market_spread_share_of_mid": spread / mid if mid > 0.0 else 0.0,
        "market_no_bid": max(0.0, 1.0 - ask),
        "market_no_ask": min(1.0, 1.0 - bid),
        "market_bid_present": 1.0 if bid > 0.0 else 0.0,
        "market_ask_present": 1.0 if ask > 0.0 else 0.0,
        "market_preclose_volume": float(row.preclose_volume),
        "market_preclose_volume_present": 1.0 if row.preclose_volume > 0 else 0.0,
        "market_preclose_volume_log": math.log1p(row.preclose_volume),
        "market_preclose_open_interest": float(row.preclose_open_interest),
        "market_preclose_open_interest_present": (
            1.0 if row.preclose_open_interest > 0 else 0.0
        ),
        "market_preclose_open_interest_log": math.log1p(row.preclose_open_interest),
        "market_preclose_yes_bid_size": bid_size,
        "market_preclose_yes_ask_size": ask_size,
        "market_preclose_size_imbalance": (
            (bid_size - ask_size) / total_size if total_size > 0.0 else 0.0
        ),
        "snapshot_staleness_seconds": float(row.snapshot_staleness_seconds),
        "snapshot_staleness_hours": row.snapshot_staleness_seconds / 3600.0,
        "hours_before_close": hours_before_close,
        "log_hours_before_close": math.log1p(hours_before_close),
        "hours_before_close_bucket_2_6": _bucket_flag(hours_before_close, 2.0, 6.0),
        "hours_before_close_bucket_6_12": _bucket_flag(hours_before_close, 6.0, 12.0),
        "hours_before_close_bucket_12_24": _bucket_flag(hours_before_close, 12.0, 24.0),
        "hours_before_close_bucket_24_48": _bucket_flag(hours_before_close, 24.0, 48.0),
        "company_prior_call_count": float(row.company_prior_call_count),
        "company_avg_call_duration_minutes_prior": row.company_avg_call_duration_minutes_prior,
        "company_avg_qa_question_count_prior": row.company_avg_qa_question_count_prior,
        "company_avg_prepared_remarks_minutes_prior": (
            row.company_avg_prepared_remarks_minutes_prior
        ),
        "company_qa_share_prior": row.company_qa_share_prior,
        "company_question_count_trend_prior": row.company_question_count_trend_prior,
        "company_transcript_coverage_count": float(row.company_transcript_coverage_count),
        "company_transcript_style_available": float(row.company_transcript_style_available),
        "company_transcript_style_missing": (
            0.0 if row.company_transcript_style_available else 1.0
        ),
        "company_avg_transcript_word_count_prior": row.company_avg_transcript_word_count_prior,
        "company_avg_phrase_mentions_prior": row.company_avg_phrase_mentions_prior,
    }


def extract_phrase_features(row: PredictionInputRow) -> dict[str, float]:
    base_phrase, count_threshold = _split_min_count(row.word_said)
    options = _word_options(base_phrase)
    tokens = _phrase_tokens(base_phrase)
    normalized_tokens = set(_normalized_tokens(base_phrase))
    return {
        "phrase_token_count": float(len(tokens)),
        "phrase_option_count": float(len(options)),
        "phrase_has_slash_alternatives": 1.0 if len(options) > 1 else 0.0,
        "phrase_has_count_threshold": 1.0 if count_threshold > 1 else 0.0,
        "phrase_count_threshold": float(count_threshold),
        "phrase_is_single_word": 1.0 if len(tokens) == 1 else 0.0,
        "phrase_is_multiword": 1.0 if len(tokens) > 1 else 0.0,
        "phrase_is_macro": 1.0 if normalized_tokens & _MACRO_TERMS else 0.0,
        "phrase_is_generic_business": 1.0
        if normalized_tokens & _GENERIC_BUSINESS_TERMS
        else 0.0,
        "phrase_is_entity_like": 1.0 if _looks_entity_like(row.word_said) else 0.0,
        **_semantic_bucket_features(base_phrase),
        **default_antonym_axis_features(base_phrase),
    }


def extract_resolution_features(row: PredictionInputRow) -> dict[str, float]:
    base_phrase, count_threshold = _split_min_count(row.normalized_word_said or row.word_said)
    options = _word_options(base_phrase)
    option_tokens = [set(_normalized_tokens(option)) for option in options]
    unique_tokens = set().union(*option_tokens) if option_tokens else set()
    average_option_tokens = (
        sum(len(tokens) for tokens in option_tokens) / len(option_tokens)
        if option_tokens
        else 0.0
    )
    breadth_denominator = max(1.0, average_option_tokens)
    return {
        "resolution_option_count": float(len(options)),
        "resolution_requires_count_threshold": 1.0 if count_threshold > 1 else 0.0,
        "resolution_minimum_count": float(count_threshold),
        "resolution_has_alternatives": 1.0 if len(options) > 1 else 0.0,
        "resolution_phrase_breadth_score": max(
            0.0,
            (len(unique_tokens) / breadth_denominator) - 1.0,
        ),
    }


def extract_scenario_features(
    row: PredictionInputRow,
    catalog: EventScenarioCatalog | None,
) -> dict[str, float]:
    if catalog is None:
        return {
            "scenario_available": 0.0,
            "scenario_text_count": 0.0,
            "scenario_topic_overlap_max": 0.0,
            "scenario_target_context_available": 0.0,
            "scenario_target_context_overlap_max": 0.0,
        }

    target = _target_base_phrase(row)
    matching_context = _matching_target_context(target, catalog.target_phrase_contexts)
    context_texts = matching_context.texts() if matching_context is not None else []
    return {
        "scenario_available": 1.0,
        "scenario_text_count": float(len(catalog.scenario_texts())),
        "scenario_topic_overlap_max": _max_overlap(target, catalog.topics),
        "scenario_target_context_available": 1.0 if matching_context is not None else 0.0,
        "scenario_target_context_overlap_max": _max_overlap(target, context_texts),
    }


def extract_web_evidence_features(
    row: PredictionInputRow,
    web_evidence: WebEvidencePacket | None,
) -> dict[str, float]:
    if web_evidence is None:
        return {
            "web_evidence_available": 0.0,
            "web_evidence_item_count": 0.0,
            "web_evidence_target_overlap": 0.0,
            "web_evidence_strength_max": 0.0,
            "web_evidence_cutoff_safe_count": 0.0,
            "web_evidence_target_match_count": 0.0,
            "web_evidence_target_match_share": 0.0,
            "web_evidence_strength_mean": 0.0,
            "web_evidence_strength_sum": 0.0,
            "web_evidence_relevance_mean": 0.0,
            "web_evidence_relevance_max": 0.0,
            "web_evidence_high_relevance_count": 0.0,
            "web_evidence_support_count": 0.0,
            "web_evidence_against_count": 0.0,
            "web_evidence_neutral_count": 0.0,
            "web_evidence_recency_min_hours": 0.0,
            "web_evidence_recency_mean_hours": 0.0,
            "web_evidence_source_company": 0.0,
            "web_evidence_source_sec": 0.0,
            "web_evidence_source_news": 0.0,
            "web_evidence_source_analyst": 0.0,
            "web_evidence_source_other": 0.0,
        }
    return web_evidence.features_for_target(
        _target_base_phrase(row),
        cutoff_time=row.snapshot_target_time,
    )


def build_feature_row(
    row: PredictionInputRow,
    *,
    catalog: EventScenarioCatalog | None = None,
    web_evidence: WebEvidencePacket | None = None,
) -> dict[str, float]:
    features = {
        **extract_market_features(row),
        **default_company_metadata_features(row.series_ticker),
        **extract_phrase_features(row),
        **extract_resolution_features(row),
        **extract_scenario_features(row, catalog),
        **extract_web_evidence_features(row, web_evidence),
    }
    return {
        key: value
        for key, value in features.items()
        if key not in _LABEL_ONLY_FIELDS
    }


def build_feature_matrix(
    rows: list[PredictionInputRow],
    catalogs_by_event: dict[str, EventScenarioCatalog],
    web_evidence_by_event: dict[str, WebEvidencePacket] | None = None,
) -> list[dict[str, float]]:
    web_evidence_by_event = web_evidence_by_event or {}
    return [
        build_feature_row(
            row,
            catalog=catalogs_by_event.get(row.event_ticker),
            web_evidence=web_evidence_by_event.get(row.event_ticker),
        )
        for row in rows
    ]


def _safe_logit(probability: float) -> float:
    clipped = min(0.999999, max(0.000001, probability))
    return math.log(clipped / (1.0 - clipped))


def _bucket_flag(value: float, low: float, high: float) -> float:
    return 1.0 if low <= value < high else 0.0


def _target_base_phrase(row: PredictionInputRow) -> str:
    base_phrase, _ = _split_min_count(row.normalized_word_said or row.word_said)
    return base_phrase


def _matching_target_context(
    target: str,
    contexts: list[TargetPhraseContext],
) -> TargetPhraseContext | None:
    target_tokens = set(_normalized_tokens(target))
    for context in contexts:
        context_tokens = set(_normalized_tokens(context.target_phrase))
        if context.target_phrase.lower() == target.lower() or target_tokens == context_tokens:
            return context
    return None


def _max_overlap(target: str, texts: list[str]) -> float:
    if not texts:
        return 0.0
    return max(_token_overlap(target, text) for text in texts)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(_normalized_tokens(left))
    right_tokens = set(_normalized_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / math.sqrt(len(left_tokens) * len(right_tokens))


def _semantic_bucket_features(phrase: str) -> dict[str, float]:
    return {
        f"phrase_semantic_{bucket}_score": _semantic_bucket_score(phrase, anchors)
        for bucket, anchors in _SEMANTIC_BUCKETS.items()
    }


def _semantic_bucket_score(phrase: str, anchors: tuple[str, ...]) -> float:
    if not anchors:
        return 0.0
    return max(_token_overlap(phrase, anchor) for anchor in anchors)


def _phrase_tokens(phrase: str) -> list[str]:
    return _word_tokens(phrase)


def _normalized_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    normalized = []
    for token in tokens:
        if len(token) > 3 and token.endswith("s"):
            normalized.append(token[:-1])
        else:
            normalized.append(token)
    return normalized


def _looks_entity_like(display_phrase: str) -> bool:
    compact_words = re.findall(r"[A-Za-z][A-Za-z0-9]*", display_phrase)
    return any(
        (word.isupper() and len(word) > 1) or any(char.isupper() for char in word[1:])
        for word in compact_words
    )
