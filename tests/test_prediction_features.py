from datetime import UTC, datetime
from decimal import Decimal

import pytest

from kalorie2.event_scenarios import EventScenarioCatalog
from kalorie2.prediction_features import (
    build_feature_matrix,
    build_feature_row,
    extract_market_features,
    extract_phrase_features,
    extract_resolution_features,
    extract_scenario_features,
    extract_web_evidence_features,
)
from kalorie2.prediction_types import PredictionInputRow
from kalorie2.web_evidence import WebEvidencePacket


def _row(**overrides) -> PredictionInputRow:
    payload = {
        "market_ticker": "KXEARNINGSMENTIONDE-26MAY21-TARI",
        "event_ticker": "KXEARNINGSMENTIONDE-26MAY21",
        "series_ticker": "KXEARNINGSMENTIONDE",
        "market_category": "earnings",
        "event_phrase": "What will John Deere say during their next earnings call?",
        "market_name": "What will John Deere say during their next earnings call? - Tariff",
        "word_said": "Tariff",
        "normalized_word_said": "tariff",
        "final_outcome": "yes",
        "status": None,
        "close_time": datetime(2026, 5, 21, 15, 41, 50, tzinfo=UTC),
        "snapshot_target_time": datetime(2026, 5, 21, 7, 41, 50, tzinfo=UTC),
        "preclose_yes_bid": Decimal("0.95"),
        "preclose_yes_ask": Decimal("0.97"),
        "preclose_yes_mid": Decimal("0.96"),
        "candle_end_ts": 1779349020,
        "snapshot_staleness_seconds": 290,
        "preclose_volume": 1200,
        "preclose_open_interest": 300,
        "preclose_yes_bid_size": 50,
        "preclose_yes_ask_size": 70,
        "settlement_ts": None,
        "source": "kalshi_search_series",
        "company_prior_call_count": 2,
        "company_avg_call_duration_minutes_prior": 60.0,
        "company_avg_qa_question_count_prior": 10.0,
        "company_avg_prepared_remarks_minutes_prior": 24.0,
        "company_qa_share_prior": 0.6,
        "company_question_count_trend_prior": 2.0,
        "company_transcript_coverage_count": 2,
        "company_transcript_style_available": 1,
        "company_avg_transcript_word_count_prior": 1250.0,
        "company_avg_phrase_mentions_prior": 1.5,
    }
    payload.update(overrides)
    return PredictionInputRow.model_validate(payload)


def _catalog() -> EventScenarioCatalog:
    return EventScenarioCatalog.model_validate(
        {
            "event_ticker": "KXEARNINGSMENTIONDE-26MAY21",
            "company_name": "John Deere",
            "topics": ["Tariffs affecting equipment costs", "Agriculture demand by region"],
            "analyst_questions": ["How are tariffs affecting margins?"],
            "management_language_patterns": ["Management may frame tariffs as a cost headwind."],
            "source_rationales": ["Latest earnings release discusses tariff exposure."],
            "target_phrase_contexts": [
                {
                    "target_phrase": "tariff",
                    "likely_contexts": [
                        "Tariff costs are likely to be discussed in margin commentary."
                    ],
                    "unlikely_contexts": ["No evidence points to tariff as a product name."],
                    "evidence_rationale": "The snippets mention tariffs and cost pressure.",
                }
            ],
        }
    )


def _web_packet() -> WebEvidencePacket:
    return WebEvidencePacket.model_validate(
        {
            "event_ticker": "KXEARNINGSMENTIONDE-26MAY21",
            "company_name": "John Deere",
            "cutoff_time": "2026-05-21T07:41:50Z",
            "items": [
                {
                    "title": "Tariff pressure before earnings",
                    "url": "https://example.com/tariff",
                    "source": "Example News",
                    "published_at": "2026-05-20T10:00:00Z",
                    "snippet": "Deere faces tariff cost pressure before earnings.",
                    "target_phrases": ["tariff"],
                    "evidence_strength": 0.75,
                }
            ],
        }
    )


def test_extract_market_features_includes_logit_spread_and_staleness():
    features = extract_market_features(_row())

    assert features["market_yes_bid"] == 0.95
    assert features["market_yes_ask"] == 0.97
    assert features["market_yes_mid"] == 0.96
    assert features["market_spread"] == pytest.approx(0.02)
    assert features["market_mid_logit"] == pytest.approx(3.178054, abs=0.000001)
    assert features["snapshot_staleness_hours"] == pytest.approx(290 / 3600)
    assert features["hours_before_close"] == 8.0
    assert features["log_hours_before_close"] == pytest.approx(2.197224, abs=0.000001)
    assert features["hours_before_close_bucket_6_12"] == 1.0
    assert features["hours_before_close_bucket_2_6"] == 0.0
    assert features["hours_before_close_bucket_12_24"] == 0.0
    assert features["hours_before_close_bucket_24_48"] == 0.0
    assert features["company_prior_call_count"] == 2.0
    assert features["company_avg_call_duration_minutes_prior"] == 60.0
    assert features["company_avg_qa_question_count_prior"] == 10.0
    assert features["company_avg_prepared_remarks_minutes_prior"] == 24.0
    assert features["company_qa_share_prior"] == 0.6
    assert features["company_question_count_trend_prior"] == 2.0
    assert features["company_transcript_coverage_count"] == 2.0
    assert features["company_transcript_style_available"] == 1.0
    assert features["company_transcript_style_missing"] == 0.0
    assert features["company_avg_transcript_word_count_prior"] == 1250.0
    assert features["company_avg_phrase_mentions_prior"] == 1.5
    assert features["market_bid_present"] == 1.0
    assert features["market_ask_present"] == 1.0
    assert features["market_spread_share_of_mid"] == pytest.approx(0.02 / 0.96)
    assert features["market_no_bid"] == pytest.approx(0.03)
    assert features["market_no_ask"] == pytest.approx(0.05)
    assert features["market_preclose_volume"] == 1200.0
    assert features["market_preclose_volume_present"] == 1.0
    assert features["market_preclose_open_interest"] == 300.0
    assert features["market_preclose_open_interest_present"] == 1.0
    assert features["market_preclose_volume_log"] == pytest.approx(7.09091, abs=0.00001)
    assert features["market_preclose_open_interest_log"] == pytest.approx(5.70711, abs=0.00001)
    assert features["market_preclose_yes_bid_size"] == 50.0
    assert features["market_preclose_yes_ask_size"] == 70.0
    assert features["market_preclose_size_imbalance"] == pytest.approx((50 - 70) / 120)
    assert "final_outcome" not in features


def test_extract_phrase_features_classifies_count_slash_macro_entity_and_generic_terms():
    count_features = extract_phrase_features(
        _row(word_said="Vega (5+ times)", normalized_word_said="vega (5+ times)")
    )
    slash_features = extract_phrase_features(
        _row(word_said="Oil / Gas / Gasoline", normalized_word_said="oil / gas / gasoline")
    )
    entity_features = extract_phrase_features(
        _row(word_said="BlackRock", normalized_word_said="blackrock")
    )
    generic_features = extract_phrase_features(
        _row(word_said="Revenue Growth", normalized_word_said="revenue growth")
    )

    assert count_features["phrase_has_count_threshold"] == 1.0
    assert count_features["phrase_count_threshold"] == 5.0
    assert slash_features["phrase_has_slash_alternatives"] == 1.0
    assert slash_features["phrase_option_count"] == 3.0
    assert slash_features["phrase_is_macro"] == 1.0
    assert entity_features["phrase_is_entity_like"] == 1.0
    assert generic_features["phrase_is_generic_business"] == 1.0
    assert generic_features["phrase_is_multiword"] == 1.0


def test_extract_phrase_features_adds_semantic_bucket_scores():
    tariff_features = extract_phrase_features(
        _row(word_said="Tariff", normalized_word_said="tariff")
    )
    membership_features = extract_phrase_features(
        _row(word_said="Membership Fee", normalized_word_said="membership fee")
    )
    ai_features = extract_phrase_features(_row(word_said="AI", normalized_word_said="ai"))

    assert tariff_features["phrase_semantic_macro_score"] > 0.75
    assert tariff_features["phrase_semantic_regulatory_score"] > 0.75
    assert tariff_features["phrase_semantic_embedding_available"] == 1.0
    assert tariff_features["phrase_semantic_axis_risk_opportunity"] > 0.0
    assert tariff_features["phrase_semantic_operations_score"] < 0.5
    assert membership_features["phrase_semantic_operations_score"] > 0.5
    assert membership_features["phrase_semantic_macro_score"] < 0.5
    assert ai_features["phrase_semantic_technology_score"] > 0.75


def test_extract_resolution_features_learns_variation_complexity_without_multiplier():
    features = extract_resolution_features(
        _row(
            market_name=(
                "What will Costco say during their next earnings call? - "
                "Membership fee / subscription fees (2+ times)"
            ),
            word_said="Membership fee / subscription fees (2+ times)",
            normalized_word_said="membership fee / subscription fees (2+ times)",
        )
    )

    assert features["resolution_option_count"] == 2.0
    assert features["resolution_requires_count_threshold"] == 1.0
    assert features["resolution_minimum_count"] == 2.0
    assert features["resolution_phrase_breadth_score"] > 0.0
    assert "resolution_probability_multiplier" not in features


def test_extract_scenario_features_scores_target_and_topic_overlap():
    features = extract_scenario_features(_row(), _catalog())

    assert features["scenario_available"] == 1.0
    assert features["scenario_topic_overlap_max"] > 0
    assert features["scenario_target_context_available"] == 1.0
    assert features["scenario_target_context_overlap_max"] > 0
    assert features["scenario_text_count"] > 0


def test_extract_web_evidence_features_uses_cutoff_safe_packet():
    features = extract_web_evidence_features(_row(), _web_packet())

    assert features["web_evidence_available"] == 1.0
    assert features["web_evidence_item_count"] == 1.0
    assert features["web_evidence_target_overlap"] == 1.0
    assert features["web_evidence_strength_max"] == 0.75


def test_build_feature_row_and_matrix_exclude_label_fields():
    row = _row()

    feature_row = build_feature_row(row, catalog=_catalog(), web_evidence=_web_packet())
    matrix = build_feature_matrix(
        [row],
        {row.event_ticker: _catalog()},
        {row.event_ticker: _web_packet()},
    )

    assert matrix == [feature_row]
    assert "final_outcome" not in feature_row
    assert "outcome_label" not in feature_row
    assert "settlement_ts" not in feature_row
    assert feature_row["market_yes_mid"] == 0.96
    assert feature_row["phrase_is_macro"] == 1.0
    assert feature_row["phrase_semantic_macro_score"] > 0.75
    assert feature_row["resolution_option_count"] == 1.0
    assert feature_row["scenario_available"] == 1.0
    assert feature_row["web_evidence_available"] == 1.0
    assert feature_row["company_metadata_available"] == 1.0
    assert feature_row["company_sector_industrials"] == 1.0
    assert feature_row["company_market_cap_large"] == 1.0
