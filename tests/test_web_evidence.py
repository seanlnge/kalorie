import json
from datetime import UTC, datetime

import pytest

from kalorie2.web_evidence import (
    WebEvidenceItem,
    WebEvidencePacket,
    build_openai_web_search_payload,
    build_web_evidence_prompt,
    parse_web_evidence_response,
)


def _packet_payload() -> dict:
    return {
        "event_ticker": "KXEARNINGSMENTIONDE-26MAY21",
        "company_name": "John Deere",
        "cutoff_time": "2026-05-21T07:41:50Z",
        "items": [
            {
                "title": "Deere discusses tariff pressure before earnings",
                "url": "https://example.com/deere-tariffs",
                "source": "Example News",
                "published_at": "2026-05-20T10:00:00Z",
                "snippet": "Deere faces tariff cost pressure ahead of its earnings call.",
                "target_phrases": ["tariff"],
                "evidence_strength": 0.8,
            },
            {
                "title": "Post-call transcript recap",
                "url": "https://example.com/post-call",
                "source": "Example News",
                "published_at": "2026-05-22T10:00:00Z",
                "snippet": "Management mentioned tariffs on the earnings call.",
                "target_phrases": ["tariff"],
                "evidence_strength": 1.0,
            },
            {
                "title": "Undated article",
                "url": "https://example.com/undated",
                "source": "Example News",
                "published_at": None,
                "snippet": "No date means no historical use.",
                "target_phrases": ["tariff"],
                "evidence_strength": 0.6,
            },
        ],
    }


def test_parse_web_evidence_response_filters_to_cutoff_safe_sources():
    packet = parse_web_evidence_response(json.dumps(_packet_payload()))

    retained = packet.cutoff_safe_items()

    assert packet.event_ticker == "KXEARNINGSMENTIONDE-26MAY21"
    assert len(retained) == 1
    assert retained[0].url == "https://example.com/deere-tariffs"
    assert retained[0].published_at == datetime(2026, 5, 20, 10, tzinfo=UTC)


def test_web_evidence_packet_normalizes_naive_datetimes_to_utc():
    payload = _packet_payload()
    payload["items"][0]["published_at"] = "2026-05-20T10:00:00"

    packet = parse_web_evidence_response(json.dumps(payload))

    retained = packet.cutoff_safe_items()

    assert len(retained) == 1
    assert retained[0].published_at == datetime(2026, 5, 20, 10, tzinfo=UTC)


def test_web_evidence_packet_treats_partial_dates_as_undated():
    payload = _packet_payload()
    payload["items"][0]["published_at"] = "2026-05"

    packet = parse_web_evidence_response(json.dumps(payload))

    retained = packet.cutoff_safe_items()

    assert len(retained) == 0
    assert packet.items[0].published_at is None


def test_web_evidence_packet_treats_placeholder_dates_as_undated():
    payload = _packet_payload()
    payload["items"][0]["published_at"] = "2026-05-??"

    packet = parse_web_evidence_response(json.dumps(payload))

    retained = packet.cutoff_safe_items()

    assert len(retained) == 0
    assert packet.items[0].published_at is None


def test_web_evidence_packet_treats_alpha_placeholder_dates_as_undated():
    payload = _packet_payload()
    payload["items"][0]["published_at"] = "2026-03-xx"

    packet = parse_web_evidence_response(json.dumps(payload))

    retained = packet.cutoff_safe_items()

    assert len(retained) == 0
    assert packet.items[0].published_at is None


def test_web_evidence_packet_treats_zero_dates_as_undated():
    payload = _packet_payload()
    payload["items"][0]["published_at"] = "2026-01-00T00:00:00+00:00"
    payload["items"][1]["published_at"] = "2025-00-00T00:00:00+00:00"

    packet = parse_web_evidence_response(json.dumps(payload))

    retained = packet.cutoff_safe_items()

    assert len(retained) == 0
    assert packet.items[0].published_at is None
    assert packet.items[1].published_at is None


def test_web_evidence_packet_features_use_only_retained_sources():
    packet = WebEvidencePacket.model_validate(_packet_payload())

    features = packet.features_for_target("tariff")

    assert features["web_evidence_available"] == 1.0
    assert features["web_evidence_item_count"] == 1.0
    assert features["web_evidence_target_overlap"] == 1.0
    assert features["web_evidence_strength_max"] == 0.8


def test_web_evidence_packet_features_include_richer_target_signal():
    payload = _packet_payload()
    payload["items"].append(
        {
            "title": "Deere files 10-Q before earnings",
            "url": "https://www.sec.gov/Archives/edgar/data/deere",
            "source": "SEC",
            "published_at": "2026-05-21T05:41:50Z",
            "snippet": "The filing discusses tariff pressure and manufacturing costs.",
            "target_phrases": ["tariff", "manufacturing"],
            "evidence_strength": 0.4,
        }
    )
    payload["items"].append(
        {
            "title": "Company release on demand",
            "url": "https://investor.deere.com/news-releases",
            "source": "John Deere Investor Relations",
            "published_at": "2026-05-21T06:41:50Z",
            "snippet": "The company previews demand trends before earnings.",
            "target_phrases": ["demand"],
            "evidence_strength": 0.9,
        }
    )
    packet = WebEvidencePacket.model_validate(payload)

    features = packet.features_for_target("tariff")

    assert features["web_evidence_cutoff_safe_count"] == 3.0
    assert features["web_evidence_target_match_count"] == 2.0
    assert features["web_evidence_target_match_share"] == 2 / 3
    assert features["web_evidence_strength_mean"] == pytest.approx(0.6)
    assert features["web_evidence_strength_sum"] == pytest.approx(1.2)
    assert features["web_evidence_recency_min_hours"] == 2.0
    assert features["web_evidence_recency_mean_hours"] == pytest.approx(11.848611)
    assert features["web_evidence_source_sec"] == 1.0
    assert features["web_evidence_source_news"] == 1.0
    assert features["web_evidence_source_company"] == 0.0


def test_web_evidence_packet_features_separate_relevant_search_results():
    payload = _packet_payload()
    payload["items"][0]["relevance_score"] = 0.95
    payload["items"][0]["evidence_direction"] = "support"
    payload["items"].append(
        {
            "title": "Generic retail article",
            "url": "https://example.com/generic-retail",
            "source": "Example News",
            "published_at": "2026-05-20T11:00:00Z",
            "snippet": "A broad article about retail stocks with no tariff context.",
            "target_phrases": ["tariff"],
            "evidence_strength": 0.9,
            "relevance_score": 0.2,
            "evidence_direction": "neutral",
        }
    )
    packet = WebEvidencePacket.model_validate(payload)

    features = packet.features_for_target("tariff")

    assert features["web_evidence_target_match_count"] == 1.0
    assert features["web_evidence_high_relevance_count"] == 1.0
    assert features["web_evidence_relevance_mean"] == 0.95
    assert features["web_evidence_support_count"] == 1.0
    assert features["web_evidence_neutral_count"] == 0.0
    assert features["web_evidence_strength_sum"] == 0.8


def test_web_evidence_packet_features_can_use_row_level_cutoff():
    payload = _packet_payload()
    payload["cutoff_time"] = "2026-05-21T12:00:00Z"
    payload["items"][0]["published_at"] = "2026-05-21T09:00:00Z"
    packet = WebEvidencePacket.model_validate(payload)

    features = packet.features_for_target(
        "tariff",
        cutoff_time=datetime(2026, 5, 21, 7, 41, 50, tzinfo=UTC),
    )

    assert features["web_evidence_available"] == 0.0
    assert features["web_evidence_cutoff_safe_count"] == 0.0


def test_build_web_evidence_prompt_includes_cutoff_and_warns_against_leakage():
    prompt = build_web_evidence_prompt(
        event={
            "event_ticker": "KXEARNINGSMENTIONDE-26MAY21",
            "company_name": "John Deere",
            "cutoff_time": "2026-05-21T07:41:50Z",
        },
        target_phrases=["tariff", "manufacturing"],
    )

    assert "2026-05-21T07:41:50Z" in prompt
    assert "published before or at the cutoff" in prompt
    assert "Do not use earnings-call transcripts" in prompt
    assert "relevance_score" in prompt
    assert "Only include sources worth using as forecasting evidence" in prompt
    assert "tariff" in prompt


def test_build_openai_web_search_payload_uses_responses_web_search_and_strict_schema():
    payload = build_openai_web_search_payload(
        prompt="Find cutoff-safe Deere evidence.",
        model="gpt-5.5",
    )

    assert payload["model"] == "gpt-5.5"
    assert payload["tools"] == [{"type": "web_search"}]
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"]["additionalProperties"] is False
    item_schema = payload["text"]["format"]["schema"]["properties"]["items"]["items"]
    assert "relevance_score" in item_schema["properties"]
    assert "evidence_direction" in item_schema["properties"]


def test_web_evidence_item_rejects_strength_outside_probability_range():
    bad_payload = {
        "title": "Bad",
        "url": "https://example.com/bad",
        "source": "Example",
        "published_at": "2026-05-20T10:00:00Z",
        "snippet": "Bad strength",
        "target_phrases": ["tariff"],
        "evidence_strength": 1.5,
    }

    try:
        WebEvidenceItem.model_validate(bad_payload)
    except ValueError as exc:
        assert "less than or equal to 1" in str(exc)
    else:
        raise AssertionError("expected validation error")
