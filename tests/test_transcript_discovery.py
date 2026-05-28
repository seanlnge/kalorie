import json

from kalorie2.transcript_discovery import (
    build_openai_transcript_discovery_payload,
    build_transcript_discovery_prompt,
    parse_transcript_discovery_response,
)


def test_build_transcript_discovery_prompt_asks_web_search_for_source_candidates():
    prompt = build_transcript_discovery_prompt(
        company_name="Costco",
        ticker="COST",
        fiscal_periods=["2025 Q1", "2025 Q2"],
    )

    assert "Costco" in prompt
    assert "COST" in prompt
    assert "2025 Q1" in prompt
    assert "Do not rely on memory" in prompt
    assert "source_url" in prompt
    assert "transcript_candidate" in prompt
    assert "call_duration_minutes" in prompt
    assert "qa_question_count" in prompt


def test_parse_transcript_discovery_response_keeps_structured_source_candidates():
    packet = parse_transcript_discovery_response(
        json.dumps(
            {
                "company_name": "Costco",
                "ticker": "COST",
                "candidates": [
                    {
                        "fiscal_period": "2025 Q1",
                        "source_url": "https://example.com/cost-q1-transcript",
                        "source_name": "Example Transcript Source",
                        "published_at": "2025-03-07T12:00:00Z",
                        "call_date": "2025-03-06T21:30:00Z",
                        "call_duration_minutes": 61.0,
                        "qa_question_count": 14,
                        "prepared_remarks_minutes": 27.0,
                        "transcript_candidate": True,
                        "confidence": 0.9,
                        "rationale": (
                            "The page title and snippet identify a Costco Q1 earnings "
                            "call transcript."
                        ),
                    },
                    {
                        "fiscal_period": "2025 Q2",
                        "source_url": "https://example.com/cost-q2-preview",
                        "source_name": "Example Preview",
                        "published_at": None,
                        "transcript_candidate": False,
                        "confidence": 0.2,
                        "rationale": "This is an earnings preview, not a transcript.",
                    },
                ],
            }
        )
    )

    assert packet.company_name == "Costco"
    assert packet.candidates[0].source_url == "https://example.com/cost-q1-transcript"
    assert packet.candidates[0].call_duration_minutes == 61.0
    assert packet.candidates[0].qa_question_count == 14
    assert packet.candidates[0].transcript_candidate
    assert packet.transcript_candidates()[0].fiscal_period == "2025 Q1"


def test_build_openai_transcript_discovery_payload_uses_web_search_schema():
    payload = build_openai_transcript_discovery_payload(
        prompt="Find Costco transcripts.",
        model="gpt-5.4-mini",
    )

    assert payload["model"] == "gpt-5.4-mini"
    assert payload["tools"] == [{"type": "web_search"}]
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"]["additionalProperties"] is False
    candidate_schema = payload["text"]["format"]["schema"]["properties"]["candidates"]["items"]
    assert "call_duration_minutes" in candidate_schema["properties"]
    assert "qa_question_count" in candidate_schema["properties"]
