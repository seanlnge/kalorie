import json

import pytest
from pydantic import ValidationError

from kalorie2.event_scenarios import (
    EventScenarioCatalog,
    build_event_scenario_prompt,
    parse_event_scenario_response,
    reject_transcript_like_output,
)


def _valid_catalog_payload() -> dict:
    return {
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
                "unlikely_contexts": ["No evidence points to tariffs as a new product name."],
                "evidence_rationale": "The supplied snippets mention tariffs and cost pressure.",
            }
        ],
    }


def test_parse_event_scenario_response_accepts_strict_json():
    payload = _valid_catalog_payload()

    catalog = parse_event_scenario_response(json.dumps(payload))

    assert catalog.event_ticker == "KXEARNINGSMENTIONDE-26MAY21"
    assert catalog.target_phrase_contexts[0].target_phrase == "tariff"
    assert catalog.scenario_texts()[0] == "Tariffs affecting equipment costs"


def test_parse_event_scenario_response_rejects_transcript_like_outputs():
    payload = _valid_catalog_payload()
    payload["management_language_patterns"] = [
        "Operator: Good morning and welcome to the Deere earnings call."
    ]

    with pytest.raises(ValueError, match="transcript"):
        parse_event_scenario_response(json.dumps(payload))


def test_parse_event_scenario_response_rejects_wrapped_json():
    payload = json.dumps(_valid_catalog_payload())

    with pytest.raises(ValueError, match="strict JSON"):
        parse_event_scenario_response(f"```json\n{payload}\n```")

    with pytest.raises(ValueError, match="strict JSON"):
        parse_event_scenario_response(f"Here is the catalog:\n{payload}")


def test_event_scenario_catalog_rejects_missing_and_extra_keys():
    payload = _valid_catalog_payload()
    payload.pop("source_rationales")

    with pytest.raises(ValidationError):
        EventScenarioCatalog.model_validate(payload)

    payload = _valid_catalog_payload()
    payload["unexpected"] = "not allowed"

    with pytest.raises(ValidationError):
        EventScenarioCatalog.model_validate(payload)


def test_reject_transcript_like_output_checks_target_contexts():
    payload = _valid_catalog_payload()
    payload["target_phrase_contexts"][0]["likely_contexts"] = [
        "Analyst: Can you discuss tariff exposure?"
    ]
    catalog = EventScenarioCatalog.model_validate(payload)

    with pytest.raises(ValueError, match="transcript"):
        reject_transcript_like_output(catalog)


def test_build_event_scenario_prompt_is_grounded_and_schema_explicit():
    prompt = build_event_scenario_prompt(
        event={
            "event_ticker": "KXEARNINGSMENTIONDE-26MAY21",
            "company_name": "John Deere",
            "snapshot_target_time": "2026-05-21T07:41:50Z",
        },
        target_phrases=["tariff", "manufacturing"],
        evidence_snippets=[
            {
                "source_id": "earnings-release",
                "text": "Management cited tariff cost pressure.",
            }
        ],
    )

    assert "Output strict JSON" in prompt
    assert "Do not use transcript text" in prompt
    assert "target_phrase_contexts" in prompt
    assert "tariff" in prompt
    assert "earnings-release" in prompt
