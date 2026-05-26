import json
from decimal import Decimal

import pytest

from kalorie2.mixmcp import (
    MixMcpEventPacket,
    apply_mixmcp_to_predictions,
    build_mixmcp_prompt,
    mix_probability,
    parse_mixmcp_response,
)
from kalorie2.prediction_types import PredictionInputRow
from kalorie2.residual_engine import ResidualPrediction


def _row(
    *,
    event_ticker: str,
    market_ticker: str,
    final_outcome: str = "yes",
) -> PredictionInputRow:
    return PredictionInputRow.model_validate(
        {
            "market_ticker": market_ticker,
            "event_ticker": event_ticker,
            "series_ticker": "KXEARNINGSMENTIONDE",
            "market_category": "earnings",
            "event_phrase": "What will John Deere say during their next earnings call?",
            "market_name": "What will John Deere say during their next earnings call? - Tariff",
            "word_said": "Tariff",
            "normalized_word_said": "tariff",
            "final_outcome": final_outcome,
            "status": None,
            "close_time": "2026-01-01T20:00:00Z",
            "snapshot_target_time": "2026-01-01T12:00:00Z",
            "preclose_yes_bid": "0.49",
            "preclose_yes_ask": "0.51",
            "preclose_yes_mid": "0.50",
            "candle_end_ts": "1767279600",
            "snapshot_staleness_seconds": "0",
            "settlement_ts": None,
            "source": "unit_test",
        }
    )


def _prediction(
    *,
    event_ticker: str,
    market_ticker: str,
    probability: str,
    training_events: list[str],
) -> ResidualPrediction:
    return ResidualPrediction(
        market_ticker=market_ticker,
        event_ticker=event_ticker,
        probability=Decimal(probability),
        market_probability=Decimal("0.50"),
        residual_delta=0.0,
        training_event_tickers=training_events,
    )


def test_parse_mixmcp_response_requires_strict_probability_packet():
    packet = parse_mixmcp_response(
        json.dumps(
            {
                "event_ticker": "EVENT1",
                "cutoff_time": "2026-01-01T12:00:00Z",
                "model": "gpt-5.4-mini",
                "targets": [
                    {
                        "market_ticker": "EVENT1-TARI",
                        "target_phrase": "tariff",
                        "market_probability": 0.5,
                        "llm_probability": 0.7,
                        "confidence": 0.8,
                        "rationale": "Tariffs were central in pre-call evidence.",
                    }
                ],
            }
        )
    )

    assert packet.event_ticker == "EVENT1"
    assert packet.targets[0].llm_probability == 0.7


def test_mix_probability_uses_logit_space_alpha():
    assert mix_probability(0.5, 0.8, alpha=1.0) == pytest.approx(0.5)
    assert mix_probability(0.5, 0.8, alpha=0.0) == pytest.approx(0.8)
    assert mix_probability(0.5, 0.8, alpha=0.5) == pytest.approx(2 / 3)


def test_apply_mixmcp_learns_alpha_from_prior_events_only():
    rows = [
        _row(event_ticker="EVENT1", market_ticker="EVENT1-TARI", final_outcome="yes"),
        _row(event_ticker="EVENT2", market_ticker="EVENT2-TARI", final_outcome="no"),
    ]
    predictions = [
        _prediction(
            event_ticker="EVENT1",
            market_ticker="EVENT1-TARI",
            probability="0.40",
            training_events=[],
        ),
        _prediction(
            event_ticker="EVENT2",
            market_ticker="EVENT2-TARI",
            probability="0.40",
            training_events=["EVENT1"],
        ),
    ]
    packets = {
        "EVENT1": MixMcpEventPacket.model_validate(
            {
                "event_ticker": "EVENT1",
                "cutoff_time": "2026-01-01T12:00:00Z",
                "model": "gpt-5.4-mini",
                "targets": [
                    {
                        "market_ticker": "EVENT1-TARI",
                        "target_phrase": "tariff",
                        "market_probability": 0.4,
                        "llm_probability": 0.9,
                        "confidence": 1.0,
                        "rationale": "prior event",
                    }
                ],
            }
        ),
        "EVENT2": MixMcpEventPacket.model_validate(
            {
                "event_ticker": "EVENT2",
                "cutoff_time": "2026-01-02T12:00:00Z",
                "model": "gpt-5.4-mini",
                "targets": [
                    {
                        "market_ticker": "EVENT2-TARI",
                        "target_phrase": "tariff",
                        "market_probability": 0.4,
                        "llm_probability": 0.9,
                        "confidence": 1.0,
                        "rationale": "current event must not train alpha",
                    }
                ],
            }
        ),
    }

    mixed = apply_mixmcp_to_predictions(
        rows,
        predictions,
        packets,
        alpha_grid=[0.0, 1.0],
        alpha_mode="global",
    )

    assert float(mixed[0].probability) == 0.4
    assert float(mixed[1].probability) == 0.9
    assert "mixmcp_alpha:1.000" in mixed[0].reasons
    assert "mixmcp_alpha:0.000" in mixed[1].reasons


def test_build_mixmcp_prompt_includes_market_prior_and_mini_model_payload():
    prompt = build_mixmcp_prompt(
        event={"event_ticker": "EVENT1", "cutoff_time": "2026-01-01T12:00:00Z"},
        targets=[
            {
                "market_ticker": "EVENT1-TARI",
                "target_phrase": "tariff",
                "market_probability": 0.5,
            }
        ],
    )

    assert "market_probability" in prompt
    assert "EVENT1-TARI" in prompt
