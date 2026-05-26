import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from kalorie2.transcript_model import (
    app,
    build_transcript_predictions,
    count_rule_matches,
    direct_probability_backtest_rows,
    fits_per_word_multiplier,
    transcript_contains_market_word,
)


def test_rule_matcher_uses_standalone_plural_possessive_and_hyphen_rules():
    text = "Apple's products include apples, but pineapple is different. New-York-based team."

    assert count_rule_matches("Apple", text) == 2
    assert count_rule_matches("New York", text) == 1
    assert count_rule_matches("AI", "artificial intelligence is not the acronym") == 0
    assert transcript_contains_market_word("Buyback / Repurchase", "The buybacks continued.")


def test_rule_matcher_honors_minimum_count_suffix():
    assert transcript_contains_market_word("Vega (5+ times)", "vega vega vega vega vega")
    assert not transcript_contains_market_word("Vega (5+ times)", "vega vega vega vega")


def test_fits_per_word_multiplier_with_global_fallback():
    observations = [
        {"word": "ai", "historical_rate": 0.25, "outcome": 1},
        {"word": "ai", "historical_rate": 0.25, "outcome": 0},
        {"word": "margin", "historical_rate": 0.50, "outcome": 1},
    ]

    multipliers = fits_per_word_multiplier(observations, min_word_observations=2, shrinkage=0)

    assert multipliers["global"]["multiplier"] == 2.0
    assert multipliers["words"]["ai"]["multiplier"] == 2.0
    assert multipliers["words"]["margin"]["multiplier"] == 2.0
    assert multipliers["words"]["margin"]["used_global_fallback"]


def test_build_transcript_predictions_uses_prior_transcripts_and_walk_forward_multipliers(
    tmp_path: Path,
):
    transcript_root = tmp_path / "earnings_call_transcripts"
    apple_dir = transcript_root / "Apple"
    apple_dir.mkdir(parents=True)
    (apple_dir / "2024_Q1_aapl_processed.txt").write_text("AI margin.", encoding="utf-8")
    (apple_dir / "2024_Q2_aapl_processed.txt").write_text("AI.", encoding="utf-8")
    (apple_dir / "2025_Q1_aapl_processed.txt").write_text("No target.", encoding="utf-8")

    rows = [
        {
            "market_ticker": "KXEARNINGSMENTIONAAPL-24JUL30-AI",
            "event_ticker": "KXEARNINGSMENTIONAAPL-24JUL30",
            "series_ticker": "KXEARNINGSMENTIONAAPL",
            "event_phrase": "What will Apple say during their next earnings call?",
            "word_said": "AI",
            "normalized_word_said": "ai",
            "close_time": "2024-07-30T20:00:00Z",
            "preclose_yes_bid": "0.20",
            "preclose_yes_ask": "0.30",
            "preclose_yes_mid": "0.25",
            "final_outcome": "yes",
        },
        {
            "market_ticker": "KXEARNINGSMENTIONAAPL-25JUL30-AI",
            "event_ticker": "KXEARNINGSMENTIONAAPL-25JUL30",
            "series_ticker": "KXEARNINGSMENTIONAAPL",
            "event_phrase": "What will Apple say during their next earnings call?",
            "word_said": "AI",
            "normalized_word_said": "ai",
            "close_time": "2025-07-30T20:00:00Z",
            "preclose_yes_bid": "0.20",
            "preclose_yes_ask": "0.30",
            "preclose_yes_mid": "0.25",
            "final_outcome": "no",
        },
    ]

    predicted = build_transcript_predictions(
        rows,
        transcript_root=transcript_root,
        min_word_observations=1,
        shrinkage=0,
    )

    assert predicted[0]["transcript_prior_count"] == "2"
    assert predicted[0]["transcript_hit_count"] == "2"
    assert predicted[0]["historical_transcript_rate"] == "1.000000"
    assert predicted[0]["transcript_model_probability"] == "1.000000"
    assert predicted[1]["transcript_prior_count"] == "3"
    assert predicted[1]["transcript_hit_count"] == "2"
    assert predicted[1]["word_multiplier_observations"] == "1"
    assert predicted[1]["transcript_model_probability"] == "0.666667"


def test_transcript_model_cli_writes_prediction_report(tmp_path: Path):
    transcript_root = tmp_path / "earnings_call_transcripts"
    apple_dir = transcript_root / "Apple"
    apple_dir.mkdir(parents=True)
    (apple_dir / "2024_Q1_aapl_processed.txt").write_text("AI.", encoding="utf-8")

    markets_csv = tmp_path / "markets.csv"
    with markets_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "market_ticker",
                "event_ticker",
                "series_ticker",
                "event_phrase",
                "word_said",
                "normalized_word_said",
                "close_time",
                "preclose_yes_bid",
                "preclose_yes_ask",
                "preclose_yes_mid",
                "final_outcome",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "market_ticker": "KXEARNINGSMENTIONAAPL-24JUL30-AI",
                "event_ticker": "KXEARNINGSMENTIONAAPL-24JUL30",
                "series_ticker": "KXEARNINGSMENTIONAAPL",
                "event_phrase": "What will Apple say during their next earnings call?",
                "word_said": "AI",
                "normalized_word_said": "ai",
                "close_time": "2024-07-30T20:00:00Z",
                "preclose_yes_bid": "0.20",
                "preclose_yes_ask": "0.30",
                "preclose_yes_mid": "0.25",
                "final_outcome": "yes",
            }
        )
    predictions_csv = tmp_path / "predictions.csv"
    report_json = tmp_path / "report.json"

    result = CliRunner().invoke(
        app,
        [
            str(markets_csv),
            "--transcript-root",
            str(transcript_root),
            "--out-csv",
            str(predictions_csv),
            "--report-json",
            str(report_json),
        ],
    )

    assert result.exit_code == 0
    assert "transcript_model_probability" in predictions_csv.read_text(encoding="utf-8")
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["rows"] == 1
    assert report["rows_with_transcript_history"] == 1


def test_direct_probability_backtest_trades_against_bid_ask():
    rows = [
        {
            "event_ticker": "EVENT1",
            "market_ticker": "EVENT1-A",
            "close_time": "2026-01-01T12:00:00Z",
            "transcript_model_probability": "0.80",
            "preclose_yes_bid": "0.40",
            "preclose_yes_ask": "0.50",
            "final_outcome": "yes",
        },
        {
            "event_ticker": "EVENT1",
            "market_ticker": "EVENT1-B",
            "close_time": "2026-01-01T12:00:00Z",
            "transcript_model_probability": "0.10",
            "preclose_yes_bid": "0.30",
            "preclose_yes_ask": "0.40",
            "final_outcome": "no",
        },
        {
            "event_ticker": "EVENT1",
            "market_ticker": "EVENT1-C",
            "close_time": "2026-01-01T12:00:00Z",
            "transcript_model_probability": "0.35",
            "preclose_yes_bid": "0.30",
            "preclose_yes_ask": "0.40",
            "final_outcome": "yes",
        },
    ]

    report = direct_probability_backtest_rows(
        rows,
        probability_column="transcript_model_probability",
    )

    assert report["summary"]["trades"] == 2
    assert report["summary"]["total_pnl"] == 0.8
    assert [trade["side"] for trade in report["trades"]] == ["YES", "NO"]
