import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from kalorie2.backtest import app, backtest_rows


def test_walk_forward_bin_strategy_trades_only_from_prior_events():
    rows = [
        {
            "event_ticker": "EVENT1",
            "market_ticker": "EVENT1-A",
            "close_time": "2026-01-01T12:00:00Z",
            "preclose_yes_bid": "0.09",
            "preclose_yes_ask": "0.11",
            "preclose_yes_mid": "0.10",
            "final_outcome": "no",
        },
        {
            "event_ticker": "EVENT1",
            "market_ticker": "EVENT1-B",
            "close_time": "2026-01-01T12:00:00Z",
            "preclose_yes_bid": "0.09",
            "preclose_yes_ask": "0.11",
            "preclose_yes_mid": "0.10",
            "final_outcome": "no",
        },
        {
            "event_ticker": "EVENT2",
            "market_ticker": "EVENT2-A",
            "close_time": "2026-01-02T12:00:00Z",
            "preclose_yes_bid": "0.09",
            "preclose_yes_ask": "0.11",
            "preclose_yes_mid": "0.10",
            "final_outcome": "no",
        },
    ]

    report = backtest_rows(rows, min_bin_count=2)

    assert report["summary"]["eligible_rows"] == 1
    assert report["summary"]["trades"] == 1
    assert report["summary"]["total_pnl"] == 0.09
    trade = report["trades"][0]
    assert trade["market_ticker"] == "EVENT2-A"
    assert trade["side"] == "NO"
    assert trade["calibrated_probability"] == 0.0
    assert trade["pnl"] == 0.09


def test_walk_forward_bin_strategy_can_buy_yes_when_prior_bin_rate_exceeds_ask():
    rows = [
        {
            "event_ticker": "EVENT1",
            "market_ticker": "EVENT1-A",
            "close_time": "2026-01-01T12:00:00Z",
            "preclose_yes_bid": "0.89",
            "preclose_yes_ask": "0.91",
            "preclose_yes_mid": "0.90",
            "final_outcome": "yes",
        },
        {
            "event_ticker": "EVENT1",
            "market_ticker": "EVENT1-B",
            "close_time": "2026-01-01T12:00:00Z",
            "preclose_yes_bid": "0.89",
            "preclose_yes_ask": "0.91",
            "preclose_yes_mid": "0.90",
            "final_outcome": "yes",
        },
        {
            "event_ticker": "EVENT2",
            "market_ticker": "EVENT2-A",
            "close_time": "2026-01-02T12:00:00Z",
            "preclose_yes_bid": "0.89",
            "preclose_yes_ask": "0.91",
            "preclose_yes_mid": "0.90",
            "final_outcome": "yes",
        },
    ]

    report = backtest_rows(rows, min_bin_count=2)

    assert report["summary"]["trades"] == 1
    trade = report["trades"][0]
    assert trade["side"] == "YES"
    assert trade["pnl"] == 0.09
    assert report["summary"]["roi_on_cost"] == 0.098901


def test_backtest_cli_writes_summary_and_trades(tmp_path: Path):
    input_csv = tmp_path / "markets.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_ticker",
                "market_ticker",
                "close_time",
                "preclose_yes_bid",
                "preclose_yes_ask",
                "preclose_yes_mid",
                "final_outcome",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "event_ticker": "EVENT1",
                    "market_ticker": "EVENT1-A",
                    "close_time": "2026-01-01T12:00:00Z",
                    "preclose_yes_bid": "0.09",
                    "preclose_yes_ask": "0.11",
                    "preclose_yes_mid": "0.10",
                    "final_outcome": "no",
                },
                {
                    "event_ticker": "EVENT1",
                    "market_ticker": "EVENT1-B",
                    "close_time": "2026-01-01T12:00:00Z",
                    "preclose_yes_bid": "0.09",
                    "preclose_yes_ask": "0.11",
                    "preclose_yes_mid": "0.10",
                    "final_outcome": "no",
                },
                {
                    "event_ticker": "EVENT2",
                    "market_ticker": "EVENT2-A",
                    "close_time": "2026-01-02T12:00:00Z",
                    "preclose_yes_bid": "0.09",
                    "preclose_yes_ask": "0.11",
                    "preclose_yes_mid": "0.10",
                    "final_outcome": "no",
                },
            ]
        )
    json_out = tmp_path / "backtest.json"
    trades_out = tmp_path / "trades.csv"

    result = CliRunner().invoke(
        app,
        [
            str(input_csv),
            "--json-out",
            str(json_out),
            "--trades-out",
            str(trades_out),
            "--min-bin-count",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(json_out.read_text(encoding="utf-8"))["summary"]["trades"] == 1
    assert "EVENT2-A" in trades_out.read_text(encoding="utf-8")
