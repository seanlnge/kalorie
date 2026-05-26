from kalorie.app.cli import (
    _kalshi_benchmark_detail_log,
    _kalshi_benchmark_diagnostics_markdown,
    _kalshi_benchmark_diagnostics_report,
    _kalshi_benchmark_metric_report,
    _kalshi_benchmark_skip_summary,
    _kalshi_benchmark_table_markdown,
)


def test_kalshi_benchmark_metric_report_compares_model_to_bid_ask_mid():
    rows = [
        {
            "event_id": "event-a",
            "label": 1,
            "model_probability": 0.8,
            "kalshi_yes_bid": 0.6,
            "kalshi_yes_ask": 0.7,
            "kalshi_yes_mid": 0.65,
        },
        {
            "event_id": "event-a",
            "label": 0,
            "model_probability": 0.2,
            "kalshi_yes_bid": 0.3,
            "kalshi_yes_ask": 0.4,
            "kalshi_yes_mid": 0.35,
        },
    ]

    report = _kalshi_benchmark_metric_report(rows)

    assert report["sample_count"] == 2
    assert report["model"]["brier_score"] == 0.04
    assert report["kalshi_yes_bid"]["brier_score"] == 0.125
    assert report["kalshi_yes_ask"]["brier_score"] == 0.125
    assert report["kalshi_yes_mid"]["brier_score"] == 0.1225
    assert report["deltas_vs_kalshi"]["model_minus_yes_mid_brier"] == -0.0825
    assert report["per_event"]["event-a"]["sample_count"] == 2


def test_kalshi_benchmark_table_markdown_shows_per_event_metrics():
    rows = [
        {
            "event_id": "event-a",
            "label": 1,
            "model_probability": 0.8,
            "kalshi_yes_bid": 0.6,
            "kalshi_yes_ask": 0.7,
            "kalshi_yes_mid": 0.65,
        },
        {
            "event_id": "event-a",
            "label": 0,
            "model_probability": 0.2,
            "kalshi_yes_bid": 0.3,
            "kalshi_yes_ask": 0.4,
            "kalshi_yes_mid": 0.35,
        },
    ]
    report = _kalshi_benchmark_metric_report(rows)

    table = _kalshi_benchmark_table_markdown(report)

    assert "| Event | Rows | Model Brier | Model ECE | Kalshi Mid Brier |" in table
    assert "| event-a | 2 | 0.040000 | 0.200000 | 0.122500 |" in table
    assert "Model minus Kalshi mid Brier" in table


def test_kalshi_benchmark_detail_log_lists_market_phrase_quotes_and_outcome():
    rows = [
        {
            "event_id": "event-a",
            "market_id": "KXEARNINGSMENTIONNVDA-26MAY20-BLACKWELL-EXTRA-LONG",
            "target_phrase": "blackwell platform supercycle phrase",
            "label": 1,
            "model_probability": 0.8,
            "kalshi_yes_bid": 0.6,
            "kalshi_yes_ask": 0.7,
            "kalshi_yes_mid": 0.65,
            "snapshot_target_time": "2026-05-20T20:50:00Z",
            "candle_end_ts": 1780000000,
        }
    ]

    log = _kalshi_benchmark_detail_log(rows)

    assert log.startswith("market_id")
    assert "\t" not in log
    header = log.splitlines()[0]
    assert header.index("ask") < header.index("bid")
    assert "mid" in log.splitlines()[0]
    assert "model" in log.splitlines()[0]
    data_line = log.splitlines()[1]
    assert "0.600" in data_line
    assert "0.700" in data_line
    assert "0.650" in data_line
    assert "0.800" in data_line
    assert "blackwell platform sup" in log
    assert "supercycle phrase" not in log
    assert "EXTRA-LONG" not in log
    assert "2026-05-20T20:50:00Z" in log


def test_kalshi_benchmark_diagnostics_report_flags_failure_modes():
    rows = [
        {
            "event_id": "event-a",
            "market_id": "macro-fp",
            "target_phrase": "tariff",
            "label": 0,
            "model_probability": 0.99,
            "kalshi_yes_bid": 0.2,
            "kalshi_yes_ask": 0.3,
            "kalshi_yes_mid": 0.25,
        },
        {
            "event_id": "event-a",
            "market_id": "codename-fn",
            "target_phrase": "nano banana",
            "label": 1,
            "model_probability": 0.01,
            "kalshi_yes_bid": 0.8,
            "kalshi_yes_ask": 0.9,
            "kalshi_yes_mid": 0.85,
        },
        {
            "event_id": "event-b",
            "market_id": "wide-spread",
            "target_phrase": "openai",
            "label": 1,
            "model_probability": 0.6,
            "kalshi_yes_bid": 0.0,
            "kalshi_yes_ask": 1.0,
            "kalshi_yes_mid": 0.5,
        },
    ]

    diagnostics = _kalshi_benchmark_diagnostics_report(rows)

    assert diagnostics["sample_count"] == 3
    assert diagnostics["false_positive_count"] == 1
    assert diagnostics["false_negative_count"] == 1
    assert diagnostics["probability_saturation"]["at_or_above_0_90_count"] == 1
    assert diagnostics["probability_saturation"]["at_or_below_0_10_count"] == 1
    assert diagnostics["quote_artifacts"]["wide_spread_count"] == 1
    assert diagnostics["phrase_categories"]["macro"]["sample_count"] == 1
    assert diagnostics["phrase_categories"]["codename_or_product"]["sample_count"] == 1
    assert diagnostics["worst_rows"][0]["market_id"] == "macro-fp"
    assert diagnostics["worst_rows"][0]["direction"] == "false_positive"

    markdown = _kalshi_benchmark_diagnostics_markdown(diagnostics)
    assert "Benchmark Diagnostics" in markdown
    assert "| macro-fp | event-a | tariff | macro | false_positive |" in markdown


def test_kalshi_benchmark_skip_summary_counts_missing_rows():
    summary = _kalshi_benchmark_skip_summary(
        total_examples=5,
        evaluated_rows=[
            {"market_id": "a"},
            {"market_id": "b"},
        ],
        missing_snapshot_count=2,
        missing_quote_count=1,
    )

    assert summary == {
        "total_examples": 5,
        "evaluated_rows": 2,
        "skipped_rows": 3,
        "missing_snapshot_count": 2,
        "missing_quote_count": 1,
    }
