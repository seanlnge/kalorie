import json

from kalorie2.model_card_cli import _latest_training_overlap_summary
from kalorie2.model_cards import (
    ConfidenceInterval,
    EvaluationRow,
    EvaluationSplit,
    MetricValue,
    ModelCard,
    build_evaluation_split,
    build_model_card_schema,
    latest_event_rows,
    parse_iso_utc,
)
from kalorie2.risk_presets import get_risk_preset
from kalorie2.risk_trials import build_risk_preset_trials


def test_model_card_schema_requires_latest30_test_metrics() -> None:
    card = ModelCard(
        model_name="kalorie-v3",
        model_version=3,
        model_type="market_anchored_linear_residual",
        training_data={
            "row_count": 3500,
            "event_count": 264,
            "source": "training/mention-markets-historical-20260523.csv",
        },
        feature_set={
            "feature_count": 57,
            "nonzero_weight_count": 49,
            "ablation_group": "resolution",
            "dropped_feature_prefixes": ["resolution_"],
        },
        evaluation_splits=[
            EvaluationSplit(
                name="latest30",
                role="test",
                event_count=30,
                market_count=380,
                metrics={
                    "brier": MetricValue(
                        value=0.162254,
                        ci95=ConfidenceInterval(low=0.13, high=0.19),
                    ),
                    "ece": MetricValue(
                        value=0.05507,
                        ci95=ConfidenceInterval(low=0.04, high=0.11),
                    ),
                    "log_loss": MetricValue(
                        value=0.5,
                        ci95=ConfidenceInterval(low=0.4, high=0.7),
                    ),
                },
            )
        ],
        caveats=["Latest-30 is the primary held-out test split."],
    )

    latest30 = card.primary_test_split

    assert latest30.name == "latest30"
    assert latest30.metrics["log_loss"].ci95 is not None


def test_build_model_card_schema_exports_json_schema() -> None:
    schema = build_model_card_schema()

    assert schema["title"] == "ModelCard"
    assert "evaluation_splits" in schema["properties"]
    assert "default_execution_policy" not in schema["properties"]
    assert "default_margin" not in schema["properties"]
    assert "risk_preset_trials" not in schema["properties"]
    assert json.dumps(schema)


def test_latest_training_overlap_summary_exposes_in_sample_latest_rows() -> None:
    raw_training_rows = [
        {"event_ticker": "E1", "market_ticker": "E1-TARI"},
        {"event_ticker": "E2", "market_ticker": "E2-TARI"},
    ]
    latest_rows = [
        EvaluationRow(
            event_ticker="E2",
            close_time=parse_iso_utc("2026-01-02T00:00:00Z"),
            outcome_label=0,
            market_probability=0.5,
            model_probability=0.4,
            yes_bid=0.49,
            yes_ask=0.51,
        ),
        EvaluationRow(
            event_ticker="E3",
            close_time=parse_iso_utc("2026-01-03T00:00:00Z"),
            outcome_label=1,
            market_probability=0.5,
            model_probability=0.6,
            yes_bid=0.49,
            yes_ask=0.51,
        ),
    ]

    summary = _latest_training_overlap_summary(raw_training_rows, latest_rows)

    assert summary == {
        "primary_test_source": "saved_runtime_scored_training_csv",
        "primary_test_training_overlap_event_count": 1,
        "primary_test_training_overlap_market_count": 1,
    }


def test_latest_event_rows_keeps_recent_events_only() -> None:
    rows = [
        EvaluationRow(
            event_ticker="E1",
            close_time=parse_iso_utc("2026-01-01T00:00:00Z"),
            outcome_label=1,
            market_probability=0.5,
            model_probability=0.6,
            yes_bid=0.49,
            yes_ask=0.51,
        ),
        EvaluationRow(
            event_ticker="E2",
            close_time=parse_iso_utc("2026-01-02T00:00:00Z"),
            outcome_label=0,
            market_probability=0.4,
            model_probability=0.3,
            yes_bid=0.39,
            yes_ask=0.41,
        ),
        EvaluationRow(
            event_ticker="E3",
            close_time=parse_iso_utc("2026-01-03T00:00:00Z"),
            outcome_label=1,
            market_probability=0.7,
            model_probability=0.8,
            yes_bid=0.69,
            yes_ask=0.71,
        ),
    ]

    latest = latest_event_rows(rows, event_count=2)

    assert {row.event_ticker for row in latest} == {"E2", "E3"}


def test_build_evaluation_split_includes_latest30_metrics_with_ci() -> None:
    rows = [
        EvaluationRow(
            event_ticker="E1",
            close_time=parse_iso_utc("2026-01-01T00:00:00Z"),
            outcome_label=1,
            market_probability=0.6,
            model_probability=0.7,
            yes_bid=0.59,
            yes_ask=0.61,
        ),
        EvaluationRow(
            event_ticker="E1",
            close_time=parse_iso_utc("2026-01-01T00:00:00Z"),
            outcome_label=0,
            market_probability=0.4,
            model_probability=0.3,
            yes_bid=0.39,
            yes_ask=0.41,
        ),
        EvaluationRow(
            event_ticker="E2",
            close_time=parse_iso_utc("2026-01-02T00:00:00Z"),
            outcome_label=1,
            market_probability=0.55,
            model_probability=0.8,
            yes_bid=0.54,
            yes_ask=0.56,
        ),
        EvaluationRow(
            event_ticker="E2",
            close_time=parse_iso_utc("2026-01-02T00:00:00Z"),
            outcome_label=0,
            market_probability=0.45,
            model_probability=0.2,
            yes_bid=0.44,
            yes_ask=0.46,
        ),
    ]

    split = build_evaluation_split(
        rows,
        name="latest30",
        role="test",
        bootstrap_samples=200,
        bootstrap_seed=7,
    )

    assert split.name == "latest30"
    assert split.role == "test"
    assert "roi_on_cost" not in split.metrics
    assert "trade_count" not in split.metrics
    assert split.metrics["brier"].ci95 is not None
    assert split.metrics["ece"].ci95 is not None
    assert split.metrics["log_loss"].ci95 is not None


def test_build_evaluation_split_includes_event_weighted_edges() -> None:
    rows = [
        EvaluationRow(
            event_ticker="E1",
            close_time=parse_iso_utc("2026-01-01T00:00:00Z"),
            outcome_label=1,
            market_probability=0.5,
            model_probability=0.9,
            yes_bid=0.49,
            yes_ask=0.51,
        ),
        EvaluationRow(
            event_ticker="E2",
            close_time=parse_iso_utc("2026-01-02T00:00:00Z"),
            outcome_label=0,
            market_probability=0.5,
            model_probability=0.9,
            yes_bid=0.49,
            yes_ask=0.51,
        ),
        EvaluationRow(
            event_ticker="E2",
            close_time=parse_iso_utc("2026-01-02T00:00:00Z"),
            outcome_label=0,
            market_probability=0.5,
            model_probability=0.9,
            yes_bid=0.49,
            yes_ask=0.51,
        ),
        EvaluationRow(
            event_ticker="E2",
            close_time=parse_iso_utc("2026-01-02T00:00:00Z"),
            outcome_label=0,
            market_probability=0.5,
            model_probability=0.9,
            yes_bid=0.49,
            yes_ask=0.51,
        ),
    ]

    split = build_evaluation_split(
        rows,
        name="latest30",
        role="test",
        bootstrap_samples=20,
        bootstrap_seed=7,
    )

    assert split.metrics["brier"].value == 0.61
    assert split.metrics["event_weighted_brier"].value == 0.41
    assert split.metrics["brier_edge_vs_market"].value == -0.36
    assert split.metrics["event_weighted_brier_edge_vs_market"].value == -0.16
    assert split.metrics["event_weighted_log_loss"].ci95 is not None


def test_build_risk_preset_trials_exports_expected_return_percentile_bands() -> None:
    rows = [
        EvaluationRow(
            event_ticker="E1",
            close_time=parse_iso_utc("2026-01-01T00:00:00Z"),
            outcome_label=0,
            market_probability=0.42,
            model_probability=0.35,
            yes_bid=0.42,
            yes_ask=0.45,
        ),
        EvaluationRow(
            event_ticker="E2",
            close_time=parse_iso_utc("2026-01-02T00:00:00Z"),
            outcome_label=1,
            market_probability=0.55,
            model_probability=0.61,
            yes_bid=0.54,
            yes_ask=0.57,
        ),
    ]

    trials = build_risk_preset_trials(
        rows,
        presets=[get_risk_preset("balanced")],
        bootstrap_samples=100,
        bootstrap_seed=11,
    )

    assert len(trials) == 1
    trial = trials[0]
    assert trial.risk_preset_id == "balanced"
    assert trial.trade_percent > 0
    assert trial.ev_per_10_markets > 0
    assert trial.return_variance_per_market >= 0
    assert trial.roi_projection[0].market_count == 0
    assert trial.roi_projection[0].roi.expected == 0
    assert trial.roi_projection[-1].market_count == len(rows)
    assert all(
        trial.roi_projection[index].market_count
        < trial.roi_projection[index + 1].market_count
        for index in range(len(trial.roi_projection) - 1)
    )
    assert all(
        point.roi.p10 <= point.roi.expected <= point.roi.p90
        for point in trial.roi_projection
    )
    assert trial.roi_paths
    assert all(path[0].market_count == 0 and path[0].roi == 0 for path in trial.roi_paths)
    assert all(path[-1].market_count == len(rows) for path in trial.roi_paths)
    assert trial.expected_return_per_market.p10 <= trial.expected_return_per_market.expected
    assert trial.expected_return_per_market.p25 <= trial.expected_return_per_market.p75
    assert trial.expected_return_per_market.p90 >= trial.expected_return_per_market.expected
