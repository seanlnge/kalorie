import json

from kalorie2.model_cards import (
    ConfidenceInterval,
    EvaluationSplit,
    MetricValue,
    ModelCard,
    build_model_card_schema,
)


def test_model_card_schema_requires_latest30_test_metrics() -> None:
    card = ModelCard(
        model_name="kalorie-v3",
        model_version=3,
        model_type="market_anchored_linear_residual",
        default_execution_policy="no_only",
        default_margin=0.02,
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
                policy="no_only",
                margin=0.02,
                metrics={
                    "roi_on_cost": MetricValue(
                        value=0.293532,
                        unit="ratio",
                        ci95=ConfidenceInterval(low=0.05, high=0.52),
                    ),
                    "trade_count": MetricValue(value=35, unit="count"),
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
    assert latest30.metrics["roi_on_cost"].ci95 is not None
    assert latest30.metrics["trade_count"].value == 35
    assert latest30.metrics["log_loss"].ci95 is not None


def test_build_model_card_schema_exports_json_schema() -> None:
    schema = build_model_card_schema()

    assert schema["title"] == "ModelCard"
    assert "evaluation_splits" in schema["properties"]
    assert json.dumps(schema)
