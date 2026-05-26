import json

from typer.testing import CliRunner

from kalorie.app.cli import app


def test_build_hedge_plan_cli_writes_allocations(tmp_path):
    predictions_path = tmp_path / "predictions.json"
    out_path = tmp_path / "hedge-plan.json"
    payload = {
        "event_ticker": "KXEARNINGSMENTIONNVDA-26MAY20",
        "rows": [
            {
                "market_id": "NVDA-AUTO",
                "phrase": "automation",
                "kalshi_yes_bid": 0.48,
                "kalshi_yes_ask": 0.50,
                "model_company_probability": 0.70,
            },
            {
                "market_id": "NVDA-TRUM",
                "phrase": "trump",
                "kalshi_yes_bid": 0.78,
                "kalshi_yes_ask": 0.80,
                "model_company_probability": 0.20,
            },
        ],
    }
    predictions_path.write_text(json.dumps(payload), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "build-hedge-plan",
            "--predictions",
            str(predictions_path),
            "--out",
            str(out_path),
            "--budget",
            "100",
            "--risk-aversion",
            "0",
            "--max-fraction-per-market",
            "0.5",
            "--force-full-deployment",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["position_count"] == 2
    assert report["deployed_dollars"] == 100.0
    assert report["expected_profit_dollars"] > 0
    assert report["context"]["event_ticker"] == "KXEARNINGSMENTIONNVDA-26MAY20"
