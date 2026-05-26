import csv
import json
from pathlib import Path

from kalorie2.saved_models import (
    CachedRuntimeSavedModelScorer,
    SavedModelRegistry,
    SavedModelScorer,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_bundle(root: Path, name: str = "unit-model") -> Path:
    model_dir = root / name
    (model_dir / "runtime").mkdir(parents=True)
    (model_dir / "training").mkdir()
    (model_dir / "README.md").write_text(
        "# Unit Model\n\nShort model summary.\n\n## Evaluation Snapshot\n\nUseful snapshot.",
        encoding="utf-8",
    )
    _write_json(
        model_dir / "artifacts" / "model.json",
        {
            "model_name": name,
            "model_type": "market_anchored_linear_residual",
            "trained_at": "2026-05-26T02:39:41+00:00",
            "training_summary": {"row_count": 10, "event_count": 2, "feature_count": 3},
            "model": {"weights": {"alpha": 0.1, "beta": -0.2}},
        },
    )
    _write_json(
        model_dir / "artifacts" / "feature-schema.json",
        {"feature_names": ["alpha", "beta", "gamma"], "nonzero_weights": {"alpha": 0.1}},
    )
    _write_json(
        model_dir / "artifacts" / "training-manifest.json",
        {
            "model_name": name,
            "training_corpus": {
                "saved_csv": "training/rows.csv",
                "web_evidence_packet_count": 2,
            },
        },
    )
    _write_json(
        model_dir / "artifacts" / "evaluation-reports.json",
        {
            "full_web_backtest": {
                "summary": {"trades": 4, "total_pnl": 1.25, "roi_on_cost": 0.125}
            },
            "temporal_holdout": {
                "backtest": {"no_only": {"trades": 3, "total_pnl": 1.0, "roi_on_cost": 0.2}}
            },
        },
    )
    (model_dir / "runtime" / "model_runtime.py").write_text(
        "print('not used in this test')\n",
        encoding="utf-8",
    )
    return model_dir


def test_registry_discovers_only_valid_saved_model_folders(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "valid-model")
    (tmp_path / "invalid-model" / "artifacts").mkdir(parents=True)
    (tmp_path / "invalid-model" / "README.md").write_text("missing runtime", encoding="utf-8")

    registry = SavedModelRegistry(models_root=tmp_path)

    models = registry.list_models()

    assert [model.name for model in models] == ["valid-model"]
    assert models[0].artifact_paths["model"] == "artifacts/model.json"
    assert models[0].health == "ready"


def test_registry_parses_metadata_summary_from_bundle_artifacts(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "unit-model")

    metadata = SavedModelRegistry(models_root=tmp_path).get_model("unit-model")

    assert metadata.name == "unit-model"
    assert metadata.readme_summary == "Short model summary."
    assert metadata.training.row_count == 10
    assert metadata.training.event_count == 2
    assert metadata.training.feature_count == 3
    assert metadata.training.nonzero_weight_count == 2
    assert metadata.training.web_evidence_packet_count == 2
    assert metadata.evaluation_snapshots[0].label == "Full web backtest"
    assert metadata.evaluation_snapshots[0].trades == 4
    assert metadata.evaluation_snapshots[0].pnl == 1.25
    assert metadata.evaluation_snapshots[0].roi == 0.125


def test_registry_uses_folder_slug_as_selectable_model_name(tmp_path: Path) -> None:
    model_dir = _write_bundle(tmp_path, "folder-slug")
    payload = json.loads((model_dir / "artifacts" / "model.json").read_text(encoding="utf-8"))
    payload["model_name"] = "payload-display-name"
    _write_json(model_dir / "artifacts" / "model.json", payload)

    registry = SavedModelRegistry(models_root=tmp_path)

    assert registry.list_models()[0].name == "folder-slug"
    assert registry.get_model("folder-slug").name == "folder-slug"


def test_scorer_invokes_runtime_and_normalizes_trade_rows(tmp_path: Path) -> None:
    model_dir = _write_bundle(tmp_path, "unit-model")
    runtime_path = model_dir / "runtime" / "model_runtime.py"
    runtime_path.write_text(
        "\n".join(
            [
                "import argparse, csv, json",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--model-dir')",
                "parser.add_argument('--csv')",
                "parser.add_argument('--row-index', type=int)",
                "args = parser.parse_args()",
                "with open(args.csv, newline='', encoding='utf-8') as handle:",
                "    row = list(csv.DictReader(handle))[args.row_index]",
                "print(json.dumps({",
                "    'market_ticker': row['market_ticker'],",
                "    'event_ticker': row['event_ticker'],",
                "    'probability': 0.42,",
                "    'market_probability': 0.50,",
                "    'residual_delta': -0.32,",
                "    'trade_decision': {'side': 'NO', 'cost': 0.48, 'edge': 0.07},",
                "    'features': {'alpha': 1.0},",
                "}))",
            ]
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["market_ticker", "event_ticker"])
        writer.writeheader()
        writer.writerow({"market_ticker": "MARKET-1", "event_ticker": "EVENT-1"})

    row = SavedModelScorer(model_dir).score_csv_row(csv_path, row_index=0)

    assert row.market_ticker == "MARKET-1"
    assert row.event_ticker == "EVENT-1"
    assert row.model_probability == 0.42
    assert row.market_probability == 0.50
    assert row.residual_delta == -0.32
    assert row.side == "NO"
    assert row.edge == 0.07
    assert row.raw["features"] == {"alpha": 1.0}


def test_cached_runtime_scorer_scores_rows_without_subprocess_contract(tmp_path: Path) -> None:
    model_dir = _write_bundle(tmp_path, "unit-model")
    (model_dir / "runtime" / "model_runtime.py").write_text(
        "\n".join(
            [
                "load_count = 0",
                "def load_model(model_dir):",
                "    global load_count",
                "    load_count += 1",
                "    return {'loaded': load_count}",
                "def load_web_evidence(model_dir):",
                "    return {'EVENT-1': {'items': []}}",
                "def score_row(row, model_payload, web_evidence_by_event=None):",
                "    return {",
                "        'market_ticker': row['market_ticker'],",
                "        'event_ticker': row['event_ticker'],",
                "        'probability': 0.61,",
                "        'market_probability': 0.55,",
                "        'residual_delta': 0.24,",
                "        'trade_decision': {'side': 'YES', 'cost': 0.6, 'edge': 0.01},",
                "        'load_count': model_payload['loaded'],",
                "    }",
            ]
        ),
        encoding="utf-8",
    )

    scorer = CachedRuntimeSavedModelScorer(model_dir)
    first = scorer.score_row_dict({"market_ticker": "M1", "event_ticker": "EVENT-1"})
    second = scorer.score_row_dict({"market_ticker": "M2", "event_ticker": "EVENT-1"})

    assert first.model_probability == 0.61
    assert second.market_ticker == "M2"
    assert first.raw["load_count"] == 1
    assert second.raw["load_count"] == 1
