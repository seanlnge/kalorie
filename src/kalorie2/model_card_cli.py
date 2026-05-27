from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from kalorie2.model_cards import (
    EvaluationRow,
    ModelCard,
    build_evaluation_split,
    build_model_card_schema,
    latest_event_rows,
    parse_iso_utc,
)
from kalorie2.risk_presets import list_risk_presets
from kalorie2.risk_trials import build_risk_preset_trials
from kalorie2.saved_models import CachedRuntimeSavedModelScorer, is_valid_model_dir


def generate_model_cards(
    *,
    models_root: Path,
    latest_event_count: int = 30,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 531,
) -> list[Path]:
    written: list[Path] = []
    for model_dir in sorted(models_root.iterdir(), key=lambda path: path.name):
        if not model_dir.is_dir() or not is_valid_model_dir(model_dir):
            continue
        generate_model_card_for_model(
            model_dir=model_dir,
            latest_event_count=latest_event_count,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        written.append(model_dir / "artifacts" / "model-card.json")
    return written


def generate_model_card_for_model(
    *,
    model_dir: Path,
    latest_event_count: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    artifacts_dir = model_dir / "artifacts"
    model_payload = _read_json(artifacts_dir / "model.json")
    feature_schema = _read_optional_json(artifacts_dir / "feature-schema.json")
    training_manifest_path = artifacts_dir / "training-manifest.json"
    training_manifest = _read_optional_json(training_manifest_path)

    training_csv_path = _training_csv_path(model_dir, training_manifest)
    raw_rows = _load_csv_rows(training_csv_path)
    scorer = CachedRuntimeSavedModelScorer(model_dir)
    rows = _evaluation_rows(raw_rows, scorer)
    latest_rows = latest_event_rows(rows, event_count=latest_event_count)

    card = ModelCard(
        model_name=str(model_payload.get("model_name") or model_dir.name),
        model_version=_optional_int(model_payload.get("model_version")),
        model_type=str(model_payload.get("model_type") or "unknown"),
        training_data={
            "row_count": len(raw_rows),
            "event_count": len({row["event_ticker"] for row in raw_rows}),
            "web_evidence_packet_count": _training_web_packet_count(training_manifest),
            "source": _display_training_source(model_dir, training_manifest, training_csv_path),
            "evaluation_protocol": (
                "Primary test split uses latest events by close time scored with saved runtime."
            ),
        },
        feature_set={
            "feature_count": _feature_count(feature_schema, model_payload),
            "nonzero_weight_count": _nonzero_weight_count(feature_schema, model_payload),
            "ablation_group": _ablation_group(feature_schema, model_payload),
            "dropped_feature_prefixes": _dropped_prefixes(feature_schema, model_payload),
        },
        evaluation_splits=[
            build_evaluation_split(
                latest_rows,
                name=f"latest{latest_event_count}",
                role="test",
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                notes=(
                    "Primary held-out style slice: "
                    f"latest {latest_event_count} events by close time."
                ),
            ),
            build_evaluation_split(
                rows,
                name="full_scored_window",
                role="backtest",
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 1,
                notes="Secondary context across all rows scored by the saved runtime.",
            ),
        ],
        caveats=[
            "CI values use event-bootstrap resampling.",
            (
                "Model card metrics evaluate predictive quality only; "
                "risk presets handle trading policy."
            ),
        ],
        recommended_use=(
            "Use this card as a comparable validation summary across saved model bundles."
        ),
    )
    risk_preset_trials = build_risk_preset_trials(
        latest_rows,
        presets=list_risk_presets(),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 2,
    )

    schema_path = artifacts_dir / "model-card.schema.json"
    card_path = artifacts_dir / "model-card.json"
    risk_trials_path = artifacts_dir / "risk-preset-trials.json"
    schema_path.write_text(
        json.dumps(build_model_card_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    card_path.write_text(
        card.model_dump_json(indent=2, exclude={"primary_test_split"}) + "\n",
        encoding="utf-8",
    )
    risk_trials_path.write_text(
        json.dumps(
            {
                "risk_preset_trials": [
                    trial.model_dump(mode="json") for trial in risk_preset_trials
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if training_manifest_path.exists():
        training_manifest.setdefault("artifacts", {})
        training_manifest["artifacts"]["model_card"] = "artifacts/model-card.json"
        training_manifest["artifacts"]["model_card_schema"] = "artifacts/model-card.schema.json"
        training_manifest["artifacts"]["risk_preset_trials"] = (
            "artifacts/risk-preset-trials.json"
        )
        training_manifest_path.write_text(
            json.dumps(training_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _evaluation_rows(
    raw_rows: list[dict[str, Any]],
    scorer: CachedRuntimeSavedModelScorer,
) -> list[EvaluationRow]:
    rows: list[EvaluationRow] = []
    for row in raw_rows:
        outcome = str(row.get("final_outcome") or "").strip().lower()
        if outcome not in {"yes", "no"}:
            continue
        score = scorer.score_row_dict(_clean_csv_row(row))
        rows.append(
            EvaluationRow(
                event_ticker=str(row["event_ticker"]),
                close_time=parse_iso_utc(str(row["close_time"])),
                outcome_label=1 if outcome == "yes" else 0,
                market_probability=float(row["preclose_yes_mid"]),
                model_probability=float(score.model_probability),
                yes_bid=float(row["preclose_yes_bid"]),
                yes_ask=float(row["preclose_yes_ask"]),
            )
        )
    return rows


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _training_csv_path(model_dir: Path, training_manifest: dict[str, Any]) -> Path:
    training_corpus = _dict(training_manifest.get("training_corpus"))
    saved_csv = training_corpus.get("saved_csv")
    if isinstance(saved_csv, str) and saved_csv.strip():
        candidate = model_dir / Path(saved_csv.replace("\\", "/"))
        if candidate.exists():
            return candidate
    default_candidate = model_dir / "training" / "mention-markets-historical-20260523.csv"
    if default_candidate.exists():
        return default_candidate
    raise FileNotFoundError(f"Could not resolve training CSV for {model_dir}")


def _display_training_source(
    model_dir: Path,
    training_manifest: dict[str, Any],
    training_csv_path: Path,
) -> str:
    training_corpus = _dict(training_manifest.get("training_corpus"))
    source = training_corpus.get("saved_csv")
    if isinstance(source, str) and source.strip():
        return source.replace("\\", "/")
    return str(training_csv_path.relative_to(model_dir)).replace("\\", "/")


def _feature_count(feature_schema: dict[str, Any], model_payload: dict[str, Any]) -> int:
    names = feature_schema.get("feature_names")
    if isinstance(names, list):
        return len(names)
    means = _dict(_dict(model_payload.get("model")).get("feature_means"))
    return len(means)


def _nonzero_weight_count(feature_schema: dict[str, Any], model_payload: dict[str, Any]) -> int:
    weights = feature_schema.get("nonzero_weights")
    if isinstance(weights, dict):
        return len(weights)
    return len(_dict(_dict(model_payload.get("model")).get("weights")))


def _ablation_group(feature_schema: dict[str, Any], model_payload: dict[str, Any]) -> str:
    if isinstance(feature_schema.get("feature_ablation_group"), str):
        return str(feature_schema["feature_ablation_group"])
    training_config = _dict(model_payload.get("training_config"))
    if isinstance(training_config.get("feature_ablation_group"), str):
        return str(training_config["feature_ablation_group"])
    return "none"


def _dropped_prefixes(feature_schema: dict[str, Any], model_payload: dict[str, Any]) -> list[str]:
    value = feature_schema.get("dropped_feature_prefixes")
    if isinstance(value, list):
        return [str(entry) for entry in value]
    training_config = _dict(model_payload.get("training_config"))
    group = training_config.get("feature_ablation_group")
    if group == "resolution":
        return ["resolution_"]
    return []


def _training_web_packet_count(training_manifest: dict[str, Any]) -> int | None:
    training_corpus = _dict(training_manifest.get("training_corpus"))
    value = training_corpus.get("web_evidence_packet_count")
    return _optional_int(value)


def _clean_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: None if value == "" and key in {"status", "settlement_ts"} else value
        for key, value in row.items()
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model cards for saved model bundles.")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path.cwd() / "models",
        help="Path containing saved model folders.",
    )
    parser.add_argument(
        "--latest-events",
        type=int,
        default=30,
        help="Number of latest events for the primary test split.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Number of event-bootstrap samples for confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=531,
        help="Base random seed for bootstrap sampling.",
    )
    args = parser.parse_args()
    written = generate_model_cards(
        models_root=args.models_root,
        latest_event_count=args.latest_events,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
