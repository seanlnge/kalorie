from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kalorie2.model_cards import EvaluationRow, latest_event_rows, parse_iso_utc


class SavedModelBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SavedModelTrainingSummary(SavedModelBase):
    row_count: int | None = None
    event_count: int | None = None
    feature_count: int | None = None
    nonzero_weight_count: int | None = None
    web_evidence_packet_count: int | None = None
    first_event: str | None = None
    last_event: str | None = None


class SavedModelEvaluationSnapshot(SavedModelBase):
    label: str
    trades: int | None = None
    pnl: float | None = None
    roi: float | None = None
    brier: float | None = None
    market_brier: float | None = None
    notes: str | None = None


class SavedModelCardPreview(SavedModelBase):
    split_name: str
    role: str | None = None
    market_count: int | None = None
    brier: float | None = None
    market_brier: float | None = None
    ece: float | None = None
    market_ece: float | None = None
    log_loss: float | None = None
    market_log_loss: float | None = None


class SavedModelMetadata(SavedModelBase):
    name: str
    path: str
    health: Literal["ready"] = "ready"
    model_type: str | None = None
    model_version: int | None = None
    trained_at: str | None = None
    readme: str
    readme_summary: str
    training: SavedModelTrainingSummary
    evaluation_snapshots: list[SavedModelEvaluationSnapshot] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    model_card: dict[str, Any] | None = None
    model_card_preview: SavedModelCardPreview | None = None
    risk_preset_trials: list[dict[str, Any]] = Field(default_factory=list)


class SavedModelScoreRow(SavedModelBase):
    market_ticker: str
    event_ticker: str
    model_probability: float
    market_probability: float
    residual_delta: float
    trade_decision: dict[str, Any]
    side: str
    edge: float
    cost: float
    raw: dict[str, Any]


class SavedModelRegistry:
    def __init__(self, *, models_root: Path) -> None:
        self._models_root = models_root

    def list_models(self) -> list[SavedModelMetadata]:
        if not self._models_root.exists():
            return []
        models = [
            self._load_metadata(path)
            for path in sorted(self._models_root.iterdir(), key=lambda current: current.name)
            if path.is_dir() and is_valid_model_dir(path)
        ]
        return sorted(models, key=_model_sort_key, reverse=True)

    def get_model(self, name: str) -> SavedModelMetadata:
        model_dir = self._models_root / name
        if not is_valid_model_dir(model_dir):
            raise FileNotFoundError(f"Saved model not found or invalid: {name}")
        return self._load_metadata(model_dir)

    def model_dir(self, name: str) -> Path:
        model_dir = self._models_root / name
        if not is_valid_model_dir(model_dir):
            raise FileNotFoundError(f"Saved model not found or invalid: {name}")
        return model_dir

    def _load_metadata(self, model_dir: Path) -> SavedModelMetadata:
        model_payload = _read_json(model_dir / "artifacts" / "model.json")
        feature_schema = _read_optional_json(model_dir / "artifacts" / "feature-schema.json")
        training_manifest = _read_optional_json(model_dir / "artifacts" / "training-manifest.json")
        evaluation_reports = _read_optional_json(
            model_dir / "artifacts" / "evaluation-reports.json"
        )
        model_card = _read_optional_json(model_dir / "artifacts" / "model-card.json") or None
        risk_trials_payload = _read_optional_json(
            model_dir / "artifacts" / "risk-preset-trials.json"
        )
        readme = (model_dir / "README.md").read_text(encoding="utf-8")

        model_block = _dict(model_payload.get("model"))
        training_payload = _dict(model_payload.get("training_summary"))
        training_corpus = _dict(training_manifest.get("training_corpus"))
        nonzero_weights = _dict(model_block.get("weights")) or _dict(
            feature_schema.get("nonzero_weights")
        )

        return SavedModelMetadata(
            name=model_dir.name,
            path=str(model_dir),
            model_type=_optional_str(model_payload.get("model_type")),
            model_version=_optional_int(model_payload.get("model_version")),
            trained_at=_optional_str(model_payload.get("trained_at")),
            readme=readme,
            readme_summary=_readme_summary(readme),
            training=SavedModelTrainingSummary(
                row_count=_optional_int(training_payload.get("row_count")),
                event_count=_optional_int(training_payload.get("event_count")),
                feature_count=_optional_int(
                    training_payload.get("feature_count")
                    or len(feature_schema.get("feature_names", []))
                ),
                nonzero_weight_count=len(nonzero_weights) if nonzero_weights else None,
                web_evidence_packet_count=_optional_int(
                    training_corpus.get("web_evidence_packet_count")
                    or _dict(model_payload.get("training_config")).get("web_evidence_events")
                ),
                first_event=_optional_str(training_payload.get("first_event")),
                last_event=_optional_str(training_payload.get("last_event")),
            ),
            evaluation_snapshots=_evaluation_snapshots(evaluation_reports),
            artifact_paths=_artifact_paths(model_dir),
            model_card=model_card,
            model_card_preview=_model_card_preview(model_card),
            risk_preset_trials=_risk_preset_trials(risk_trials_payload),
        )


class SavedModelScorer:
    def __init__(self, model_dir: Path) -> None:
        if not is_valid_model_dir(model_dir):
            raise FileNotFoundError(f"Saved model not found or invalid: {model_dir}")
        self._model_dir = model_dir

    def score_csv_row(self, csv_path: Path, *, row_index: int = 0) -> SavedModelScoreRow:
        runtime_path = self._model_dir / "runtime" / "model_runtime.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(runtime_path),
                "--model-dir",
                str(self._model_dir),
                "--csv",
                str(csv_path),
                "--row-index",
                str(row_index),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return normalize_score_payload(json.loads(completed.stdout))

    def score_csv_rows(self, csv_path: Path, *, row_indices: list[int]) -> list[SavedModelScoreRow]:
        return [self.score_csv_row(csv_path, row_index=row_index) for row_index in row_indices]


class CachedRuntimeSavedModelScorer:
    def __init__(self, model_dir: Path) -> None:
        if not is_valid_model_dir(model_dir):
            raise FileNotFoundError(f"Saved model not found or invalid: {model_dir}")
        self._model_dir = model_dir
        self._runtime = _load_runtime_module(model_dir / "runtime" / "model_runtime.py")
        self._model_payload = self._runtime.load_model(model_dir)
        self._web_evidence_by_event = self._runtime.load_web_evidence(model_dir)

    def score_row_dict(
        self,
        row: dict[str, Any],
        *,
        web_evidence_by_event: dict[str, Any] | None = None,
    ) -> SavedModelScoreRow:
        effective_web_evidence = self._web_evidence_by_event
        if web_evidence_by_event:
            effective_web_evidence = {**effective_web_evidence, **web_evidence_by_event}
        payload = self._runtime.score_row(
            row,
            self._model_payload,
            effective_web_evidence,
        )
        return normalize_score_payload(payload)


def is_valid_model_dir(model_dir: Path) -> bool:
    return (
        (model_dir / "artifacts" / "model.json").is_file()
        and (model_dir / "runtime" / "model_runtime.py").is_file()
        and (model_dir / "README.md").is_file()
    )


def normalize_score_payload(payload: dict[str, Any]) -> SavedModelScoreRow:
    trade_decision = _dict(payload.get("trade_decision"))
    side = str(trade_decision.get("side") or "NONE")
    edge = float(trade_decision.get("edge") or 0.0)
    cost = float(trade_decision.get("cost") or 0.0)
    return SavedModelScoreRow(
        market_ticker=str(payload.get("market_ticker") or ""),
        event_ticker=str(payload.get("event_ticker") or ""),
        model_probability=float(payload.get("model_probability", payload.get("probability"))),
        market_probability=float(payload.get("market_probability")),
        residual_delta=float(payload.get("residual_delta")),
        trade_decision=trade_decision,
        side=side,
        edge=edge,
        cost=cost,
        raw=payload,
    )


def read_sample_rows(csv_path: Path, *, limit: int = 10) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for index, row in enumerate(csv.DictReader(handle)):
            if index >= limit:
                break
            rows.append({"row_index": str(index), **row})
    return rows


def build_saved_model_evaluation_rows(
    model_dir: Path,
    *,
    latest_event_count: int = 30,
) -> list[EvaluationRow]:
    rows = _all_saved_model_evaluation_rows(model_dir)
    return latest_event_rows(rows, event_count=latest_event_count)


def _all_saved_model_evaluation_rows(model_dir: Path) -> list[EvaluationRow]:
    csv_path = _training_csv_path(model_dir)
    if not csv_path.exists():
        return []
    scorer = CachedRuntimeSavedModelScorer(model_dir)
    rows: list[EvaluationRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            outcome = str(raw_row.get("final_outcome") or "").strip().lower()
            if outcome not in {"yes", "no"}:
                continue
            try:
                score = scorer.score_row_dict(_clean_csv_row(raw_row))
                rows.append(
                    EvaluationRow(
                        event_ticker=str(raw_row["event_ticker"]),
                        close_time=parse_iso_utc(str(raw_row["close_time"])),
                        outcome_label=1 if outcome == "yes" else 0,
                        market_probability=float(raw_row["preclose_yes_mid"]),
                        model_probability=score.model_probability,
                        yes_bid=float(raw_row["preclose_yes_bid"]),
                        yes_ask=float(raw_row["preclose_yes_ask"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _training_csv_path(model_dir: Path) -> Path:
    manifest_path = model_dir / "artifacts" / "training-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        training_corpus = _dict(manifest.get("training_corpus"))
        saved_csv = training_corpus.get("saved_csv")
        if saved_csv:
            return model_dir / str(saved_csv)
    return model_dir / "training" / "mention-markets-historical-20260523.csv"


def _clean_csv_row(row: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in row.items() if value not in {"", None}}


def _artifact_paths(model_dir: Path) -> dict[str, str]:
    paths = {
        "model": model_dir / "artifacts" / "model.json",
        "runtime": model_dir / "runtime" / "model_runtime.py",
        "readme": model_dir / "README.md",
        "feature_schema": model_dir / "artifacts" / "feature-schema.json",
        "training_manifest": model_dir / "artifacts" / "training-manifest.json",
        "evaluation_reports": model_dir / "artifacts" / "evaluation-reports.json",
        "model_card": model_dir / "artifacts" / "model-card.json",
        "risk_preset_trials": model_dir / "artifacts" / "risk-preset-trials.json",
    }
    return {
        name: str(path.relative_to(model_dir)).replace("\\", "/")
        for name, path in paths.items()
        if path.exists()
    }


def _load_runtime_module(runtime_path: Path) -> ModuleType:
    module_name = f"kalorie2_saved_runtime_{abs(hash(runtime_path))}"
    spec = importlib.util.spec_from_file_location(module_name, runtime_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load saved model runtime: {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evaluation_snapshots(payload: dict[str, Any]) -> list[SavedModelEvaluationSnapshot]:
    snapshots: list[SavedModelEvaluationSnapshot] = []
    full_eval = _dict(payload.get("full_web_evaluation"))
    full_eval_summary = _dict(full_eval.get("summary"))
    if full_eval_summary:
        snapshots.append(
            SavedModelEvaluationSnapshot(
                label="Full web evaluation",
                brier=_optional_float(full_eval_summary.get("brier_score")),
                market_brier=_optional_float(full_eval_summary.get("market_brier_score")),
                notes=_optional_str(full_eval.get("run_id")),
            )
        )

    full_backtest = _dict(payload.get("full_web_backtest"))
    if full_backtest:
        snapshots.append(_snapshot_from_summary("Full web backtest", full_backtest.get("summary")))

    holdout = _dict(payload.get("temporal_holdout"))
    holdout_backtest = _dict(holdout.get("backtest"))
    all_sides = holdout_backtest.get("all_sides") or holdout_backtest.get("all")
    no_only = holdout_backtest.get("no_only")
    if all_sides:
        snapshots.append(_snapshot_from_summary("Latest-30 holdout all-sides", all_sides))
    if no_only:
        snapshots.append(_snapshot_from_summary("Latest-30 holdout NO-only", no_only))

    no_only_ci = _dict(payload.get("no_only_ci"))
    for label, section in no_only_ci.items():
        if isinstance(section, dict) and "point_estimate" in section:
            snapshots.append(
                _snapshot_from_summary(
                    f"NO-only CI {label.replace('_', ' ')}",
                    section.get("point_estimate"),
                )
            )
    return snapshots


def _model_card_preview(payload: dict[str, Any] | None) -> SavedModelCardPreview | None:
    if not payload:
        return None
    splits = payload.get("evaluation_splits")
    if not isinstance(splits, list):
        return None
    split = _primary_model_card_split(splits)
    if split is None:
        return None
    metrics = _dict(split.get("metrics"))
    market_count = _optional_int(split.get("market_count"))
    return SavedModelCardPreview(
        split_name=str(split.get("name") or "evaluation"),
        role=_optional_str(split.get("role")),
        market_count=market_count,
        brier=_optional_float(_metric_value(metrics, "brier")),
        market_brier=_optional_float(_metric_value(metrics, "market_brier")),
        ece=_optional_float(_metric_value(metrics, "ece")),
        market_ece=_optional_float(_metric_value(metrics, "market_ece")),
        log_loss=_optional_float(_metric_value(metrics, "log_loss")),
        market_log_loss=_optional_float(_metric_value(metrics, "market_log_loss")),
    )


def _primary_model_card_split(
    splits: list[Any],
) -> dict[str, Any] | None:
    split_dicts = [split for split in splits if isinstance(split, dict)]
    if not split_dicts:
        return None
    for split in split_dicts:
        if split.get("role") == "test":
            return split
    return split_dicts[0]


def _metric_value(metrics: dict[str, Any], metric_name: str) -> Any:
    metric = metrics.get(metric_name)
    if isinstance(metric, dict):
        return metric.get("value")
    return metric


def _risk_preset_trials(payload: dict[str, Any]) -> list[dict[str, Any]]:
    trials = payload.get("risk_preset_trials")
    if not isinstance(trials, list):
        return []
    return [trial for trial in trials if isinstance(trial, dict)]


def _model_sort_key(model: SavedModelMetadata) -> tuple[int, str, str]:
    card_version = _optional_int(_dict(model.model_card or {}).get("model_version"))
    version = card_version if card_version is not None else model.model_version
    return (
        version if version is not None else -1,
        model.trained_at or "",
        model.name,
    )


def _snapshot_from_summary(label: str, summary: object) -> SavedModelEvaluationSnapshot:
    summary_payload = _dict(summary)
    return SavedModelEvaluationSnapshot(
        label=label,
        trades=_optional_int(summary_payload.get("trades")),
        pnl=_optional_float(summary_payload.get("total_pnl")),
        roi=_optional_float(summary_payload.get("roi_on_cost")),
    )


def _readme_summary(readme: str) -> str:
    for line in readme.splitlines():
        cleaned = line.strip()
        if cleaned and not cleaned.startswith("#") and not cleaned.startswith("```"):
            return cleaned
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    return _dict(json.loads(path.read_text(encoding="utf-8")))


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
