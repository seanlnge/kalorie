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
    policy: str | None = None
    trade_count: int | None = None
    market_count: int | None = None
    trade_percent: float | None = None
    brier: float | None = None
    roi_on_cost: float | None = None
    ev_per_10_trades: float | None = None


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


def _artifact_paths(model_dir: Path) -> dict[str, str]:
    paths = {
        "model": model_dir / "artifacts" / "model.json",
        "runtime": model_dir / "runtime" / "model_runtime.py",
        "readme": model_dir / "README.md",
        "feature_schema": model_dir / "artifacts" / "feature-schema.json",
        "training_manifest": model_dir / "artifacts" / "training-manifest.json",
        "evaluation_reports": model_dir / "artifacts" / "evaluation-reports.json",
        "model_card": model_dir / "artifacts" / "model-card.json",
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
    default_policy = _optional_str(payload.get("default_execution_policy"))
    split = _primary_model_card_split(splits, default_policy=default_policy)
    if split is None:
        return None
    metrics = _dict(split.get("metrics"))
    trade_count = _optional_int(_metric_value(metrics, "trade_count"))
    market_count = _optional_int(split.get("market_count"))
    total_cost = _optional_float(_metric_value(metrics, "total_cost"))
    total_pnl = _optional_float(_metric_value(metrics, "total_pnl"))
    roi_on_cost = _optional_float(_metric_value(metrics, "roi_on_cost"))
    ev_per_10_trades = _ev_per_10_trades(
        roi_on_cost=roi_on_cost,
        total_cost=total_cost,
        total_pnl=total_pnl,
        trade_count=trade_count,
    )
    return SavedModelCardPreview(
        split_name=str(split.get("name") or "evaluation"),
        role=_optional_str(split.get("role")),
        policy=_optional_str(split.get("policy")),
        trade_count=trade_count,
        market_count=market_count,
        trade_percent=trade_count / market_count
        if trade_count is not None and market_count
        else None,
        brier=_optional_float(_metric_value(metrics, "brier")),
        roi_on_cost=roi_on_cost,
        ev_per_10_trades=ev_per_10_trades,
    )


def _primary_model_card_split(
    splits: list[Any],
    *,
    default_policy: str | None,
) -> dict[str, Any] | None:
    split_dicts = [split for split in splits if isinstance(split, dict)]
    if not split_dicts:
        return None
    for split in split_dicts:
        if split.get("role") == "test" and (
            default_policy is None or split.get("policy") == default_policy
        ):
            return split
    for split in split_dicts:
        if split.get("role") == "test":
            return split
    return split_dicts[0]


def _metric_value(metrics: dict[str, Any], metric_name: str) -> Any:
    metric = metrics.get(metric_name)
    if isinstance(metric, dict):
        return metric.get("value")
    return metric


def _ev_per_10_trades(
    *,
    roi_on_cost: float | None,
    total_cost: float | None,
    total_pnl: float | None,
    trade_count: int | None,
) -> float | None:
    if not trade_count:
        return None
    if roi_on_cost is not None and total_cost is not None:
        return (roi_on_cost * total_cost / trade_count) * 10
    if total_pnl is not None:
        return (total_pnl / trade_count) * 10
    return None


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
