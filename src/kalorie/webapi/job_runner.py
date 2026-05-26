from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from kalorie.domain.models import FeatureVector, TargetPhrase
from kalorie.io.documents import chunk_text, extract_text_from_pdf, normalize_text
from kalorie.market.markets import MentionMarketParseError, parse_mention_market_title
from kalorie.ml.datasets import HistoricalTrainingExample
from kalorie.ml.features import extract_feature_vectors
from kalorie.ml.labeling import label_document_chunks
from kalorie.ml.model1 import (
    MentionModelArtifact,
    predict_company_model1,
    predict_model1,
    train_company_model1,
)
from kalorie.ml.modeling import RuleBasedBaseline
from kalorie.webapi.data_cache import CacheSignature, DataCacheManager
from kalorie.webapi.job_registry import JobRegistry
from kalorie.webapi.kalshi_service import KalshiWebService, WebMentionMarket
from kalorie.webapi.run_store import EventScope, RunStore


@dataclass(frozen=True)
class JobExecutionContext:
    job_id: str
    scope: EventScope
    run_id: str
    market_ticker: str
    effective_cutoff_ts: datetime


class JobRunner:
    def __init__(
        self,
        *,
        run_store: RunStore,
        job_registry: JobRegistry,
        cache_manager: DataCacheManager,
        kalshi_service: KalshiWebService,
        project_root: Path,
    ) -> None:
        self._run_store = run_store
        self._job_registry = job_registry
        self._cache_manager = cache_manager
        self._kalshi_service = kalshi_service
        self._project_root = project_root

    def run_job(self, context: JobExecutionContext) -> None:
        run = self._run_store.get_run(context.scope, context.run_id)
        if run is None:
            return
        try:
            self._run_store.update_status(
                scope=context.scope,
                run_id=context.run_id,
                status="running",
            )
            signature = CacheSignature(
                pipeline_version="webapi-v1",
                feature_schema_version="features-v1",
                model_recipe_version="model1-company-v1",
                phrase_catalog_hash="default",
                source_manifest_hash="default",
                cutoff_policy_version="decision-cutoff-v1",
            )
            cache_reused, manifest_path = self._cache_manager.ensure_event_cache(
                scope=context.scope,
                cutoff_ts=context.effective_cutoff_ts,
                signature=signature,
            )
            prediction_rows, model_metadata = self._build_prediction_rows(
                scope=context.scope,
                run_dir=run.run_dir,
                effective_cutoff_ts=context.effective_cutoff_ts,
            )
            summary_payload = {
                "run_id": context.run_id,
                "market_ticker": context.market_ticker,
                "effective_decision_cutoff_ts": context.effective_cutoff_ts.isoformat(),
                "cache_reused": cache_reused,
                "cache_manifest_path": str(manifest_path),
                "completed_at": datetime.now(tz=UTC).isoformat(),
                "warnings": [],
                "model": model_metadata,
            }
            (run.run_dir / "summary.json").write_text(
                json.dumps(summary_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            result_payload = {
                "run_id": context.run_id,
                "market_ticker": context.market_ticker,
                "rows": prediction_rows,
            }
            (run.run_dir / "result.json").write_text(
                json.dumps(result_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            self._job_registry.mark_completed(context.job_id)
            self._run_store.update_status(
                scope=context.scope,
                run_id=context.run_id,
                status="completed",
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._run_store.update_status(
                scope=context.scope,
                run_id=context.run_id,
                status="failed",
                message=str(exc),
            )

    def _build_prediction_rows(
        self,
        *,
        scope: EventScope,
        run_dir: Path,
        effective_cutoff_ts: datetime,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        uploaded_chunks = self._read_uploaded_chunks(
            upload_dir=run_dir / "uploads",
            effective_cutoff_ts=effective_cutoff_ts,
        )
        event_markets = [
            market
            for market in self._kalshi_service.list_open_mention_markets()
            if market.event_ticker == scope.event_key
        ]
        if not uploaded_chunks or not event_markets:
            return [], {"kind": "no_uploaded_evidence_or_markets"}

        targets: list[TargetPhrase] = []
        market_by_target: dict[str, WebMentionMarket] = {}
        for market in event_markets:
            try:
                target = parse_mention_market_title(market.title).target_phrase
            except MentionMarketParseError:
                continue
            targets.append(target)
            market_by_target[target.normalized_phrase] = market
        if not targets:
            return [], {"kind": "no_parseable_targets"}

        labels = label_document_chunks(uploaded_chunks, targets)
        feature_vectors = extract_feature_vectors(uploaded_chunks, targets, labels)
        predictions, model_metadata = self._predict_feature_vectors(
            scope=scope,
            feature_vectors=feature_vectors,
        )
        rows: list[dict[str, object]] = []
        for prediction in predictions:
            market = market_by_target.get(prediction.target_phrase)
            if market is None:
                continue
            spread = (market.yes_ask - market.yes_bid).quantize(Decimal("0.01"))
            rows.append(
                {
                    "target_phrase": prediction.target_phrase,
                    "yes_bid": str(market.yes_bid),
                    "yes_ask": str(market.yes_ask),
                    "spread": str(spread),
                    "volume": market.volume,
                    "model_prediction": round(prediction.probability, 6),
                    "reasons": prediction.reasons,
                }
            )
        return rows, model_metadata

    def _read_uploaded_chunks(
        self,
        *,
        upload_dir: Path,
        effective_cutoff_ts: datetime,
    ) -> list:
        chunks = []
        for path in sorted(upload_dir.iterdir()):
            if not path.is_file():
                continue
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            if modified_at > effective_cutoff_ts:
                continue
            if path.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(path)
            else:
                text = normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
            if not text:
                continue
            chunks.extend(chunk_text(text))
        return chunks

    def _predict_feature_vectors(
        self,
        *,
        scope: EventScope,
        feature_vectors: list[FeatureVector],
    ) -> tuple[list, dict[str, object]]:
        examples = self._load_training_examples()
        if examples:
            try:
                company_model = train_company_model1(
                    examples,
                    company_symbol=scope.normalized_company_symbol(),
                    min_company_rows=8,
                    regularization_c=1.0,
                    class_weight_balanced=True,
                    include_target_indicator=True,
                    recency_ema_half_life_quarters=6.0,
                )
                return (
                    [
                        predict_company_model1(company_model, feature_vector=feature_vector)
                        for feature_vector in feature_vectors
                    ],
                    {
                        "kind": "company_finetune",
                        "model_version": company_model.model_version,
                        "training_rows": company_model.training_rows,
                    },
                )
            except ValueError:
                pass

        base_model = self._load_base_model()
        if base_model is not None:
            return (
                [
                    predict_model1(
                        base_model,
                        company_symbol=scope.normalized_company_symbol(),
                        feature_vector=feature_vector,
                    )
                    for feature_vector in feature_vectors
                ],
                {"kind": "base_model", "model_version": base_model.model_version},
            )
        baseline = RuleBasedBaseline()
        return (
            [baseline.predict_proba(feature_vector) for feature_vector in feature_vectors],
            {"kind": "rule_based_fallback", "model_version": "rule-based-v0"},
        )

    def _load_training_examples(self) -> list[HistoricalTrainingExample]:
        for candidate in (
            self._project_root / "artifacts/model1/datasets/synthetic-phrase-presence-2kplus.json",
            self._project_root
            / "artifacts/model1/datasets/synthetic-phrase-presence-2kplus-with-templates.json",
            self._project_root
            / "data/datasets/training/historical/synthetic-phrase-presence-examples.json",
        ):
            if candidate.exists():
                rows = json.loads(candidate.read_text(encoding="utf-8"))
                return [HistoricalTrainingExample.model_validate(row) for row in rows]
        return []

    def _load_base_model(self) -> MentionModelArtifact | None:
        for candidate in (
            self._project_root / "artifacts/model1/models/model1-final.json",
            self._project_root / "artifacts/model1/models/model1-optimized.json",
            self._project_root / "artifacts/model1/models/model1.json",
        ):
            if candidate.exists():
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                return MentionModelArtifact.model_validate(payload)
        return None

