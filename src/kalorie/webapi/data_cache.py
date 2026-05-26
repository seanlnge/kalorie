from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kalorie.webapi.run_store import EventScope, RunStore


@dataclass(frozen=True)
class CacheSignature:
    pipeline_version: str
    feature_schema_version: str
    model_recipe_version: str
    phrase_catalog_hash: str
    source_manifest_hash: str
    cutoff_policy_version: str

    def as_dict(self) -> dict[str, str]:
        return {
            "pipeline_version": self.pipeline_version,
            "feature_schema_version": self.feature_schema_version,
            "model_recipe_version": self.model_recipe_version,
            "phrase_catalog_hash": self.phrase_catalog_hash,
            "source_manifest_hash": self.source_manifest_hash,
            "cutoff_policy_version": self.cutoff_policy_version,
        }


class DataCacheManager:
    def __init__(self, *, run_store: RunStore) -> None:
        self._run_store = run_store

    def ensure_event_cache(
        self,
        *,
        scope: EventScope,
        cutoff_ts: datetime,
        signature: CacheSignature,
    ) -> tuple[bool, Path]:
        data_dir = self._run_store.event_data_dir(scope)
        data_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = data_dir / "cache_manifest.json"
        reused = False
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            reused = (
                existing.get("cutoff_ts") == cutoff_ts.isoformat()
                and existing.get("signature") == signature.as_dict()
            )
        if not reused:
            signature_payload = signature.as_dict()
            payload = {
                "scope": {
                    "company_symbol": scope.normalized_company_symbol(),
                    "event_key": scope.event_key,
                },
                "cutoff_ts": cutoff_ts.isoformat(),
                "signature": signature_payload,
                **signature_payload,
            }
            manifest_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return reused, manifest_path

