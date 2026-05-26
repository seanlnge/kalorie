from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import get_ident
from typing import Any

from pydantic import ValidationError

from kalorie.data_grepping.event_scenarios import EventScenarioCatalog
from kalorie.data_grepping.materials import load_material_snippets
from kalorie.io.public_documents import PublicDocumentManifest
from kalorie.workflows.models import PhraseCatalog, TranscriptInventoryRow

PROMPT_VERSION = "event-dossier-v1"
EVIDENCE_CUTOFF_LEAD = timedelta(minutes=10)
NOISY_VARIANT_TARGETS = {
    "a",
    "an",
    "are",
    "i",
    "it",
    "re",
    "s",
    "the",
    "they",
    "this",
    "we",
    "what",
    "you",
    "your",
}


def event_dossier_id(company_symbol: str, fiscal_year: int, fiscal_quarter: int) -> str:
    return f"{company_symbol.upper()}-{fiscal_year}-Q{fiscal_quarter}"


def source_digest_for_event(
    *,
    manifests: list[PublicDocumentManifest],
    target_phrases: list[str],
    prompt_version: str = PROMPT_VERSION,
    llm_model: str | None = None,
    max_documents: int = 20,
    max_chars_per_document: int = 3000,
    max_items: int = 20,
) -> str:
    payload = {
        "generation": {
            "llm_model": llm_model,
            "max_chars_per_document": max_chars_per_document,
            "max_documents": max_documents,
            "max_items": max_items,
        },
        "prompt_version": prompt_version,
        "target_phrases": sorted(
            {phrase.strip().lower() for phrase in target_phrases if phrase.strip()}
        ),
        "sources": [
            {
                "source_url": manifest.source_url,
                "content_hash": manifest.content_hash,
                "published_at": manifest.published_at.isoformat(),
                "source_type": manifest.source_type,
            }
            for manifest in sorted(
                manifests,
                key=lambda row: (
                    row.source_url,
                    row.content_hash,
                    row.published_at.isoformat(),
                    row.source_type,
                ),
            )
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_event_dossiers(
    *,
    inventory_rows: list[TranscriptInventoryRow],
    manifests: list[PublicDocumentManifest],
    phrase_catalog: PhraseCatalog,
    generator: Any,
    cache_dir: Path,
    max_documents: int = 20,
    max_chars_per_document: int = 3000,
    max_items: int = 20,
    prompt_version: str = PROMPT_VERSION,
    max_workers: int = 1,
    progress_callback: Callable[[int, int, str, bool], None] | None = None,
) -> list[EventScenarioCatalog]:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    manifests_by_period = _manifests_by_period(manifests)
    phrases_by_period = _phrases_by_period(phrase_catalog)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in sorted(
        inventory_rows,
        key=lambda item: (item.company_symbol, item.fiscal_year, item.fiscal_quarter),
    ):
        period_key = (row.company_symbol, row.fiscal_year, row.fiscal_quarter)
        event_manifests = _manifests_before_evidence_cutoff(
            row,
            manifests_by_period.get(period_key, []),
        )
        target_phrases = phrases_by_period.get(period_key, [])
        if not event_manifests or not target_phrases:
            continue
        dossier_id = event_dossier_id(row.company_symbol, row.fiscal_year, row.fiscal_quarter)
        digest = source_digest_for_event(
            manifests=event_manifests,
            target_phrases=target_phrases,
            prompt_version=prompt_version,
            llm_model=getattr(generator, "model", None),
            max_documents=max_documents,
            max_chars_per_document=max_chars_per_document,
            max_items=max_items,
        )
        jobs.append((row, event_manifests, target_phrases, dossier_id, digest))
    catalogs_by_index: dict[int, EventScenarioCatalog] = {}
    done = 0

    def run_job(index: int) -> tuple[int, EventScenarioCatalog, bool]:
        row, event_manifests, target_phrases, dossier_id, digest = jobs[index]
        cache_path = cache_dir / f"{dossier_id}.json"
        cached = _load_cached_catalog(cache_path, expected_source_digest=digest)
        if cached is not None:
            return index, cached, True
        snippets = load_material_snippets(
            event_manifests,
            max_documents=max_documents,
            max_chars_per_document=max_chars_per_document,
            company_symbol=row.company_symbol,
        )
        catalog = generator.generate(
            event_id=dossier_id,
            company_symbol=row.company_symbol,
            company_name=row.company_name,
            target_phrases=target_phrases,
            material_snippets=snippets,
            max_items=max_items,
        )
        catalog.source_digest = digest
        catalog.prompt_version = prompt_version
        _write_cached_catalog(cache_path, catalog)
        return index, catalog, False

    if max_workers == 1:
        for index in range(len(jobs)):
            job_index, catalog, reused = run_job(index)
            catalogs_by_index[job_index] = catalog
            done += 1
            if progress_callback is not None:
                progress_callback(done, len(jobs), catalog.event_id, reused)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_job, index) for index in range(len(jobs))]
            for future in as_completed(futures):
                job_index, catalog, reused = future.result()
                catalogs_by_index[job_index] = catalog
                done += 1
                if progress_callback is not None:
                    progress_callback(done, len(jobs), catalog.event_id, reused)
    return [catalogs_by_index[index] for index in sorted(catalogs_by_index)]


def scenario_texts_by_event(
    catalogs: list[EventScenarioCatalog],
) -> dict[str, list[str]]:
    return {
        catalog.event_id: catalog.scenario_texts()
        for catalog in catalogs
        if catalog.scenario_texts()
    }


def phrase_variants_by_event(
    catalogs: list[EventScenarioCatalog],
) -> dict[str, dict[str, list[str]]]:
    variants_by_event: dict[str, dict[str, list[str]]] = {}
    for catalog in catalogs:
        event_variants: dict[str, list[str]] = {}
        for phrase, variants in catalog.target_phrase_variants.items():
            normalized_phrase = phrase.strip().lower()
            if is_noisy_phrase_target(normalized_phrase):
                continue
            cleaned_variants = [
                variant
                for variant in dict.fromkeys(value.strip() for value in variants)
                if variant
            ]
            if cleaned_variants:
                event_variants[normalized_phrase] = cleaned_variants
        if event_variants:
            variants_by_event[catalog.event_id] = event_variants
    return variants_by_event


def load_event_dossier_catalogs(path: Path) -> list[EventScenarioCatalog]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    return [EventScenarioCatalog.model_validate(row) for row in rows]


def _load_cached_catalog(
    path: Path,
    *,
    expected_source_digest: str,
) -> EventScenarioCatalog | None:
    if not path.exists():
        return None
    try:
        catalog = EventScenarioCatalog.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError):
        return None
    if catalog.source_digest != expected_source_digest:
        return None
    return catalog


def _write_cached_catalog(path: Path, catalog: EventScenarioCatalog) -> None:
    tmp_path = path.with_name(f"{path.name}.{get_ident()}.tmp")
    tmp_path.write_text(catalog.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _manifests_before_evidence_cutoff(
    row: TranscriptInventoryRow,
    manifests: list[PublicDocumentManifest],
) -> list[PublicDocumentManifest]:
    if row.estimated_call_time is None:
        return manifests
    cutoff = _aware_datetime(row.estimated_call_time) - EVIDENCE_CUTOFF_LEAD
    return [
        manifest
        for manifest in manifests
        if _aware_datetime(manifest.published_at) <= cutoff
    ]


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def is_noisy_phrase_target(phrase: str) -> bool:
    return len(phrase) <= 1 or phrase in NOISY_VARIANT_TARGETS


def _manifests_by_period(
    manifests: list[PublicDocumentManifest],
) -> dict[tuple[str, int, int], list[PublicDocumentManifest]]:
    grouped: dict[tuple[str, int, int], list[PublicDocumentManifest]] = {}
    for manifest in manifests:
        key = (manifest.company_symbol, manifest.fiscal_year, manifest.fiscal_quarter)
        grouped.setdefault(key, []).append(manifest)
    return grouped


def _phrases_by_period(phrase_catalog: PhraseCatalog) -> dict[tuple[str, int, int], list[str]]:
    grouped: dict[tuple[str, int, int], list[str]] = {}
    for entry in phrase_catalog.entries:
        key = (entry.company_symbol, entry.fiscal_year, entry.fiscal_quarter)
        grouped.setdefault(key, []).append(entry.phrase)
    return {
        key: list(dict.fromkeys(phrase.strip().lower() for phrase in phrases if phrase.strip()))
        for key, phrases in grouped.items()
    }
