import json
from pathlib import Path

from kalorie.domain.models import TargetPhrase
from kalorie.ml.labeling import normalize_phrase

AliasManifest = dict[str, list[str]]


def load_alias_manifest(path: Path) -> AliasManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("alias manifest must be a JSON object")
    manifest: AliasManifest = {}
    for phrase, aliases in raw.items():
        if not isinstance(phrase, str) or not isinstance(aliases, list):
            raise ValueError("alias manifest entries must map phrases to alias lists")
        manifest[normalize_phrase(phrase)] = _normalize_aliases(
            str(alias) for alias in aliases
        )
    return manifest


def resolve_target_aliases(
    target: TargetPhrase,
    *,
    manifest_aliases: AliasManifest | None = None,
) -> list[str]:
    normalized_target = normalize_phrase(target.normalized_phrase)
    aliases = [
        *target.aliases,
        *((manifest_aliases or {}).get(normalized_target, [])),
    ]
    return [
        alias
        for alias in _normalize_aliases(aliases)
        if alias != normalized_target
    ]


def _normalize_aliases(aliases: object) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        value = normalize_phrase(str(alias))
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized
