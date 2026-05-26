import json

from kalorie.domain.models import TargetPhrase
from kalorie.ml.aliases import load_alias_manifest, resolve_target_aliases


def test_resolve_target_aliases_merges_and_normalizes_manifest_aliases():
    target = TargetPhrase(
        phrase="Gemini Image Model",
        normalized_phrase="gemini image model",
        aliases=["Nano Banana"],
    )
    aliases = resolve_target_aliases(
        target,
        manifest_aliases={
            "gemini image model": ["nano banana", "Image generation model"],
            "other": ["ignored"],
        },
    )

    assert aliases == ["nano banana", "image generation model"]


def test_load_alias_manifest_reads_phrase_to_aliases_mapping(tmp_path):
    path = tmp_path / "aliases.json"
    path.write_text(
        json.dumps(
            {
                "gemini image model": ["Nano Banana", "image generation model"],
                "fairwater": ["Microsoft Fairwater"],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_alias_manifest(path)

    assert manifest == {
        "gemini image model": ["nano banana", "image generation model"],
        "fairwater": ["microsoft fairwater"],
    }
