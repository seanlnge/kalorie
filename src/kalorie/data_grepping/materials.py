from pathlib import Path

from kalorie.io.documents import normalize_text
from kalorie.io.public_documents import PublicDocumentManifest


def load_material_snippets(
    manifests: list[PublicDocumentManifest],
    *,
    max_documents: int,
    max_chars_per_document: int,
    company_symbol: str | None = None,
) -> list[str]:
    filtered_manifests = manifests
    if company_symbol is not None:
        normalized_symbol = company_symbol.upper()
        company_rows = [
            manifest for manifest in manifests if manifest.company_symbol == normalized_symbol
        ]
        if company_rows:
            filtered_manifests = company_rows
    snippets: list[str] = []
    for manifest in sorted(
        filtered_manifests,
        key=lambda row: (row.published_at, row.company_symbol),
        reverse=True,
    )[:max_documents]:
        text = normalize_text(Path(manifest.raw_path).read_text(encoding="utf-8"))
        snippets.append(text[:max_chars_per_document])
    return snippets
