import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import httpx
import typer

app = typer.Typer(help="Find semantic patterns in Kalshi earnings mention words.")

OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return round(numerator / (left_norm * right_norm), 12)


def build_phrase_records(rows: list[dict[str, str]]) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        term = row.get("normalized_word_said") or row.get("word_said", "").strip().lower()
        if not term:
            continue
        probability = float(row["preclose_yes_mid"])
        outcome = _parse_outcome(row["final_outcome"])
        records[term].append(
            {
                "term": term,
                "display": row.get("word_said") or term,
                "probability": probability,
                "yes_bid": float(row["preclose_yes_bid"]),
                "yes_ask": float(row["preclose_yes_ask"]),
                "outcome": outcome,
                "squared_error": (probability - outcome) ** 2,
                "market_ticker": row.get("market_ticker", ""),
                "event_ticker": row.get("event_ticker", ""),
            }
        )
    return dict(records)


def cluster_phrase_embeddings(
    embeddings: dict[str, list[float]],
    *,
    similarity_threshold: float = 0.82,
) -> list[dict]:
    terms = sorted(embeddings)
    parent = {term: term for term in terms}

    def find(term: str) -> str:
        while parent[term] != term:
            parent[term] = parent[parent[term]]
            term = parent[term]
        return term

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, left in enumerate(terms):
        for right in terms[index + 1 :]:
            if cosine_similarity(embeddings[left], embeddings[right]) >= similarity_threshold:
                union(left, right)

    groups: dict[str, list[str]] = defaultdict(list)
    for term in terms:
        groups[find(term)].append(term)
    clusters = [
        {"cluster_id": cluster_id, "terms": sorted(group)}
        for cluster_id, group in enumerate(
            sorted(groups.values(), key=lambda values: (-len(values), values[0])),
            start=1,
        )
    ]
    return clusters


def summarize_clusters(
    clusters: list[dict],
    phrase_records: dict[str, list[dict]],
) -> list[dict]:
    summaries = []
    for cluster in clusters:
        rows = [
            record
            for term in cluster["terms"]
            for record in phrase_records.get(term, [])
        ]
        if not rows:
            continue
        count = len(rows)
        mean_probability = sum(row["probability"] for row in rows) / count
        outcome_rate = sum(row["outcome"] for row in rows) / count
        brier_score = sum(row["squared_error"] for row in rows) / count
        summaries.append(
            {
                "cluster_id": cluster["cluster_id"],
                "terms": cluster["terms"],
                "representative_terms": cluster["terms"][:8],
                "unique_terms": len(cluster["terms"]),
                "row_count": count,
                "mean_probability": round(mean_probability, 6),
                "outcome_rate": round(outcome_rate, 6),
                "calibration_gap": round(outcome_rate - mean_probability, 6),
                "brier_score": round(brier_score, 6),
                "mean_yes_bid": round(sum(row["yes_bid"] for row in rows) / count, 6),
                "mean_yes_ask": round(sum(row["yes_ask"] for row in rows) / count, 6),
                "yes_count": sum(row["outcome"] for row in rows),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (abs(row["calibration_gap"]), row["row_count"]),
        reverse=True,
    )


@app.command("analyze")
def analyze_command(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    embeddings_cache: Annotated[Path, typer.Option()] = Path(
        "artifacts/embeddings/word-embeddings.json"
    ),
    json_out: Annotated[Path | None, typer.Option()] = None,
    csv_out: Annotated[Path | None, typer.Option()] = None,
    model: Annotated[str, typer.Option()] = "text-embedding-3-small",
    similarity_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.82,
    batch_size: Annotated[int, typer.Option(min=1, max=2048)] = 100,
    min_cluster_rows: Annotated[int, typer.Option(min=1)] = 3,
    env_file: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    if env_file is not None:
        _load_env_file(env_file)
    report = analyze_csv(
        input_csv,
        embeddings_cache=embeddings_cache,
        model=model,
        similarity_threshold=similarity_threshold,
        batch_size=batch_size,
        min_cluster_rows=min_cluster_rows,
    )
    if json_out is not None:
        _write_json(json_out, report)
    if csv_out is not None:
        _write_clusters_csv(csv_out, report["clusters"])
    typer.echo(
        f"Rows: {report['summary']['rows']} | "
        f"Unique terms: {report['summary']['unique_terms']} | "
        f"Clusters: {report['summary']['clusters']} | "
        f"Report clusters: {report['summary']['reported_clusters']}"
    )


def analyze_csv(
    input_csv: Path,
    *,
    embeddings_cache: Path,
    model: str,
    similarity_threshold: float,
    batch_size: int = 100,
    min_cluster_rows: int = 3,
) -> dict:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    phrase_records = build_phrase_records(rows)
    embeddings = load_or_create_embeddings(
        sorted(phrase_records),
        cache_path=embeddings_cache,
        model=model,
        batch_size=batch_size,
    )
    clusters = cluster_phrase_embeddings(embeddings, similarity_threshold=similarity_threshold)
    summaries = [
        cluster
        for cluster in summarize_clusters(clusters, phrase_records)
        if cluster["row_count"] >= min_cluster_rows
    ]
    return {
        "summary": {
            "rows": len(rows),
            "unique_terms": len(phrase_records),
            "embedded_terms": len(embeddings),
            "clusters": len(clusters),
            "reported_clusters": len(summaries),
            "similarity_threshold": similarity_threshold,
            "min_cluster_rows": min_cluster_rows,
            "model": model,
        },
        "clusters": summaries,
    }


def load_or_create_embeddings(
    terms: list[str],
    *,
    cache_path: Path,
    model: str,
    batch_size: int,
) -> dict[str, list[float]]:
    cache = _read_embedding_cache(cache_path)
    if cache.get("model") != model:
        cache = {"model": model, "embeddings": {}}
    embeddings: dict[str, list[float]] = dict(cache.get("embeddings", {}))
    missing_terms = [term for term in terms if term not in embeddings]
    if missing_terms:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required to create missing embeddings")
        for start in range(0, len(missing_terms), batch_size):
            batch = missing_terms[start : start + batch_size]
            for term, embedding in zip(
                batch,
                _fetch_openai_embeddings(batch, api_key=api_key, model=model),
                strict=True,
            ):
                embeddings[term] = embedding
        _write_json(cache_path, {"model": model, "embeddings": embeddings})
    return {term: embeddings[term] for term in terms}


def _fetch_openai_embeddings(
    terms: list[str],
    *,
    api_key: str,
    model: str,
) -> list[list[float]]:
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            OPENAI_EMBEDDINGS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "input": terms},
        )
        response.raise_for_status()
    data = response.json()["data"]
    return [row["embedding"] for row in sorted(data, key=lambda row: row["index"])]


def _read_embedding_cache(path: Path) -> dict:
    if not path.exists():
        return {"model": None, "embeddings": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_outcome(value: str) -> int:
    normalized = value.strip().lower()
    if normalized == "yes":
        return 1
    if normalized == "no":
        return 0
    raise ValueError(f"Unsupported final_outcome: {value}")


def _load_env_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_clusters_csv(path: Path, clusters: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cluster_id",
        "unique_terms",
        "row_count",
        "mean_probability",
        "outcome_rate",
        "calibration_gap",
        "brier_score",
        "mean_yes_bid",
        "mean_yes_ask",
        "yes_count",
        "representative_terms",
        "terms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cluster in clusters:
            row = dict(cluster)
            row["representative_terms"] = "; ".join(cluster["representative_terms"])
            row["terms"] = "; ".join(cluster["terms"])
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    app()
