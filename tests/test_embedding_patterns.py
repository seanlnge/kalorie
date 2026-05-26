import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from kalorie2.embedding_patterns import (
    app,
    build_phrase_records,
    cluster_phrase_embeddings,
    cosine_similarity,
    summarize_clusters,
)


def test_cosine_similarity_handles_orthogonal_and_identical_vectors():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0


def test_cluster_phrase_embeddings_groups_connected_semantic_neighbors():
    embeddings = {
        "ai": [1.0, 0.0],
        "artificial intelligence": [0.99, 0.01],
        "inventory": [0.0, 1.0],
    }

    clusters = cluster_phrase_embeddings(embeddings, similarity_threshold=0.95)

    cluster_terms = [set(cluster["terms"]) for cluster in clusters]
    assert {"ai", "artificial intelligence"} in cluster_terms
    assert {"inventory"} in cluster_terms


def test_summarize_clusters_reports_calibration_and_brier_by_semantic_group():
    rows = [
        {
            "word_said": "AI",
            "normalized_word_said": "ai",
            "preclose_yes_mid": "0.80",
            "preclose_yes_bid": "0.78",
            "preclose_yes_ask": "0.82",
            "final_outcome": "yes",
        },
        {
            "word_said": "Artificial Intelligence",
            "normalized_word_said": "artificial intelligence",
            "preclose_yes_mid": "0.70",
            "preclose_yes_bid": "0.68",
            "preclose_yes_ask": "0.72",
            "final_outcome": "no",
        },
        {
            "word_said": "Inventory",
            "normalized_word_said": "inventory",
            "preclose_yes_mid": "0.20",
            "preclose_yes_bid": "0.18",
            "preclose_yes_ask": "0.22",
            "final_outcome": "no",
        },
    ]
    phrase_records = build_phrase_records(rows)
    clusters = [
        {"cluster_id": 1, "terms": ["ai", "artificial intelligence"]},
        {"cluster_id": 2, "terms": ["inventory"]},
    ]

    summaries = summarize_clusters(clusters, phrase_records)

    ai_cluster = summaries[0]
    assert ai_cluster["cluster_id"] == 1
    assert ai_cluster["row_count"] == 2
    assert ai_cluster["outcome_rate"] == 0.5
    assert ai_cluster["mean_probability"] == 0.75
    assert ai_cluster["calibration_gap"] == -0.25
    assert ai_cluster["brier_score"] == 0.265


def test_embedding_pattern_cli_can_use_cached_embeddings(tmp_path: Path):
    input_csv = tmp_path / "markets.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "word_said",
                "normalized_word_said",
                "preclose_yes_mid",
                "preclose_yes_bid",
                "preclose_yes_ask",
                "final_outcome",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "word_said": "AI",
                    "normalized_word_said": "ai",
                    "preclose_yes_mid": "0.80",
                    "preclose_yes_bid": "0.78",
                    "preclose_yes_ask": "0.82",
                    "final_outcome": "yes",
                },
                {
                    "word_said": "Artificial Intelligence",
                    "normalized_word_said": "artificial intelligence",
                    "preclose_yes_mid": "0.70",
                    "preclose_yes_bid": "0.68",
                    "preclose_yes_ask": "0.72",
                    "final_outcome": "no",
                },
            ]
        )
    cache = tmp_path / "embeddings.json"
    cache.write_text(
        json.dumps(
            {
                "model": "test-model",
                "embeddings": {
                    "ai": [1.0, 0.0],
                    "artificial intelligence": [0.99, 0.01],
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "patterns.json"

    result = CliRunner().invoke(
        app,
        [
            str(input_csv),
            "--embeddings-cache",
            str(cache),
            "--json-out",
            str(out),
            "--model",
            "test-model",
            "--similarity-threshold",
            "0.95",
            "--min-cluster-rows",
            "1",
        ],
    )

    assert result.exit_code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["unique_terms"] == 2
    assert report["clusters"][0]["terms"] == ["ai", "artificial intelligence"]
