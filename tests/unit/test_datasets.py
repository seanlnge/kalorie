from datetime import UTC, datetime
from decimal import Decimal

import pytest

from kalorie.domain.models import (
    DocumentChunk,
    MentionMarketContract,
    SourceDocument,
    TargetPhrase,
)
from kalorie.ml.datasets import build_historical_training_examples


class _TemplateEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "traffic": [1.0, 0.0],
            "traffic growth": [1.0, 0.0],
            "Likely discussion of traffic growth.": [1.0, 0.0],
            "Traffic improved, and restaurant-level profit margin expanded.": [1.0, 0.0],
        }
        return [vectors[text] for text in texts]


class _AliasEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "gemini image model": [0.0, 1.0],
            "nano banana": [1.0, 0.0],
            "Nano Banana appeared in the evidence.": [1.0, 0.0],
        }
        return [vectors[text] for text in texts]


def _contract(phrase: str) -> MentionMarketContract:
    return MentionMarketContract(
        venue="kalshi",
        market_id=f"CAVA-{phrase.replace(' ', '-').upper()}",
        event_ticker="CAVA-26Q1",
        title=f"Will CAVA mention {phrase} during earnings?",
        rules_text=f"If {phrase} is said by any CAVA representative.",
        target_phrase=TargetPhrase(phrase=phrase, normalized_phrase=phrase, aliases=[]),
        yes_bid=Decimal("0.40"),
        yes_ask=Decimal("0.45"),
        observed_at=datetime(2026, 5, 19, 14, 30, tzinfo=UTC),
    )


def test_build_historical_training_examples_separates_evidence_from_transcript_labels():
    cutoff = datetime(2026, 5, 19, 20, 0, tzinfo=UTC)
    evidence_doc = SourceDocument(
        source_id="CAVA-2026-Q1-press",
        company_symbol="CAVA",
        document_type="earnings_press_release",
        source_path="press.txt",
        published_at=datetime(2026, 5, 19, 12, 0, tzinfo=UTC),
        content_hash="abc",
    )
    evidence_chunks = [
        DocumentChunk(
            document_id=evidence_doc.source_id,
            chunk_index=0,
            text="Traffic improved, and restaurant-level profit margin expanded.",
            section="headline",
            token_start=0,
            token_end=8,
        )
    ]
    transcript_chunks = [
        DocumentChunk(
            document_id="CAVA-2026-Q1-transcript",
            chunk_index=0,
            text="Our operators discussed traffic but not unrelated topics.",
            section="qa",
            token_start=0,
            token_end=8,
        )
    ]

    examples = build_historical_training_examples(
        company_symbol="CAVA",
        fiscal_year=2026,
        fiscal_quarter=1,
        evidence_cutoff=cutoff,
        contracts=[_contract("traffic"), _contract("robotaxi")],
        evidence_documents=[evidence_doc],
        evidence_chunks=evidence_chunks,
        transcript_chunks=transcript_chunks,
    )

    by_target = {example.target_phrase: example for example in examples}
    assert by_target["traffic"].label == 1
    assert by_target["traffic"].features["exact_match_count"] == 1.0
    assert by_target["traffic"].features["evidence_source_reliability_mean"] == 0.9
    assert by_target["traffic"].features["evidence_news_doc_ratio"] == 0.0
    assert by_target["traffic"].document_ids == ["CAVA-2026-Q1-press"]
    assert by_target["traffic"].market_probability == Decimal("0.45")
    assert by_target["robotaxi"].label == 0


def test_build_historical_training_examples_rejects_post_cutoff_evidence():
    late_doc = SourceDocument(
        source_id="late",
        company_symbol="CAVA",
        document_type="earnings_press_release",
        source_path="late.txt",
        published_at=datetime(2026, 5, 20, tzinfo=UTC),
        content_hash="abc",
    )

    with pytest.raises(ValueError, match="after evidence cutoff"):
        build_historical_training_examples(
            company_symbol="CAVA",
            fiscal_year=2026,
            fiscal_quarter=1,
            evidence_cutoff=datetime(2026, 5, 19, tzinfo=UTC),
            contracts=[_contract("traffic")],
            evidence_documents=[late_doc],
            evidence_chunks=[],
            transcript_chunks=[],
        )


def test_build_historical_training_examples_allows_event_baseline_after_cutoff_with_role():
    cutoff = datetime(2026, 5, 19, 20, 0, tzinfo=UTC)
    baseline_doc = SourceDocument(
        source_id="CAVA-2026-Q1-press",
        company_symbol="CAVA",
        document_type="earnings_press_release",
        source_path="press.txt",
        published_at=datetime(2026, 5, 19, 20, 5, tzinfo=UTC),
        content_hash="abc",
    )
    evidence_chunks = [
        DocumentChunk(
            document_id=baseline_doc.source_id,
            chunk_index=0,
            text="Traffic improved, and restaurant-level profit margin expanded.",
            section="headline",
            token_start=0,
            token_end=8,
        )
    ]
    transcript_chunks = [
        DocumentChunk(
            document_id="CAVA-2026-Q1-transcript",
            chunk_index=0,
            text="We discussed traffic during prepared remarks.",
            section="prepared",
            token_start=0,
            token_end=6,
        )
    ]

    examples = build_historical_training_examples(
        company_symbol="CAVA",
        fiscal_year=2026,
        fiscal_quarter=1,
        evidence_cutoff=cutoff,
        contracts=[_contract("traffic")],
        evidence_documents=[baseline_doc],
        evidence_chunks=evidence_chunks,
        transcript_chunks=transcript_chunks,
        evidence_document_roles={baseline_doc.source_id: "event_baseline"},
    )

    assert len(examples) == 1
    assert examples[0].document_ids == ["CAVA-2026-Q1-press"]
    assert examples[0].evidence_document_roles == {
        "CAVA-2026-Q1-press": "event_baseline"
    }


def test_build_historical_training_examples_ignores_chunks_for_unselected_documents():
    cutoff = datetime(2026, 5, 19, 20, 0, tzinfo=UTC)
    selected_doc = SourceDocument(
        source_id="CAVA-2026-Q1-press",
        company_symbol="CAVA",
        document_type="earnings_press_release",
        source_path="press.txt",
        published_at=datetime(2026, 5, 19, 12, 0, tzinfo=UTC),
        content_hash="abc",
    )
    evidence_chunks = [
        DocumentChunk(
            document_id=selected_doc.source_id,
            chunk_index=0,
            text="Restaurant-level profit margin expanded.",
            section="headline",
            token_start=0,
            token_end=5,
        ),
        DocumentChunk(
            document_id="CAVA-2026-Q1-late-news",
            chunk_index=0,
            text="Traffic was leaked by a late news article.",
            section="headline",
            token_start=0,
            token_end=7,
        ),
    ]
    transcript_chunks = [
        DocumentChunk(
            document_id="CAVA-2026-Q1-transcript",
            chunk_index=0,
            text="We discussed traffic during prepared remarks.",
            section="prepared",
            token_start=0,
            token_end=6,
        )
    ]

    examples = build_historical_training_examples(
        company_symbol="CAVA",
        fiscal_year=2026,
        fiscal_quarter=1,
        evidence_cutoff=cutoff,
        contracts=[_contract("traffic")],
        evidence_documents=[selected_doc],
        evidence_chunks=evidence_chunks,
        transcript_chunks=transcript_chunks,
        evidence_document_roles={selected_doc.source_id: "event_baseline"},
    )

    assert examples[0].document_ids == ["CAVA-2026-Q1-press"]
    assert examples[0].features["exact_match_count"] == 0.0


def test_build_historical_training_examples_adds_template_embedding_features():
    cutoff = datetime(2026, 5, 19, 20, 0, tzinfo=UTC)
    evidence_doc = SourceDocument(
        source_id="CAVA-2026-Q1-press",
        company_symbol="CAVA",
        document_type="earnings_press_release",
        source_path="press.txt",
        published_at=datetime(2026, 5, 19, 12, 0, tzinfo=UTC),
        content_hash="abc",
    )
    evidence_chunks = [
        DocumentChunk(
            document_id=evidence_doc.source_id,
            chunk_index=0,
            text="Traffic improved, and restaurant-level profit margin expanded.",
            section="headline",
            token_start=0,
            token_end=8,
        )
    ]
    transcript_chunks = [
        DocumentChunk(
            document_id="CAVA-2026-Q1-transcript",
            chunk_index=0,
            text="We saw traffic strength in the quarter.",
            section="qa",
            token_start=0,
            token_end=8,
        )
    ]

    examples = build_historical_training_examples(
        company_symbol="CAVA",
        fiscal_year=2026,
        fiscal_quarter=1,
        evidence_cutoff=cutoff,
        contracts=[_contract("traffic")],
        evidence_documents=[evidence_doc],
        evidence_chunks=evidence_chunks,
        transcript_chunks=transcript_chunks,
        template_phrases_by_target={"traffic": ["traffic growth"]},
        embedding_provider=_TemplateEmbeddingProvider(),
    )

    features = examples[0].features
    assert features["template_phrase_count"] == 1.0
    assert features["max_template_embedding_similarity"] == 1.0


def test_build_historical_training_examples_adds_scenario_embedding_features():
    cutoff = datetime(2026, 5, 19, 20, 0, tzinfo=UTC)
    evidence_doc = SourceDocument(
        source_id="CAVA-2026-Q1-press",
        company_symbol="CAVA",
        document_type="earnings_press_release",
        source_path="press.txt",
        published_at=datetime(2026, 5, 19, 12, 0, tzinfo=UTC),
        content_hash="abc",
    )
    evidence_chunks = [
        DocumentChunk(
            document_id=evidence_doc.source_id,
            chunk_index=0,
            text="Traffic improved, and restaurant-level profit margin expanded.",
            section="headline",
            token_start=0,
            token_end=8,
        )
    ]
    transcript_chunks = [
        DocumentChunk(
            document_id="CAVA-2026-Q1-transcript",
            chunk_index=0,
            text="We saw traffic strength in the quarter.",
            section="qa",
            token_start=0,
            token_end=8,
        )
    ]

    examples = build_historical_training_examples(
        company_symbol="CAVA",
        fiscal_year=2026,
        fiscal_quarter=1,
        evidence_cutoff=cutoff,
        contracts=[_contract("traffic")],
        evidence_documents=[evidence_doc],
        evidence_chunks=evidence_chunks,
        transcript_chunks=transcript_chunks,
        scenario_texts=["Likely discussion of traffic growth."],
        embedding_provider=_TemplateEmbeddingProvider(),
    )

    features = examples[0].features
    assert features["scenario_text_count"] == 1.0
    assert features["max_scenario_embedding_similarity"] == 1.0


def test_build_historical_training_examples_adds_alias_embedding_features():
    cutoff = datetime(2026, 5, 19, 20, 0, tzinfo=UTC)
    evidence_doc = SourceDocument(
        source_id="GOOGL-2026-Q1-press",
        company_symbol="GOOGL",
        document_type="earnings_press_release",
        source_path="press.txt",
        published_at=datetime(2026, 5, 19, 12, 0, tzinfo=UTC),
        content_hash="abc",
    )
    evidence_chunks = [
        DocumentChunk(
            document_id=evidence_doc.source_id,
            chunk_index=0,
            text="Nano Banana appeared in the evidence.",
            section="headline",
            token_start=0,
            token_end=6,
        )
    ]
    transcript_chunks = [
        DocumentChunk(
            document_id="GOOGL-2026-Q1-transcript",
            chunk_index=0,
            text="We discussed gemini image model progress.",
            section="prepared",
            token_start=0,
            token_end=6,
        )
    ]
    contract = _contract("gemini image model")
    contract.target_phrase.aliases.append("nano banana")

    examples = build_historical_training_examples(
        company_symbol="GOOGL",
        fiscal_year=2026,
        fiscal_quarter=1,
        evidence_cutoff=cutoff,
        contracts=[contract],
        evidence_documents=[evidence_doc],
        evidence_chunks=evidence_chunks,
        transcript_chunks=transcript_chunks,
        embedding_provider=_AliasEmbeddingProvider(),
    )

    assert examples[0].features["alias_max_embedding_similarity"] == 1.0


def test_build_historical_training_examples_includes_news_reliability_features():
    cutoff = datetime(2026, 5, 19, 20, 0, tzinfo=UTC)
    news_doc = SourceDocument(
        source_id="NVDA-2026-Q1-news",
        company_symbol="NVDA",
        document_type="news_article_opinion_reliability_072",
        source_path="news.txt",
        published_at=datetime(2026, 5, 19, 11, 0, tzinfo=UTC),
        content_hash="abc",
    )
    evidence_chunks = [
        DocumentChunk(
            document_id=news_doc.source_id,
            chunk_index=0,
            text="Analysts shared an earnings opinion on NVIDIA data center trends.",
            section="headline",
            token_start=0,
            token_end=9,
        )
    ]
    transcript_chunks = [
        DocumentChunk(
            document_id="NVDA-2026-Q1-transcript",
            chunk_index=0,
            text="NVIDIA discussed data center demand and AI factory growth.",
            section="qa",
            token_start=0,
            token_end=10,
        )
    ]

    examples = build_historical_training_examples(
        company_symbol="NVDA",
        fiscal_year=2026,
        fiscal_quarter=1,
        evidence_cutoff=cutoff,
        contracts=[_contract("data center")],
        evidence_documents=[news_doc],
        evidence_chunks=evidence_chunks,
        transcript_chunks=transcript_chunks,
    )
    features = examples[0].features
    assert features["evidence_source_reliability_mean"] == 0.72
    assert features["evidence_source_reliability_min"] == 0.72
    assert features["evidence_news_doc_ratio"] == 1.0


def test_build_historical_training_examples_merges_transcript_recurrence_features():
    cutoff = datetime(2026, 5, 19, 20, 0, tzinfo=UTC)
    evidence_doc = SourceDocument(
        source_id="CAVA-2026-Q1-press",
        company_symbol="CAVA",
        document_type="earnings_press_release",
        source_path="press.txt",
        published_at=datetime(2026, 5, 19, 12, 0, tzinfo=UTC),
        content_hash="abc",
    )
    evidence_chunks = [
        DocumentChunk(
            document_id=evidence_doc.source_id,
            chunk_index=0,
            text="Traffic improved.",
            section="headline",
            token_start=0,
            token_end=2,
        )
    ]
    transcript_chunks = [
        DocumentChunk(
            document_id="CAVA-2026-Q1-transcript",
            chunk_index=0,
            text="We discussed traffic during prepared remarks.",
            section="prepared",
            token_start=0,
            token_end=6,
        )
    ]

    examples = build_historical_training_examples(
        company_symbol="CAVA",
        fiscal_year=2026,
        fiscal_quarter=1,
        evidence_cutoff=cutoff,
        contracts=[_contract("traffic")],
        evidence_documents=[evidence_doc],
        evidence_chunks=evidence_chunks,
        transcript_chunks=transcript_chunks,
        transcript_recurrence_features_by_target={
            "traffic": {
                "prior_call_count": 3.0,
                "prior_mention_count": 2.0,
                "prior_mention_rate": 0.666667,
                "prior_recent_mention_binary": 1.0,
                "prior_mention_streak": 2.0,
            }
        },
    )

    features = examples[0].features
    assert features["prior_call_count"] == 3.0
    assert features["prior_mention_count"] == 2.0
    assert features["prior_mention_rate"] == 0.666667
    assert features["prior_recent_mention_binary"] == 1.0
    assert features["prior_mention_streak"] == 2.0
