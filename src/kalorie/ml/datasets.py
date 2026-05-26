import re
from datetime import datetime
from decimal import Decimal

from pydantic import Field, field_validator

from kalorie.domain.models import (
    DocumentChunk,
    KalorieModel,
    MentionMarketContract,
    SourceDocument,
    TargetPhrase,
)
from kalorie.ml.embeddings import EmbeddingProvider
from kalorie.ml.features import (
    extract_alias_embedding_feature_vectors,
    extract_feature_vectors,
    extract_scenario_embedding_feature_vectors,
    extract_template_embedding_feature_vectors,
)
from kalorie.ml.labeling import label_document_chunks
from kalorie.ml.priors import phrase_category_features


class HistoricalTrainingExample(KalorieModel):
    company_symbol: str
    fiscal_year: int
    fiscal_quarter: int = Field(ge=1, le=4)
    evidence_cutoff: datetime
    market_id: str
    target_phrase: str
    label: int = Field(ge=0, le=1)
    features: dict[str, float]
    document_ids: list[str]
    market_probability: Decimal
    market_venue: str = "unknown"
    event_ticker: str | None = None
    evidence_document_roles: dict[str, str] = Field(default_factory=dict)

    @field_validator("company_symbol")
    @classmethod
    def normalize_company_symbol(cls, value: str) -> str:
        return value.upper()


def build_historical_training_examples(
    *,
    company_symbol: str,
    fiscal_year: int,
    fiscal_quarter: int,
    evidence_cutoff: datetime,
    contracts: list[MentionMarketContract],
    evidence_documents: list[SourceDocument],
    evidence_chunks: list[DocumentChunk],
    transcript_chunks: list[DocumentChunk],
    template_phrases_by_target: dict[str, list[str]] | None = None,
    scenario_texts: list[str] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    evidence_document_roles: dict[str, str] | None = None,
    transcript_recurrence_features_by_target: dict[str, dict[str, float]] | None = None,
) -> list[HistoricalTrainingExample]:
    evidence_document_roles = evidence_document_roles or {}
    transcript_recurrence_features_by_target = transcript_recurrence_features_by_target or {}
    document_ids = []
    for document in evidence_documents:
        role = evidence_document_roles.get(document.source_id, "time_sensitive")
        if document.published_at > evidence_cutoff and role != "event_baseline":
            raise ValueError(f"Document {document.source_id} is after evidence cutoff")
        document_ids.append(document.source_id)

    targets = [contract.target_phrase for contract in contracts]
    selected_document_ids = set(document_ids)
    evidence_chunks = [
        chunk for chunk in evidence_chunks if chunk.document_id in selected_document_ids
    ]
    evidence_labels = label_document_chunks(evidence_chunks, targets)
    settlement_labels = {
        label.target_phrase: label
        for label in label_document_chunks(
            transcript_chunks,
            targets,
            entity_scope="company_employee",
        )
    }
    features_by_target = {
        feature.target_phrase: feature
        for feature in extract_feature_vectors(evidence_chunks, targets, evidence_labels)
    }
    template_features_by_target: dict[str, dict[str, float]] = {}
    if template_phrases_by_target and embedding_provider is not None:
        template_features_by_target = {
            feature.target_phrase: feature.features
            for feature in extract_template_embedding_feature_vectors(
                evidence_chunks,
                targets,
                template_phrases_by_target,
                embedding_provider,
            )
        }
    scenario_features_by_target: dict[str, dict[str, float]] = {}
    if scenario_texts and embedding_provider is not None:
        scenario_features_by_target = {
            feature.target_phrase: feature.features
            for feature in extract_scenario_embedding_feature_vectors(
                chunks=evidence_chunks,
                targets=targets,
                scenario_texts=scenario_texts,
                provider=embedding_provider,
            )
        }
    alias_embedding_features_by_target: dict[str, dict[str, float]] = {}
    if embedding_provider is not None:
        alias_embedding_features_by_target = {
            feature.target_phrase: feature.features
            for feature in extract_alias_embedding_feature_vectors(
                evidence_chunks,
                targets,
                embedding_provider,
            )
        }
    reliability_features = _evidence_reliability_features(evidence_documents)

    examples: list[HistoricalTrainingExample] = []
    for contract in contracts:
        target = _normalized_target(contract.target_phrase)
        settlement_label = settlement_labels[target.normalized_phrase]
        examples.append(
            HistoricalTrainingExample(
                company_symbol=company_symbol,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                evidence_cutoff=evidence_cutoff,
                market_id=contract.market_id,
                target_phrase=target.normalized_phrase,
                label=1 if settlement_label.exact_mentioned else 0,
                features={
                    **features_by_target[target.normalized_phrase].features,
                    **phrase_category_features(target.normalized_phrase),
                    **template_features_by_target.get(target.normalized_phrase, {}),
                    **scenario_features_by_target.get(target.normalized_phrase, {}),
                    **alias_embedding_features_by_target.get(target.normalized_phrase, {}),
                    **reliability_features,
                    **transcript_recurrence_features_by_target.get(
                        target.normalized_phrase,
                        {},
                    ),
                },
                document_ids=document_ids,
                market_probability=contract.yes_ask,
                market_venue=contract.venue,
                event_ticker=contract.event_ticker,
                evidence_document_roles={
                    document_id: evidence_document_roles.get(document_id, "time_sensitive")
                    for document_id in document_ids
                },
            )
        )
    return examples


def _evidence_reliability_features(evidence_documents: list[SourceDocument]) -> dict[str, float]:
    if not evidence_documents:
        return {
            "evidence_source_reliability_mean": 0.0,
            "evidence_source_reliability_min": 0.0,
            "evidence_news_doc_ratio": 0.0,
        }
    scores = [_source_reliability_score(document.document_type) for document in evidence_documents]
    news_count = sum(
        document.document_type.startswith("news_article") for document in evidence_documents
    )
    return {
        "evidence_source_reliability_mean": round(sum(scores) / len(scores), 6),
        "evidence_source_reliability_min": round(min(scores), 6),
        "evidence_news_doc_ratio": round(news_count / len(evidence_documents), 6),
    }


def _source_reliability_score(document_type: str) -> float:
    if document_type.startswith("news_article"):
        encoded = _parse_reliability_from_document_type(document_type)
        if encoded is not None:
            return encoded
        return 0.65
    if document_type.startswith("sec_"):
        return 0.95
    if document_type.endswith("press_release") or document_type == "earnings_press_release":
        return 0.9
    return 0.8


def _parse_reliability_from_document_type(document_type: str) -> float | None:
    match = re.search(r"_reliability_(\d{1,3})$", document_type)
    if not match:
        return None
    value = int(match.group(1))
    return max(0.0, min(1.0, value / 100.0))


def _normalized_target(target: TargetPhrase) -> TargetPhrase:
    return target.model_copy(update={"normalized_phrase": target.normalized_phrase.lower()})
