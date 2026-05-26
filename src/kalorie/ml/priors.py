from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from kalorie.ml.datasets import HistoricalTrainingExample

PHRASE_CATEGORIES = [
    "alias",
    "macro",
    "competitor",
    "codename_or_product",
    "multiword",
    "generic",
]

MACRO_TERMS = {
    "china",
    "inflation",
    "iran",
    "oil",
    "rate cut",
    "recession",
    "tariff",
}
COMPETITOR_TERMS = {
    "anthropic",
    "deepmind",
    "gemini",
    "nvidia",
    "openai",
}
CODENAME_TERMS = {
    "alexa+",
    "circle to search",
    "fairwater",
    "ironwood",
    "liquid glass",
    "nano banana",
    "project kuiper",
    "rufus",
    "siri",
    "wiz",
}


class PriorBucket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positive_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    rate: float = Field(ge=0.0, le=1.0)


class HierarchicalPriorModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_global: dict[str, PriorBucket] = Field(default_factory=dict)
    company_target: dict[str, PriorBucket] = Field(default_factory=dict)
    company_category: dict[str, PriorBucket] = Field(default_factory=dict)
    company_global: dict[str, PriorBucket] = Field(default_factory=dict)
    category_global: dict[str, PriorBucket] = Field(default_factory=dict)

    def features_for(self, *, company_symbol: str, target_phrase: str) -> dict[str, float]:
        category = phrase_category(target_phrase)
        company = company_symbol.upper()
        return {
            **_bucket_features(
                "prior_target_global",
                self.target_global.get(_target_key(target_phrase)),
            ),
            **_bucket_features(
                "prior_company_target",
                self.company_target.get(_company_target_key(company, target_phrase)),
            ),
            **_bucket_features(
                "prior_company_category",
                self.company_category.get(_company_category_key(company, category)),
            ),
            **_bucket_features(
                "prior_company_global",
                self.company_global.get(company),
            ),
            **_bucket_features(
                "prior_category_global",
                self.category_global.get(category),
            ),
        }


@dataclass
class _Counter:
    positive_count: int = 0
    sample_count: int = 0

    def add(self, label: int) -> None:
        self.positive_count += int(label)
        self.sample_count += 1

    def bucket(self) -> PriorBucket:
        return PriorBucket(
            positive_count=self.positive_count,
            sample_count=self.sample_count,
            rate=self.positive_count / self.sample_count if self.sample_count else 0.0,
        )


def phrase_category(phrase: str) -> str:
    normalized = phrase.lower().strip()
    if "/" in normalized:
        return "alias"
    if normalized in MACRO_TERMS:
        return "macro"
    if normalized in COMPETITOR_TERMS:
        return "competitor"
    if normalized in CODENAME_TERMS:
        return "codename_or_product"
    if len(normalized.split()) >= 2:
        return "multiword"
    return "generic"


def phrase_category_features(phrase: str) -> dict[str, float]:
    category = phrase_category(phrase)
    return {
        f"phrase_category_{candidate}": 1.0 if candidate == category else 0.0
        for candidate in PHRASE_CATEGORIES
    }


def market_prior_features(probability: float | None) -> dict[str, float]:
    probability = 0.5 if probability is None else probability
    clipped = min(0.999999, max(0.000001, float(probability)))
    return {
        "market_probability": clipped,
        "market_logit": math.log(clipped / (1.0 - clipped)),
    }


def fit_hierarchical_prior_model(
    examples: list[HistoricalTrainingExample],
) -> HierarchicalPriorModel:
    target_global: defaultdict[str, _Counter] = defaultdict(_Counter)
    company_target: defaultdict[str, _Counter] = defaultdict(_Counter)
    company_category: defaultdict[str, _Counter] = defaultdict(_Counter)
    company_global: defaultdict[str, _Counter] = defaultdict(_Counter)
    category_global: defaultdict[str, _Counter] = defaultdict(_Counter)
    for example in examples:
        category = phrase_category(example.target_phrase)
        company = example.company_symbol.upper()
        target_global[_target_key(example.target_phrase)].add(example.label)
        company_target[_company_target_key(company, example.target_phrase)].add(example.label)
        company_category[_company_category_key(company, category)].add(example.label)
        company_global[company].add(example.label)
        category_global[category].add(example.label)
    return HierarchicalPriorModel(
        target_global={key: value.bucket() for key, value in target_global.items()},
        company_target={key: value.bucket() for key, value in company_target.items()},
        company_category={key: value.bucket() for key, value in company_category.items()},
        company_global={key: value.bucket() for key, value in company_global.items()},
        category_global={key: value.bucket() for key, value in category_global.items()},
    )


def add_historical_prior_features(
    examples: list[HistoricalTrainingExample],
) -> list[HistoricalTrainingExample]:
    counters = _PriorCounters()
    enriched: list[HistoricalTrainingExample] = []
    for example in sorted(
        examples,
        key=lambda item: (
            item.evidence_cutoff,
            item.company_symbol,
            item.fiscal_year,
            item.fiscal_quarter,
            item.target_phrase,
        ),
    ):
        features = {
            **example.features,
            **phrase_category_features(example.target_phrase),
            **counters.features_for(example),
        }
        enriched.append(example.model_copy(update={"features": features}))
        counters.add(example)
    return enriched


class _PriorCounters:
    def __init__(self) -> None:
        self.target_global: defaultdict[str, _Counter] = defaultdict(_Counter)
        self.company_target: defaultdict[str, _Counter] = defaultdict(_Counter)
        self.company_category: defaultdict[str, _Counter] = defaultdict(_Counter)
        self.company_global: defaultdict[str, _Counter] = defaultdict(_Counter)
        self.category_global: defaultdict[str, _Counter] = defaultdict(_Counter)

    def add(self, example: HistoricalTrainingExample) -> None:
        category = phrase_category(example.target_phrase)
        company = example.company_symbol.upper()
        self.target_global[_target_key(example.target_phrase)].add(example.label)
        self.company_target[_company_target_key(company, example.target_phrase)].add(example.label)
        self.company_category[_company_category_key(company, category)].add(example.label)
        self.company_global[company].add(example.label)
        self.category_global[category].add(example.label)

    def features_for(self, example: HistoricalTrainingExample) -> dict[str, float]:
        category = phrase_category(example.target_phrase)
        company = example.company_symbol.upper()
        return {
            **_bucket_features(
                "prior_target_global",
                self.target_global.get(_target_key(example.target_phrase)),
            ),
            **_bucket_features(
                "prior_company_target",
                self.company_target.get(_company_target_key(company, example.target_phrase)),
            ),
            **_bucket_features(
                "prior_company_category",
                self.company_category.get(_company_category_key(company, category)),
            ),
            **_bucket_features("prior_company_global", self.company_global.get(company)),
            **_bucket_features("prior_category_global", self.category_global.get(category)),
        }


def _bucket_features(prefix: str, bucket: PriorBucket | _Counter | None) -> dict[str, float]:
    if bucket is None:
        return {
            f"{prefix}_rate": 0.0,
            f"{prefix}_log_count": 0.0,
        }
    sample_count = bucket.sample_count
    rate = bucket.rate if isinstance(bucket, PriorBucket) else bucket.bucket().rate
    return {
        f"{prefix}_rate": round(rate, 6),
        f"{prefix}_log_count": round(math.log1p(sample_count), 6),
    }


def _target_key(target_phrase: str) -> str:
    return target_phrase.lower().strip()


def _company_target_key(company_symbol: str, target_phrase: str) -> str:
    return f"{company_symbol.upper()}::{_target_key(target_phrase)}"


def _company_category_key(company_symbol: str, category: str) -> str:
    return f"{company_symbol.upper()}::{category}"
