import re
from collections import defaultdict

import pandas as pd
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from kalorie.ml.datasets import HistoricalTrainingExample


class HistoricalEvaluationReport(BaseModel):
    split_strategy: str
    sample_count: int = Field(ge=0)
    global_model_version: str
    company_adapted_model_version: str
    global_brier_score: float = Field(ge=0)
    global_log_loss: float = Field(ge=0)
    company_adapted_brier_score: float = Field(ge=0)
    company_adapted_log_loss: float = Field(ge=0)
    company_adaptation_improved_brier: bool


def train_and_evaluate(
    examples: list[HistoricalTrainingExample],
    *,
    test_fraction: float = 0.25,
    company_prior_strength: float = 5.0,
) -> HistoricalEvaluationReport:
    validate_historical_examples(examples)
    if len(examples) < 4:
        raise ValueError("at least 4 examples are required for historical train/eval")
    train_examples, test_examples = _time_split(examples, test_fraction=test_fraction)
    feature_columns = _feature_columns(train_examples)
    global_probabilities = _fit_global_probabilities(train_examples, test_examples, feature_columns)
    adapted_probabilities = _adapt_with_company_priors(
        train_examples=train_examples,
        test_examples=test_examples,
        global_probabilities=global_probabilities,
        prior_strength=company_prior_strength,
    )
    labels = [example.label for example in test_examples]
    global_brier = _brier_score(global_probabilities, labels)
    adapted_brier = _brier_score(adapted_probabilities, labels)

    return HistoricalEvaluationReport(
        split_strategy="time",
        sample_count=len(test_examples),
        global_model_version="historical-logistic-v0",
        company_adapted_model_version="company-shrinkage-v0",
        global_brier_score=global_brier,
        global_log_loss=_safe_log_loss(labels, global_probabilities),
        company_adapted_brier_score=adapted_brier,
        company_adapted_log_loss=_safe_log_loss(labels, adapted_probabilities),
        company_adaptation_improved_brier=adapted_brier < global_brier,
    )


def validate_historical_examples(examples: list[HistoricalTrainingExample]) -> None:
    for example in examples:
        market_company = _market_company_from_id(example.market_id)
        if market_company is None:
            continue
        if market_company != example.company_symbol:
            raise ValueError(
                f"market company {market_company} does not match example company "
                f"{example.company_symbol} for {example.market_id}"
            )


def _market_company_from_id(market_id: str) -> str | None:
    normalized = market_id.upper()
    kalshi_match = re.search(r"KXEARNINGSMENTION([A-Z]+)(?:-|$)", normalized)
    if kalshi_match:
        return kalshi_match.group(1)
    prefix_match = re.match(r"^([A-Z]{1,6})-\d{4}-Q[1-4]-", normalized)
    if prefix_match:
        return prefix_match.group(1)
    return None


def _time_split(
    examples: list[HistoricalTrainingExample],
    *,
    test_fraction: float,
) -> tuple[list[HistoricalTrainingExample], list[HistoricalTrainingExample]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    events: dict[tuple[int, int, str], list[HistoricalTrainingExample]] = defaultdict(list)
    for example in examples:
        event_key = (example.fiscal_year, example.fiscal_quarter, example.company_symbol)
        events[event_key].append(example)
    ordered_event_keys = sorted(events)
    test_event_count = max(1, round(len(ordered_event_keys) * test_fraction))
    train_event_count = len(ordered_event_keys) - test_event_count
    if train_event_count < 1:
        raise ValueError("time split leaves too few training examples")
    train_keys = set(ordered_event_keys[:train_event_count])
    test_keys = set(ordered_event_keys[train_event_count:])
    train_examples = [
        example
        for event_key in ordered_event_keys
        if event_key in train_keys
        for example in events[event_key]
    ]
    test_examples = [
        example
        for event_key in ordered_event_keys
        if event_key in test_keys
        for example in events[event_key]
    ]
    if len(train_examples) < 2:
        raise ValueError("time split leaves too few training examples")
    return train_examples, test_examples


def _feature_columns(examples: list[HistoricalTrainingExample]) -> list[str]:
    columns = sorted({key for example in examples for key in example.features})
    if not columns:
        raise ValueError("training examples must contain at least one feature")
    return columns


def _fit_global_probabilities(
    train_examples: list[HistoricalTrainingExample],
    test_examples: list[HistoricalTrainingExample],
    feature_columns: list[str],
) -> list[float]:
    train_frame = _to_frame(train_examples, feature_columns)
    test_frame = _to_frame(test_examples, feature_columns)
    labels = train_frame["label"].tolist()
    if len(set(labels)) < 2:
        base_rate = _smoothed_rate(sum(labels), len(labels))
        return [base_rate for _ in test_examples]
    model = LogisticRegression(random_state=0, solver="liblinear")
    model.fit(train_frame[feature_columns], train_frame["label"])
    probabilities = model.predict_proba(test_frame[feature_columns])[:, 1]
    return [_clamp(float(value)) for value in probabilities]


def _to_frame(examples: list[HistoricalTrainingExample], columns: list[str]) -> pd.DataFrame:
    rows = []
    for example in examples:
        row = {column: example.features.get(column, 0.0) for column in columns}
        row["label"] = example.label
        rows.append(row)
    return pd.DataFrame(rows)


def _adapt_with_company_priors(
    *,
    train_examples: list[HistoricalTrainingExample],
    test_examples: list[HistoricalTrainingExample],
    global_probabilities: list[float],
    prior_strength: float,
) -> list[float]:
    train_successes = sum(example.label for example in train_examples)
    global_rate = _smoothed_rate(train_successes, len(train_examples))
    company_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for example in train_examples:
        company_counts[example.company_symbol][0] += example.label
        company_counts[example.company_symbol][1] += 1

    adapted: list[float] = []
    for example, global_probability in zip(test_examples, global_probabilities, strict=True):
        successes, total = company_counts[example.company_symbol]
        company_prior = (successes + prior_strength * global_rate) / (total + prior_strength)
        adapted.append(_clamp(0.75 * global_probability + 0.25 * company_prior))
    return adapted


def _smoothed_rate(successes: int, total: int) -> float:
    return _clamp((successes + 1) / (total + 2))


def _brier_score(probabilities: list[float], labels: list[int]) -> float:
    squared_errors = [
        (probability - label) ** 2
        for probability, label in zip(probabilities, labels, strict=True)
    ]
    return round(
        sum(squared_errors) / len(labels),
        6,
    )


def _safe_log_loss(labels: list[int], probabilities: list[float]) -> float:
    return round(float(log_loss(labels, probabilities, labels=[0, 1])), 6)


def _clamp(value: float) -> float:
    return min(0.99, max(0.01, round(value, 6)))
