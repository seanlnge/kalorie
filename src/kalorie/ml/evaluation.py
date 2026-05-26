from typing import Literal

from pydantic import BaseModel, Field

from kalorie.domain.models import MentionLabel, Prediction


class EvaluationReport(BaseModel):
    evaluation_kind: Literal["smoke", "train_eval"]
    sample_count: int = Field(ge=0)
    brier_score: float = Field(ge=0)
    expected_calibration_error: float = Field(ge=0, le=1)
    trained_model: bool
    note: str


def evaluate_binary_predictions(
    predictions: list[Prediction],
    labels: list[MentionLabel],
    evaluation_kind: Literal["smoke", "train_eval"],
    trained_model: bool = False,
) -> EvaluationReport:
    labels_by_target = {label.target_phrase: label for label in labels}
    squared_errors: list[float] = []
    probabilities: list[float] = []
    outcomes: list[float] = []
    for prediction in predictions:
        label = labels_by_target[prediction.target_phrase]
        outcome = 1.0 if label.exact_mentioned else 0.0
        probabilities.append(prediction.probability)
        outcomes.append(outcome)
        squared_errors.append((prediction.probability - outcome) ** 2)

    brier = round(sum(squared_errors) / len(squared_errors), 6) if squared_errors else 0.0
    return EvaluationReport(
        evaluation_kind=evaluation_kind,
        sample_count=len(squared_errors),
        brier_score=brier,
        expected_calibration_error=_expected_calibration_error(probabilities, outcomes),
        trained_model=trained_model,
        note=(
            "Brier score is the primary metric for binary probability labels. "
            "This may be a smoke metric if the evaluation set is small."
        ),
    )


def _expected_calibration_error(
    probabilities: list[float],
    outcomes: list[float],
    *,
    bins: int = 10,
) -> float:
    if not probabilities:
        return 0.0
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must be the same length")
    ece = 0.0
    total = len(probabilities)
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            members = [
                row
                for row in zip(probabilities, outcomes, strict=True)
                if lower <= row[0] <= upper
            ]
        else:
            members = [
                row
                for row in zip(probabilities, outcomes, strict=True)
                if lower <= row[0] < upper
            ]
        if not members:
            continue
        mean_probability = sum(row[0] for row in members) / len(members)
        mean_outcome = sum(row[1] for row in members) / len(members)
        ece += (len(members) / total) * abs(mean_outcome - mean_probability)
    return round(ece, 6)
