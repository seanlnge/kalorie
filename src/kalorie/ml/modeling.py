import pandas as pd
from sklearn.linear_model import LogisticRegression

from kalorie.domain.models import FeatureVector, Prediction


def _clamp_probability(value: float) -> float:
    return min(0.99, max(0.01, round(value, 6)))


class RuleBasedBaseline:
    model_version = "rule-based-v0"

    def __init__(self, base_rate: float = 0.25) -> None:
        self.base_rate = base_rate

    def predict_proba(self, feature_vector: FeatureVector) -> Prediction:
        features = feature_vector.features
        probability = self.base_rate
        reasons = ["base_rate"]

        if features.get("exact_match_count", 0.0) > 0:
            probability += 0.35
            reasons.append("exact_match")
        if features.get("lexical_match_count", 0.0) > 0:
            probability += 0.15
            reasons.append("lexical_match")
        similarity = features.get("max_tfidf_similarity", 0.0)
        if similarity > 0:
            probability += min(0.20, similarity * 0.20)
            reasons.append("tfidf_similarity")
        if features.get("appears_in_headline_or_first_chunk", 0.0) > 0:
            probability += 0.05
            reasons.append("early_document_signal")

        return Prediction(
            target_phrase=feature_vector.target_phrase,
            model_version=self.model_version,
            probability=_clamp_probability(probability),
            reasons=reasons,
        )


class LogisticBaseline:
    model_version = "logistic-baseline-v0"

    def __init__(self, min_training_rows: int = 20) -> None:
        self.min_training_rows = min_training_rows
        self._feature_columns: list[str] = []
        self._model: LogisticRegression | None = None

    def fit(self, frame: pd.DataFrame, label_column: str = "label") -> "LogisticBaseline":
        if len(frame) < self.min_training_rows:
            raise ValueError("not enough training rows for LogisticBaseline")
        self._feature_columns = [column for column in frame.columns if column != label_column]
        self._model = LogisticRegression(random_state=0, solver="liblinear")
        self._model.fit(frame[self._feature_columns], frame[label_column])
        return self

    def predict_proba(self, feature_vector: FeatureVector) -> Prediction:
        if self._model is None:
            raise ValueError("LogisticBaseline must be fit before prediction")
        row = pd.DataFrame(
            [{column: feature_vector.features.get(column, 0.0) for column in self._feature_columns}]
        )
        probability = float(self._model.predict_proba(row)[0][1])
        return Prediction(
            target_phrase=feature_vector.target_phrase,
            model_version=self.model_version,
            probability=_clamp_probability(probability),
            reasons=["logistic_regression"],
        )
