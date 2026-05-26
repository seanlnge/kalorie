from kalorie.domain.models import FeatureVector, Prediction
from kalorie.ml.model1 import MentionModelArtifact, predict_model1


def smoke_predict(
    artifact: MentionModelArtifact,
    *,
    company_symbol: str,
    features: list[FeatureVector],
) -> list[Prediction]:
    return [
        predict_model1(artifact, company_symbol=company_symbol, feature_vector=feature)
        for feature in features
    ]
