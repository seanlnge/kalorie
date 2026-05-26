from decimal import Decimal

from kalorie.domain.models import MarketSnapshot, PaperTradeComparison, Prediction

PROBABILITY_QUANT = Decimal("0.01")


def implied_yes_probability(snapshot: MarketSnapshot) -> Decimal:
    return snapshot.yes_ask.quantize(PROBABILITY_QUANT)


def implied_no_probability(snapshot: MarketSnapshot) -> Decimal:
    return (Decimal("1") - snapshot.yes_bid).quantize(PROBABILITY_QUANT)


def compare_prediction_to_market(
    prediction: Prediction,
    snapshot: MarketSnapshot,
    min_edge: Decimal = Decimal("0.05"),
) -> PaperTradeComparison:
    model_probability = Decimal(str(prediction.probability)).quantize(PROBABILITY_QUANT)
    yes_probability = implied_yes_probability(snapshot)
    no_probability = implied_no_probability(snapshot)
    yes_edge = (model_probability - yes_probability).quantize(PROBABILITY_QUANT)
    no_edge = ((Decimal("1") - model_probability) - no_probability).quantize(PROBABILITY_QUANT)
    spread = (snapshot.yes_ask - snapshot.yes_bid).quantize(PROBABILITY_QUANT)
    reasons: list[str] = []
    if spread >= Decimal("0.15"):
        reasons.append("wide_spread")

    if yes_edge >= min_edge and yes_edge >= no_edge:
        side = "yes"
        market_probability = yes_probability
        edge = yes_edge
        reasons.append("yes_edge")
    elif no_edge >= min_edge:
        side = "no"
        market_probability = no_probability
        edge = no_edge
        reasons.append("no_edge")
    else:
        side = "skip"
        market_probability = yes_probability
        edge = max(yes_edge, no_edge).quantize(PROBABILITY_QUANT)
        reasons.append("edge_below_threshold")

    return PaperTradeComparison(
        target_phrase=prediction.target_phrase,
        model_probability=model_probability,
        market_probability=market_probability,
        edge=edge,
        side=side,
        reasons=reasons,
        spread=spread,
    )
