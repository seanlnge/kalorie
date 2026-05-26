from kalorie.clients.kalshi import KalshiPublicClient, parse_mention_market_contracts
from kalorie.domain.models import MentionMarketContract


def pull_historical_mention_contracts(
    client: KalshiPublicClient,
    *,
    status: str = "closed",
    search: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
) -> tuple[dict, list[MentionMarketContract]]:
    payload = client.get_historical_markets(
        status=status,
        search=search,
        limit=limit,
        cursor=cursor,
    )
    contracts = parse_mention_market_contracts(payload, event_ticker="historical")
    return payload, contracts
