from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import httpx
import typer
from pydantic import BaseModel, ConfigDict, Field

from kalorie2.collector import (
    EARNINGS_MENTION_PREFIX,
    KalshiMentionClient,
    classify_market_category,
    extract_target_phrase,
    is_earnings_mention_market,
)
from kalorie2.saved_models import (
    CachedRuntimeSavedModelScorer,
    SavedModelRegistry,
    normalize_score_payload,
)

app = typer.Typer(help="Poll active Kalshi mention markets with a saved model.")

TradeSide = Literal["YES", "NO", "NONE"]


class MarketPollerBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ActiveMarketRow(MarketPollerBase):
    market_ticker: str
    event_ticker: str
    series_ticker: str
    event_title: str
    market_title: str
    target_phrase: str
    yes_bid: float = Field(ge=0.0, le=1.0)
    yes_ask: float = Field(ge=0.0, le=1.0)
    yes_mid: float = Field(ge=0.0, le=1.0)
    volume: int = 0

    def to_runtime_row(self) -> dict[str, str]:
        now = datetime.now(tz=UTC)
        return {
            "market_ticker": self.market_ticker,
            "event_ticker": self.event_ticker,
            "series_ticker": self.series_ticker,
            "market_category": classify_market_category(
                series_ticker=self.series_ticker,
                event_title=self.event_title,
                market_title=self.market_title,
            ),
            "event_phrase": self.event_title,
            "market_name": self.market_title,
            "word_said": self.target_phrase,
            "normalized_word_said": self.target_phrase.lower(),
            "final_outcome": "no",
            "status": "open",
            "close_time": now.isoformat(),
            "snapshot_target_time": now.isoformat(),
            "preclose_yes_bid": _format_probability(self.yes_bid),
            "preclose_yes_ask": _format_probability(self.yes_ask),
            "preclose_yes_mid": _format_probability(self.yes_mid),
            "candle_end_ts": str(int(now.timestamp())),
            "snapshot_staleness_seconds": "0",
            "settlement_ts": "",
            "source": "active_market_poll",
        }


class PollPredictionRow(MarketPollerBase):
    market_ticker: str
    event_ticker: str
    target_phrase: str
    model_name: str
    model_probability: float
    market_probability: float
    yes_bid: float
    yes_ask: float
    residual_delta: float
    side: TradeSide | str
    edge: float
    cost: float
    volume: int = 0


class MarketPollSnapshot(MarketPollerBase):
    poll_id: str
    model_name: str
    started_at: datetime
    completed_at: datetime
    market_count: int
    prediction_count: int
    trade_count: int
    prediction_rows: list[PollPredictionRow] = Field(default_factory=list)
    trade_rows: list[PollPredictionRow] = Field(default_factory=list)


class ActiveMarketSource(Protocol):
    def list_active_markets(self) -> list[ActiveMarketRow]:
        pass


class MarketScorer(Protocol):
    def score_active_markets(
        self,
        markets: list[ActiveMarketRow],
        *,
        model_name: str,
    ) -> list[PollPredictionRow]:
        pass


class KalshiActiveMarketSource:
    def __init__(
        self,
        *,
        http_client: httpx.Client,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        max_pages: int | None = None,
    ) -> None:
        self._client = KalshiMentionClient(http_client=http_client, base_url=base_url)
        self._max_pages = max_pages

    def list_active_markets(self) -> list[ActiveMarketRow]:
        rows: list[ActiveMarketRow] = []
        for event_payload in self._client.iter_mention_series(
            status="open",
            category="Mentions",
            query=EARNINGS_MENTION_PREFIX,
            max_pages=self._max_pages,
        ):
            raw_markets = event_payload.get("markets", [])
            if not isinstance(raw_markets, list):
                raw_markets = []
            if not raw_markets:
                event_ticker = str(event_payload.get("event_ticker") or "").strip()
                if event_ticker:
                    raw_markets = list(
                        self._client.iter_event_markets(
                            event_ticker=event_ticker,
                            status="open",
                            historical=False,
                        )
                    )
            for market_payload in raw_markets:
                if not isinstance(market_payload, dict):
                    continue
                row = normalize_active_market(
                    event_payload=event_payload,
                    market_payload=market_payload,
                )
                if row is not None:
                    rows.append(row)
        return sorted(rows, key=lambda row: (-row.volume, row.market_ticker))


class CachedSavedModelMarketScorer:
    def __init__(self, *, models_root: Path) -> None:
        self._registry = SavedModelRegistry(models_root=models_root)
        self._scorers: dict[str, CachedRuntimeSavedModelScorer] = {}

    def score_active_markets(
        self,
        markets: list[ActiveMarketRow],
        *,
        model_name: str,
    ) -> list[PollPredictionRow]:
        model_dir = self._registry.model_dir(model_name)
        scorer = self._scorers.get(model_name)
        if scorer is None:
            scorer = CachedRuntimeSavedModelScorer(model_dir)
            self._scorers[model_name] = scorer
        predictions = []
        for market in markets:
            score_row = scorer.score_row_dict(market.to_runtime_row())
            predictions.append(_poll_row_from_score(market, model_name, score_row.model_dump()))
        return predictions


class MarketPollCacheStore:
    def __init__(self, *, root: Path) -> None:
        self._root = root

    @property
    def latest_snapshot_path(self) -> Path:
        return self._root / "latest-poll.json"

    @property
    def latest_trades_path(self) -> Path:
        return self._root / "latest-trades.json"

    def write_snapshot(self, snapshot: MarketPollSnapshot) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        history_path = self._root / "history" / f"{snapshot.poll_id}.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        payload = snapshot.model_dump(mode="json")
        _write_json(history_path, payload)
        _write_json(self.latest_snapshot_path, payload)
        _write_json(
            self.latest_trades_path,
            [row.model_dump(mode="json") for row in snapshot.trade_rows],
        )

    def read_latest_snapshot(self) -> MarketPollSnapshot | None:
        if not self.latest_snapshot_path.exists():
            return None
        return MarketPollSnapshot.model_validate(
            json.loads(self.latest_snapshot_path.read_text(encoding="utf-8"))
        )

    def read_latest_trades(self) -> list[PollPredictionRow]:
        if not self.latest_trades_path.exists():
            return []
        return [
            PollPredictionRow.model_validate(row)
            for row in json.loads(self.latest_trades_path.read_text(encoding="utf-8"))
        ]


class ActiveMarketPoller:
    def __init__(
        self,
        *,
        market_source: ActiveMarketSource,
        scorer: MarketScorer,
        cache_store: MarketPollCacheStore,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._market_source = market_source
        self._scorer = scorer
        self._cache_store = cache_store
        self._now = now or (lambda: datetime.now(tz=UTC))

    def run_once(self, *, model_name: str) -> MarketPollSnapshot:
        started_at = self._now()
        markets = self._market_source.list_active_markets()
        predictions = self._scorer.score_active_markets(markets, model_name=model_name)
        trade_rows = [row for row in predictions if row.side in {"YES", "NO"}]
        completed_at = self._now()
        snapshot = MarketPollSnapshot(
            poll_id=f"{started_at:%Y%m%d-%H%M%S}",
            model_name=model_name,
            started_at=started_at,
            completed_at=completed_at,
            market_count=len(markets),
            prediction_count=len(predictions),
            trade_count=len(trade_rows),
            prediction_rows=predictions,
            trade_rows=trade_rows,
        )
        self._cache_store.write_snapshot(snapshot)
        return snapshot


def normalize_active_market(
    *,
    event_payload: dict[str, Any],
    market_payload: dict[str, Any],
) -> ActiveMarketRow | None:
    merged_payload = {**market_payload}
    event_ticker = str(
        market_payload.get("event_ticker") or event_payload.get("event_ticker") or ""
    ).strip()
    series_ticker = str(
        market_payload.get("series_ticker") or event_payload.get("series_ticker") or ""
    ).strip()
    market_ticker = str(
        market_payload.get("ticker") or market_payload.get("market_ticker") or ""
    ).strip()
    event_title = str(
        event_payload.get("event_title") or event_payload.get("title") or ""
    ).strip()
    market_title = str(market_payload.get("title") or event_title).strip()
    merged_payload["event_ticker"] = event_ticker
    merged_payload["series_ticker"] = series_ticker
    if not market_ticker or not is_earnings_mention_market(merged_payload):
        return None
    yes_bid = _price_probability(market_payload, "yes_bid")
    yes_ask = _price_probability(market_payload, "yes_ask")
    if yes_bid is None or yes_ask is None or yes_bid > yes_ask:
        return None
    target_phrase = extract_target_phrase(merged_payload)
    if not target_phrase:
        return None
    return ActiveMarketRow(
        market_ticker=market_ticker,
        event_ticker=event_ticker,
        series_ticker=series_ticker,
        event_title=event_title,
        market_title=market_title,
        target_phrase=target_phrase,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        yes_mid=(yes_bid + yes_ask) / 2.0,
        volume=int(market_payload.get("volume") or 0),
    )


@app.command("once")
def poll_once_command(
    model_name: Annotated[str | None, typer.Option("--model-name")] = None,
    models_root: Annotated[Path | None, typer.Option()] = None,
    cache_root: Annotated[Path | None, typer.Option()] = None,
    max_pages: Annotated[int | None, typer.Option()] = None,
) -> None:
    models_root = models_root or default_models_root()
    cache_root = cache_root or default_poll_cache_root()
    resolved_model_name = preferred_model_name(models_root, model_name)
    with httpx.Client(timeout=30) as http_client:
        poller = ActiveMarketPoller(
            market_source=KalshiActiveMarketSource(http_client=http_client, max_pages=max_pages),
            scorer=CachedSavedModelMarketScorer(models_root=models_root),
            cache_store=MarketPollCacheStore(root=cache_root),
        )
        snapshot = poller.run_once(model_name=resolved_model_name)
    typer.echo(
        f"Poll {snapshot.poll_id}: {snapshot.prediction_count} predictions, "
        f"{snapshot.trade_count} trade opportunities"
    )


@app.command("loop")
def poll_loop_command(
    model_name: Annotated[str | None, typer.Option("--model-name")] = None,
    models_root: Annotated[Path | None, typer.Option()] = None,
    cache_root: Annotated[Path | None, typer.Option()] = None,
    interval_seconds: Annotated[int, typer.Option(min=1)] = 600,
    max_pages: Annotated[int | None, typer.Option()] = None,
) -> None:
    models_root = models_root or default_models_root()
    cache_root = cache_root or default_poll_cache_root()
    resolved_model_name = preferred_model_name(models_root, model_name)
    scorer = CachedSavedModelMarketScorer(models_root=models_root)
    cache_store = MarketPollCacheStore(root=cache_root)
    while True:
        with httpx.Client(timeout=30) as http_client:
            poller = ActiveMarketPoller(
                market_source=KalshiActiveMarketSource(
                    http_client=http_client,
                    max_pages=max_pages,
                ),
                scorer=scorer,
                cache_store=cache_store,
            )
            snapshot = poller.run_once(model_name=resolved_model_name)
        typer.echo(
            f"Poll {snapshot.poll_id}: {snapshot.prediction_count} predictions, "
            f"{snapshot.trade_count} trade opportunities"
        )
        time.sleep(interval_seconds)


def _poll_row_from_score(
    market: ActiveMarketRow,
    model_name: str,
    score_payload: dict[str, Any],
) -> PollPredictionRow:
    score_row = normalize_score_payload(score_payload.get("raw", score_payload))
    return PollPredictionRow(
        market_ticker=market.market_ticker,
        event_ticker=market.event_ticker,
        target_phrase=market.target_phrase,
        model_name=model_name,
        model_probability=score_row.model_probability,
        market_probability=score_row.market_probability,
        yes_bid=market.yes_bid,
        yes_ask=market.yes_ask,
        residual_delta=score_row.residual_delta,
        side=score_row.side,
        edge=score_row.edge,
        cost=score_row.cost,
        volume=market.volume,
    )


def default_models_root() -> Path:
    return Path(__file__).resolve().parents[3] / "models"


def default_poll_cache_root() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "runtime" / "workstation"


def preferred_model_name(models_root: Path, requested_model_name: str | None) -> str:
    if requested_model_name:
        return requested_model_name
    registry = SavedModelRegistry(models_root=models_root)
    models = registry.list_models()
    if not models:
        raise typer.BadParameter(f"No valid saved models found under {models_root}")
    names = {model.name for model in models}
    for candidate in ("kalorie-v2", "earnings-mention-full-web-residual-v1"):
        if candidate in names:
            return candidate
    return models[0].name


def _write_runtime_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "market_ticker",
        "event_ticker",
        "series_ticker",
        "market_category",
        "event_phrase",
        "market_name",
        "word_said",
        "normalized_word_said",
        "final_outcome",
        "status",
        "close_time",
        "snapshot_target_time",
        "preclose_yes_bid",
        "preclose_yes_ask",
        "preclose_yes_mid",
        "candle_end_ts",
        "snapshot_staleness_seconds",
        "settlement_ts",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _price_probability(payload: dict[str, Any], key: str) -> float | None:
    for candidate_key in (f"{key}_dollars", key):
        value = payload.get(candidate_key)
        if value is None or value == "":
            continue
        numeric = float(value)
        return numeric / 100.0 if numeric > 1.0 else numeric
    return None


def _format_probability(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")
