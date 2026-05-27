from __future__ import annotations

import csv
import json
import os
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
from kalorie2.web_evidence import (
    build_openai_web_search_payload,
    build_web_evidence_prompt,
    parse_web_evidence_response,
)

app = typer.Typer(help="Poll active Kalshi mention markets with a saved model.")

TradeSide = Literal["YES", "NO", "NONE"]


class MarketPollerBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ActiveMarketRow(MarketPollerBase):
    market_ticker: str
    event_ticker: str
    series_ticker: str
    event_datetime: str | None = None
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
    event_datetime: str | None = None
    event_title: str = ""
    target_phrase: str
    model_name: str
    risk_preset_id: str | None = None
    model_probability: float
    market_probability: float
    yes_bid: float
    yes_ask: float
    residual_delta: float
    side: TradeSide | str
    edge: float
    cost: float
    ev_per_contract: float | None = None
    kelly_fraction_raw: float | None = None
    recommended_fraction: float | None = None
    passes_risk_filter: bool | None = None
    volume: int = 0
    recommended_dollars: float | None = None
    recommended_contracts: int | None = None


class MarketPollSnapshot(MarketPollerBase):
    poll_id: str
    model_name: str
    risk_preset_id: str | None = None
    started_at: datetime
    completed_at: datetime
    market_polled_at: datetime | None = None
    model_run_started_at: datetime | None = None
    model_run_completed_at: datetime | None = None
    next_market_poll_at: datetime | None = None
    next_model_run_at: datetime | None = None
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


class LiveWebEvidenceSource(Protocol):
    def fetch_packets(self, markets: list[ActiveMarketRow]) -> dict[str, Any]:
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
        rows_by_ticker: dict[str, ActiveMarketRow] = {}
        events_by_ticker: dict[str, dict[str, Any]] = {}

        def hydrated_event_payload(event_payload: dict[str, Any]) -> dict[str, Any]:
            event_ticker = str(event_payload.get("event_ticker") or "").strip()
            if not event_ticker:
                return event_payload
            if event_ticker not in events_by_ticker:
                events_by_ticker[event_ticker] = self._client.get_event(event_ticker)
            hydrated = {**event_payload, **events_by_ticker[event_ticker]}
            if events_by_ticker[event_ticker].get("title"):
                hydrated["event_title"] = events_by_ticker[event_ticker]["title"]
            return hydrated

        def upsert_row(row: ActiveMarketRow) -> None:
            existing = rows_by_ticker.get(row.market_ticker)
            if existing is None:
                rows_by_ticker[row.market_ticker] = row
                return
            if row.volume > existing.volume:
                rows_by_ticker[row.market_ticker] = row
                return
            if existing.event_datetime is None and row.event_datetime is not None:
                rows_by_ticker[row.market_ticker] = row

        def collect_row(event_payload: dict[str, Any], market_payload: dict[str, Any]) -> None:
            row = normalize_active_market(
                event_payload=hydrated_event_payload(event_payload),
                market_payload=market_payload,
            )
            if row is not None:
                upsert_row(row)

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
                collect_row(event_payload, market_payload)
        for market_payload in self._client.iter_markets(
            status="open",
            historical=False,
            limit=200,
            max_pages=self._max_pages,
        ):
            event_payload = {
                "event_ticker": str(market_payload.get("event_ticker") or "").strip(),
                "series_ticker": str(market_payload.get("series_ticker") or "").strip(),
                "event_title": str(
                    market_payload.get("event_title") or market_payload.get("subtitle") or ""
                ).strip(),
            }
            collect_row(event_payload, market_payload)
        rows = list(rows_by_ticker.values())
        return sorted(
            rows,
            key=lambda row: (
                _event_datetime_sort_key(row.event_datetime),
                -row.volume,
                row.market_ticker,
            ),
        )


class CachedSavedModelMarketScorer:
    def __init__(
        self,
        *,
        models_root: Path,
        web_evidence_source: LiveWebEvidenceSource | None = None,
    ) -> None:
        self._registry = SavedModelRegistry(models_root=models_root)
        self._scorers: dict[str, CachedRuntimeSavedModelScorer] = {}
        self._web_evidence_source = web_evidence_source

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
        live_web_evidence_packets = {}
        if self._web_evidence_source is not None:
            raw_packets = self._web_evidence_source.fetch_packets(markets)
            live_web_evidence_packets = _coerce_live_packets_for_runtime(
                scorer,
                raw_packets,
            )
        predictions = []
        for market in markets:
            score_row = scorer.score_row_dict(
                market.to_runtime_row(),
                web_evidence_by_event=live_web_evidence_packets or None,
            )
            predictions.append(_poll_row_from_score(market, model_name, score_row.model_dump()))
        return predictions


class OpenAIWebEvidenceSource:
    def __init__(
        self,
        *,
        model: str,
        timeout_seconds: float = 120.0,
        fetch_web_evidence_packet: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._fetch_web_evidence_packet = fetch_web_evidence_packet or _fetch_web_evidence_packet

    def fetch_packets(self, markets: list[ActiveMarketRow]) -> dict[str, Any]:
        if not markets:
            return {}
        grouped_markets: dict[str, list[ActiveMarketRow]] = {}
        for market in markets:
            grouped_markets.setdefault(market.event_ticker, []).append(market)
        packets = {}
        cutoff_time_iso = datetime.now(tz=UTC).isoformat()
        for event_ticker, event_markets in grouped_markets.items():
            first_market = event_markets[0]
            target_phrases = sorted(
                {
                    event_market.target_phrase.strip().lower()
                    for event_market in event_markets
                    if event_market.target_phrase.strip()
                }
            )
            try:
                payload = self._fetch_web_evidence_packet(
                    event_ticker=event_ticker,
                    company_name=_company_name_from_event_title(first_market.event_title),
                    cutoff_time_iso=cutoff_time_iso,
                    target_phrases=target_phrases,
                    model=self._model,
                    timeout_seconds=self._timeout_seconds,
                )
                packets[event_ticker] = payload
            except Exception as exc:  # noqa: BLE001
                typer.echo(
                    f"warning: live web evidence fetch failed for {event_ticker}: {exc}",
                    err=True,
                )
        return packets


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

    def read_history(self, *, limit: int = 50) -> list[MarketPollSnapshot]:
        history_root = self._root / "history"
        if not history_root.exists():
            return []
        snapshots: list[MarketPollSnapshot] = []
        for path in sorted(history_root.glob("*.json"), reverse=True):
            if len(snapshots) >= limit:
                break
            snapshots.append(
                MarketPollSnapshot.model_validate(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            )
        return snapshots


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
        event_datetime=_event_datetime_iso(event_payload, market_payload),
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
    live_web_evidence: Annotated[
        bool, typer.Option("--live-web-evidence/--no-live-web-evidence")
    ] = True,
    web_search_model: Annotated[str, typer.Option()] = "gpt-5.4-mini",
    web_search_timeout_seconds: Annotated[float, typer.Option(min=1.0)] = 120.0,
) -> None:
    models_root = models_root or default_models_root()
    cache_root = cache_root or default_poll_cache_root()
    _load_env_file(_default_env_path())
    resolved_model_name = preferred_model_name(models_root, model_name)
    web_evidence_source = (
        OpenAIWebEvidenceSource(
            model=web_search_model,
            timeout_seconds=web_search_timeout_seconds,
        )
        if live_web_evidence
        else None
    )
    with httpx.Client(timeout=30) as http_client:
        poller = ActiveMarketPoller(
            market_source=KalshiActiveMarketSource(http_client=http_client, max_pages=max_pages),
            scorer=CachedSavedModelMarketScorer(
                models_root=models_root,
                web_evidence_source=web_evidence_source,
            ),
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
    live_web_evidence: Annotated[
        bool, typer.Option("--live-web-evidence/--no-live-web-evidence")
    ] = True,
    web_search_model: Annotated[str, typer.Option()] = "gpt-5.4-mini",
    web_search_timeout_seconds: Annotated[float, typer.Option(min=1.0)] = 120.0,
) -> None:
    models_root = models_root or default_models_root()
    cache_root = cache_root or default_poll_cache_root()
    _load_env_file(_default_env_path())
    resolved_model_name = preferred_model_name(models_root, model_name)
    web_evidence_source = (
        OpenAIWebEvidenceSource(
            model=web_search_model,
            timeout_seconds=web_search_timeout_seconds,
        )
        if live_web_evidence
        else None
    )
    scorer = CachedSavedModelMarketScorer(
        models_root=models_root,
        web_evidence_source=web_evidence_source,
    )
    cache_store = MarketPollCacheStore(root=cache_root)
    while True:
        try:
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
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Poll iteration failed; retrying in {interval_seconds}s: {exc}", err=True)
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
        event_datetime=market.event_datetime,
        event_title=market.event_title,
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


def _default_env_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"


def preferred_model_name(models_root: Path, requested_model_name: str | None) -> str:
    if requested_model_name:
        return requested_model_name
    registry = SavedModelRegistry(models_root=models_root)
    models = registry.list_models()
    if not models:
        raise typer.BadParameter(f"No valid saved models found under {models_root}")
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


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


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


def _event_datetime_iso(
    event_payload: dict[str, Any],
    market_payload: dict[str, Any],
) -> str | None:
    for key in ("event_subtitle", "sub_title", "subtitle"):
        parsed = _event_subtitle_datetime_iso(event_payload.get(key))
        if parsed:
            return parsed
    for key in (
        "close_time",
        "expiration_time",
        "latest_expiration_time",
        "expected_expiration_time",
        "open_time",
    ):
        parsed = _parse_datetime_iso(event_payload.get(key))
        if parsed:
            return parsed
    for key in (
        "close_time",
        "expiration_time",
        "latest_expiration_time",
        "expected_expiration_time",
        "open_time",
    ):
        parsed = _parse_datetime_iso(market_payload.get(key))
        if parsed:
            return parsed
    return None


def _event_subtitle_datetime_iso(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text.lower().startswith("on "):
        return None
    for pattern in ("On %B %d, %Y", "On %b %d, %Y"):
        try:
            parsed = datetime.strptime(text, pattern).replace(tzinfo=UTC)
        except ValueError:
            continue
        return parsed.isoformat().replace("+00:00", "Z")
    return None


def _parse_datetime_iso(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.isoformat().replace("+00:00", "Z")


def _event_datetime_sort_key(value: str | None) -> datetime:
    parsed = _parse_datetime_iso(value)
    if parsed is None:
        return datetime.max.replace(tzinfo=UTC)
    return datetime.fromisoformat(parsed.replace("Z", "+00:00"))


def _company_name_from_event_title(event_title: str) -> str:
    lowered = event_title.lower()
    if lowered.startswith("what will ") and " say" in lowered:
        return event_title[len("What will ") : lowered.index(" say")].strip()
    return event_title.strip()


def _coerce_live_packets_for_runtime(
    scorer: CachedRuntimeSavedModelScorer,
    packets: dict[str, Any],
) -> dict[str, Any]:
    existing_packets = getattr(scorer, "_web_evidence_by_event", {})
    if not existing_packets:
        return packets
    sample_packet = next(iter(existing_packets.values()), None)
    if sample_packet is None:
        return packets
    coerced = {}
    for event_ticker, packet in packets.items():
        if isinstance(sample_packet, dict):
            if hasattr(packet, "model_dump"):
                coerced[event_ticker] = packet.model_dump(mode="json")
            else:
                coerced[event_ticker] = packet
            continue
        if hasattr(sample_packet.__class__, "model_validate"):
            if isinstance(packet, sample_packet.__class__):
                coerced[event_ticker] = packet
            elif hasattr(packet, "model_dump"):
                coerced[event_ticker] = sample_packet.__class__.model_validate(
                    packet.model_dump(mode="json")
                )
            else:
                coerced[event_ticker] = sample_packet.__class__.model_validate(packet)
            continue
        coerced[event_ticker] = packet
    return coerced


def _fetch_web_evidence_packet(
    *,
    event_ticker: str,
    company_name: str,
    cutoff_time_iso: str,
    target_phrases: list[str],
    model: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --live-web-evidence")
    prompt = build_web_evidence_prompt(
        event={
            "event_ticker": event_ticker,
            "company_name": company_name,
            "cutoff_time": cutoff_time_iso,
        },
        target_phrases=target_phrases,
    )
    payload = build_openai_web_search_payload(prompt=prompt, model=model)
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
    packet = parse_web_evidence_response(_response_output_text(response.json()))
    return packet.model_dump(mode="json")


def _response_output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    for output in payload.get("output", []):
        for content in output.get("content", []):
            if isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("OpenAI response did not include output text")


if __name__ == "__main__":
    app()
