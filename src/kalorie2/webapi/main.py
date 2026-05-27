from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

import httpx
import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from kalorie2.kalshi_account import KalshiAccountClient, build_account_summary, load_env_file
from kalorie2.market_poller import (
    ActiveMarketRow,
    CachedSavedModelMarketScorer,
    KalshiActiveMarketSource,
    MarketPollCacheStore,
    MarketPollSnapshot,
    PollPredictionRow,
    default_poll_cache_root,
)
from kalorie2.risk_presets import (
    RiskPreset,
    apply_risk_preset_to_market,
    get_risk_preset,
    list_risk_presets,
)
from kalorie2.saved_models import (
    SavedModelRegistry,
    SavedModelScorer,
    read_sample_rows,
)

ExecutionMode = Literal["all", "no_only"]
MARKET_POLL_INTERVAL_SECONDS = 60
MODEL_RUN_INTERVAL_SECONDS = 60 * 60


class CurrentMarketsRiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_preset: RiskPreset | None = None


@dataclass
class CurrentMarketPredictionCacheEntry:
    rows_by_ticker: dict[str, PollPredictionRow]
    markets: list[ActiveMarketRow]
    market_polled_at: datetime
    model_run_started_at: datetime
    model_run_completed_at: datetime


def create_app(
    *,
    models_root: Path | None = None,
    poll_cache_root: Path | None = None,
    env_path: Path | None = None,
) -> FastAPI:
    load_env_file(env_path or _default_env_path())
    app = FastAPI(title="Kalorie2 Saved Model Workstation API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.models_root = models_root or _default_models_root()
    app.state.poll_cache_root = poll_cache_root or default_poll_cache_root()
    app.state.registry = SavedModelRegistry(models_root=app.state.models_root)
    app.state.poll_cache_store = MarketPollCacheStore(root=app.state.poll_cache_root)
    app.state.current_market_scorer = CachedSavedModelMarketScorer(
        models_root=app.state.models_root
    )
    app.state.current_market_prediction_cache: dict[str, CurrentMarketPredictionCacheEntry] = {}

    @app.get("/api/models")
    def list_models() -> JSONResponse:
        registry: SavedModelRegistry = app.state.registry
        return JSONResponse(
            {"models": [model.model_dump(mode="json") for model in registry.list_models()]}
        )

    @app.get("/api/models/{model_name}")
    def get_model(model_name: str) -> JSONResponse:
        registry: SavedModelRegistry = app.state.registry
        try:
            model = registry.get_model(model_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"model": model.model_dump(mode="json")})

    @app.get("/api/risk-presets")
    def risk_presets() -> JSONResponse:
        return JSONResponse(
            {
                "risk_presets": [
                    preset.model_dump(mode="json") for preset in list_risk_presets()
                ]
            }
        )

    @app.get("/api/account/summary")
    def account_summary() -> JSONResponse:
        try:
            with httpx.Client(timeout=15) as http_client:
                account_client = KalshiAccountClient.from_env(http_client=http_client)
                if account_client is None:
                    summary = build_account_summary(
                        balance_payload=None,
                        positions_payload=None,
                        error="Kalshi account auth is not configured",
                    )
                else:
                    balance_payload = account_client.get_balance()
                    positions_payload = None
                    positions_error = None
                    try:
                        positions_payload = account_client.list_positions()
                    except Exception as exc:  # noqa: BLE001
                        positions_error = f"Failed to load Kalshi positions: {exc}"
                    summary = build_account_summary(
                        balance_payload=balance_payload,
                        positions_payload=positions_payload,
                        error=positions_error,
                    )
        except Exception as exc:  # noqa: BLE001
            summary = build_account_summary(
                balance_payload=None,
                positions_payload=None,
                error=f"Failed to load Kalshi account summary: {exc}",
            )
        return JSONResponse({"summary": summary.model_dump(mode="json")})

    @app.get("/api/models/{model_name}/sample-rows")
    def sample_rows(model_name: str, limit: int = 10) -> JSONResponse:
        registry: SavedModelRegistry = app.state.registry
        try:
            model_dir = registry.model_dir(model_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        rows = read_sample_rows(_training_csv_path(model_dir), limit=limit)
        return JSONResponse({"rows": rows})

    @app.post("/api/models/{model_name}/score")
    async def score_model(
        model_name: str,
        row_index: int = Form(default=0),  # noqa: B008
        execution_mode: ExecutionMode = Form(default="all"),  # noqa: B008
        csv_file: UploadFile | None = File(default=None),  # noqa: B008
    ) -> JSONResponse:
        registry: SavedModelRegistry = app.state.registry
        try:
            model_dir = registry.model_dir(model_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        scorer = SavedModelScorer(model_dir)
        if csv_file is None:
            score_row = scorer.score_csv_row(_training_csv_path(model_dir), row_index=row_index)
        else:
            with tempfile.TemporaryDirectory(prefix="kalorie2-score-") as temp_dir:
                upload_path = Path(temp_dir) / (csv_file.filename or "rows.csv")
                upload_path.write_bytes(await csv_file.read())
                score_row = scorer.score_csv_row(upload_path, row_index=row_index)

        rows = [score_row]
        if execution_mode == "no_only":
            rows = [row for row in rows if row.side == "NO"]

        return JSONResponse(
            {
                "model_name": model_name,
                "execution_mode": execution_mode,
                "rows": [row.model_dump(mode="json") for row in rows],
            }
        )

    @app.get("/api/polls/latest")
    def latest_poll() -> JSONResponse:
        cache_store: MarketPollCacheStore = app.state.poll_cache_store
        snapshot = cache_store.read_latest_snapshot()
        if snapshot is None:
            raise HTTPException(status_code=404, detail="No poll snapshot found")
        return JSONResponse({"snapshot": snapshot.model_dump(mode="json")})

    @app.get("/api/trades/latest")
    def latest_trades() -> JSONResponse:
        cache_store: MarketPollCacheStore = app.state.poll_cache_store
        return JSONResponse(
            {"trades": [row.model_dump(mode="json") for row in cache_store.read_latest_trades()]}
        )

    @app.get("/api/polls/history")
    def poll_history(limit: int = 50) -> JSONResponse:
        cache_store: MarketPollCacheStore = app.state.poll_cache_store
        snapshots = cache_store.read_history(limit=limit)
        return JSONResponse(
            {"snapshots": [snapshot.model_dump(mode="json") for snapshot in snapshots]}
        )

    @app.post("/api/models/{model_name}/current-markets")
    def score_current_markets(
        model_name: str,
        risk_preset_id: str = "balanced",
        max_pages: int | None = 3,
        force_model_run: bool = False,
        refresh_markets: bool = True,
        model_run_interval_seconds: int = MODEL_RUN_INTERVAL_SECONDS,
        risk_request: Annotated[CurrentMarketsRiskRequest | None, Body()] = None,
    ) -> JSONResponse:
        registry: SavedModelRegistry = app.state.registry
        try:
            registry.model_dir(model_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if risk_request and risk_request.risk_preset:
            risk_preset = risk_request.risk_preset
        else:
            try:
                risk_preset = get_risk_preset(risk_preset_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        started_at = datetime.now(tz=UTC)
        try:
            cached_entry: CurrentMarketPredictionCacheEntry | None = (
                app.state.current_market_prediction_cache.get(model_name)
            )
            if not refresh_markets and cached_entry is not None:
                markets = cached_entry.markets
                market_polled_at = cached_entry.market_polled_at
            else:
                with httpx.Client(timeout=30) as http_client:
                    market_source = KalshiActiveMarketSource(
                        http_client=http_client,
                        max_pages=max_pages,
                    )
                    markets = market_source.list_active_markets()
                market_polled_at = datetime.now(tz=UTC)
            cache_entry = _prediction_cache_entry(
                app=app,
                model_name=model_name,
                markets=markets,
                market_polled_at=market_polled_at,
                force_model_run=force_model_run,
                interval_seconds=model_run_interval_seconds,
                now=market_polled_at,
            )
            prediction_rows = _risk_overlay_rows(
                cache_entry=cache_entry,
                markets=markets,
                risk_preset=risk_preset,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502, detail=f"Failed to score current markets: {exc}"
            ) from exc
        completed_at = datetime.now(tz=UTC)
        trade_rows = [row for row in prediction_rows if row.side in {"YES", "NO"}]
        snapshot = MarketPollSnapshot(
            poll_id=f"{started_at:%Y%m%d-%H%M%S}",
            model_name=model_name,
            risk_preset_id=risk_preset.id,
            started_at=started_at,
            completed_at=completed_at,
            market_polled_at=market_polled_at,
            model_run_started_at=cache_entry.model_run_started_at,
            model_run_completed_at=cache_entry.model_run_completed_at,
            next_market_poll_at=market_polled_at
            + timedelta(seconds=MARKET_POLL_INTERVAL_SECONDS),
            next_model_run_at=cache_entry.model_run_completed_at
            + timedelta(seconds=model_run_interval_seconds),
            market_count=len(markets),
            prediction_count=len(prediction_rows),
            trade_count=len(trade_rows),
            prediction_rows=prediction_rows,
            trade_rows=trade_rows,
        )
        return JSONResponse({"snapshot": snapshot.model_dump(mode="json")})

    return app


def _prediction_cache_entry(
    *,
    app: FastAPI,
    model_name: str,
    markets: list[ActiveMarketRow],
    market_polled_at: datetime,
    force_model_run: bool,
    interval_seconds: int,
    now: datetime,
) -> CurrentMarketPredictionCacheEntry:
    cache: dict[str, CurrentMarketPredictionCacheEntry] = app.state.current_market_prediction_cache
    existing = cache.get(model_name)
    market_tickers = {market.market_ticker for market in markets}
    cached_tickers = set(existing.rows_by_ticker) if existing else set()
    due = (
        existing is None
        or force_model_run
        or now >= existing.model_run_completed_at + timedelta(seconds=interval_seconds)
        or not market_tickers.issubset(cached_tickers)
    )
    if not due and existing is not None:
        existing.markets = markets
        existing.market_polled_at = market_polled_at
        return existing

    scorer: CachedSavedModelMarketScorer = app.state.current_market_scorer
    model_run_started_at = datetime.now(tz=UTC)
    base_rows = scorer.score_active_markets(markets, model_name=model_name)
    model_run_completed_at = datetime.now(tz=UTC)
    entry = CurrentMarketPredictionCacheEntry(
        rows_by_ticker={row.market_ticker: row for row in base_rows},
        markets=markets,
        market_polled_at=market_polled_at,
        model_run_started_at=model_run_started_at,
        model_run_completed_at=model_run_completed_at,
    )
    cache[model_name] = entry
    return entry


def _risk_overlay_rows(
    *,
    cache_entry: CurrentMarketPredictionCacheEntry,
    markets: list[ActiveMarketRow],
    risk_preset: RiskPreset,
) -> list[PollPredictionRow]:
    rows: list[PollPredictionRow] = []
    for market in markets:
        cached_row = cache_entry.rows_by_ticker.get(market.market_ticker)
        if cached_row is None:
            continue
        refreshed = _refresh_market_data(cached_row, market)
        rows.append(_apply_risk_preset_to_poll_row(refreshed, risk_preset=risk_preset))
    return rows


def _refresh_market_data(row: PollPredictionRow, market: ActiveMarketRow) -> PollPredictionRow:
    return row.model_copy(
        update={
            "event_ticker": market.event_ticker,
            "event_datetime": market.event_datetime,
            "event_title": market.event_title,
            "target_phrase": market.target_phrase,
            "market_probability": market.yes_mid,
            "yes_bid": market.yes_bid,
            "yes_ask": market.yes_ask,
            "residual_delta": row.model_probability - market.yes_mid,
            "volume": market.volume,
        }
    )


def _apply_risk_preset_to_poll_row(row, *, risk_preset: RiskPreset):
    decision = apply_risk_preset_to_market(
        preset=risk_preset,
        model_probability=row.model_probability,
        yes_bid=row.yes_bid,
        yes_ask=row.yes_ask,
    )
    return row.model_copy(
        update={
            "risk_preset_id": risk_preset.id,
            "side": decision.side,
            "edge": decision.edge,
            "cost": decision.cost,
            "ev_per_contract": decision.ev_per_contract,
            "kelly_fraction_raw": decision.kelly_fraction_raw,
            "recommended_fraction": decision.recommended_fraction,
            "passes_risk_filter": decision.passes_filter,
        }
    )


def run() -> None:
    uvicorn.run(
        "kalorie2.webapi.main:create_app",
        host="127.0.0.1",
        port=8000,
        factory=True,
        reload=False,
    )


def _default_models_root() -> Path:
    cwd = Path.cwd()
    candidates = [cwd / "models", cwd.parent / "models"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return cwd / "models"


def _default_env_path() -> Path:
    return Path.cwd() / ".env"


def _training_csv_path(model_dir: Path) -> Path:
    manifest_path = model_dir / "artifacts" / "training-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        training_corpus = manifest.get("training_corpus", {})
        if isinstance(training_corpus, dict):
            saved_csv = training_corpus.get("saved_csv")
            if saved_csv:
                return model_dir / str(saved_csv)
    return model_dir / "training" / "mention-markets-historical-20260523.csv"
