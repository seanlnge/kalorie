from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from kalorie2.market_poller import (
    CachedSavedModelMarketScorer,
    KalshiActiveMarketSource,
    MarketPollCacheStore,
    MarketPollSnapshot,
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


class CurrentMarketsRiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_preset: RiskPreset | None = None


def create_app(
    *,
    models_root: Path | None = None,
    poll_cache_root: Path | None = None,
) -> FastAPI:
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
        risk_request: CurrentMarketsRiskRequest | None = Body(default=None),
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
        scorer: CachedSavedModelMarketScorer = app.state.current_market_scorer
        started_at = datetime.now(tz=UTC)
        try:
            with httpx.Client(timeout=30) as http_client:
                market_source = KalshiActiveMarketSource(
                    http_client=http_client,
                    max_pages=max_pages,
                )
                markets = market_source.list_active_markets()
            prediction_rows = scorer.score_active_markets(markets, model_name=model_name)
            prediction_rows = [
                _apply_risk_preset_to_poll_row(row, risk_preset=risk_preset)
                for row in prediction_rows
            ]
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
            market_count=len(markets),
            prediction_count=len(prediction_rows),
            trade_count=len(trade_rows),
            prediction_rows=prediction_rows,
            trade_rows=trade_rows,
        )
        return JSONResponse({"snapshot": snapshot.model_dump(mode="json")})

    return app


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
