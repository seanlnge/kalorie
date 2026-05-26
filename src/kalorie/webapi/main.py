from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

from kalorie.webapi.data_cache import DataCacheManager
from kalorie.webapi.job_registry import (
    IdempotencyConflictError,
    JobRegistry,
    JobSubmissionRequest,
)
from kalorie.webapi.job_runner import JobExecutionContext, JobRunner
from kalorie.webapi.kalshi_service import KalshiWebService, WebMentionMarket
from kalorie.webapi.run_store import EventScope, RunStore


def create_app(
    *,
    run_root: Path | None = None,
    kalshi_service: KalshiWebService | Any | None = None,
) -> FastAPI:
    app = FastAPI(title="Kalorie Web API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    root = run_root or Path("runs/web")
    service = kalshi_service
    if service is None:
        service = KalshiWebService(http_client=httpx.Client(timeout=30))

    app.state.kalshi_service = service
    app.state.run_store = RunStore(root=root)
    app.state.job_registry = JobRegistry(
        max_active_jobs=8,
        max_cpu_slots=8,
        max_openai_slots=16,
        max_provider_slots=32,
    )
    app.state.cache_manager = DataCacheManager(run_store=app.state.run_store)
    app.state.job_runner = JobRunner(
        run_store=app.state.run_store,
        job_registry=app.state.job_registry,
        cache_manager=app.state.cache_manager,
        kalshi_service=app.state.kalshi_service,
        project_root=Path.cwd(),
    )
    app.state.job_run_scope: dict[str, tuple[EventScope, str]] = {}

    @app.get("/api/markets/open")
    def get_open_markets() -> JSONResponse:
        markets = app.state.kalshi_service.list_open_mention_markets()
        return JSONResponse(
            {
                "markets": [_serialize_market(market) for market in markets],
                "events": _serialize_market_events(markets),
            }
        )

    @app.get("/api/events/{event_ticker}/markets")
    def get_event_markets(event_ticker: str) -> JSONResponse:
        markets = app.state.kalshi_service.list_event_mention_markets(event_ticker)
        return JSONResponse({"markets": [_serialize_market(market) for market in markets]})

    @app.get("/api/markets/{market_ticker}/runs")
    def list_runs(market_ticker: str) -> JSONResponse:
        scope = _resolve_event_scope(market_ticker=market_ticker, service=app.state.kalshi_service)
        runs = app.state.run_store.list_runs(scope)
        return JSONResponse({"runs": [_serialize_run(run) for run in runs]})

    @app.get("/api/markets/{market_ticker}/runs/latest")
    def latest_run(market_ticker: str) -> JSONResponse:
        scope = _resolve_event_scope(market_ticker=market_ticker, service=app.state.kalshi_service)
        run = app.state.run_store.latest_completed_run(scope)
        if run is None:
            raise HTTPException(status_code=404, detail="No completed runs found for market scope")
        return JSONResponse({"run": _serialize_run(run)})

    @app.get("/api/markets/{market_ticker}/runs/{run_id}")
    def get_run(market_ticker: str, run_id: str) -> JSONResponse:
        scope = _resolve_event_scope(market_ticker=market_ticker, service=app.state.kalshi_service)
        run = app.state.run_store.get_run(scope, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        result_path = run.run_dir / "result.json"
        result_payload: dict[str, Any] | None = None
        if result_path.exists():
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        return JSONResponse({"run": _serialize_run(run), "result": result_payload})

    @app.post("/api/markets/{market_ticker}/jobs")
    async def create_job(
        market_ticker: str,
        background_tasks: BackgroundTasks,
        files: list[UploadFile] = File(default_factory=list),  # noqa: B008
        data_mode: str = Form(default="mixed_best_effort"),  # noqa: B008
        history_window: str = Form(default="all_available"),  # noqa: B008
        decision_cutoff_ts: datetime | None = Form(default=None),  # noqa: B008
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),  # noqa: B008
    ) -> JSONResponse:
        if not idempotency_key:
            raise HTTPException(status_code=400, detail="Idempotency-Key header is required")

        scope = _resolve_event_scope(market_ticker=market_ticker, service=app.state.kalshi_service)
        effective_cutoff = decision_cutoff_ts or datetime.now(tz=UTC)
        payload = {
            "market_ticker": market_ticker,
            "data_mode": data_mode,
            "history_window": history_window,
            "decision_cutoff_ts": effective_cutoff.isoformat(),
            "files": sorted(file.filename for file in files),
        }
        payload_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        try:
            job = app.state.job_registry.submit(
                JobSubmissionRequest(
                    idempotency_key=idempotency_key,
                    payload_hash=payload_hash,
                    market_ticker=market_ticker,
                    cpu_slots=1,
                    openai_slots=1,
                    provider_slots=1,
                )
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        existing_scope_run = app.state.job_run_scope.get(job.job_id)
        if existing_scope_run is not None:
            existing_scope, run_id = existing_scope_run
            run = app.state.run_store.get_run(existing_scope, run_id)
            if run is not None:
                return JSONResponse({"job": _serialize_job(job), "run": _serialize_run(run)})

        run = app.state.run_store.create_run(
            scope=scope,
            market_ticker=market_ticker,
            options={
                "data_mode": data_mode,
                "history_window": history_window,
                "effective_decision_cutoff_ts": effective_cutoff.isoformat(),
            },
        )
        app.state.job_run_scope[job.job_id] = (scope, run.run_id)
        for upload in files:
            upload_path = run.run_dir / "uploads" / (upload.filename or "upload.bin")
            upload_path.write_bytes(await upload.read())

        app.state.run_store.update_status(scope=scope, run_id=run.run_id, status=job.status)
        if job.status == "running":
            background_tasks.add_task(
                app.state.job_runner.run_job,
                JobExecutionContext(
                    job_id=job.job_id,
                    scope=scope,
                    run_id=run.run_id,
                    market_ticker=market_ticker,
                    effective_cutoff_ts=effective_cutoff,
                ),
            )
        return JSONResponse({"job": _serialize_job(job), "run": _serialize_run(run)})

    @app.get("/api/jobs")
    def list_jobs() -> JSONResponse:
        jobs = app.state.job_registry.list_jobs()
        return JSONResponse({"jobs": [_serialize_job(job) for job in jobs]})

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> JSONResponse:
        job = app.state.job_registry.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JSONResponse({"job": _serialize_job(job)})

    @app.websocket("/api/jobs/stream")
    async def jobs_stream(socket: WebSocket) -> None:
        await socket.accept()
        try:
            while True:
                jobs = [_serialize_job(job) for job in app.state.job_registry.list_jobs()]
                await socket.send_json({"jobs": jobs})
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            return

    return app


def _resolve_event_scope(*, market_ticker: str, service: Any) -> EventScope:
    match = re.search(r"KXEARNINGSMENTION([A-Z]+)-", market_ticker.upper())
    if match:
        return EventScope(
            company_symbol=match.group(1),
            event_key=market_ticker.rsplit("-", 1)[0],
        )

    for market in service.list_open_mention_markets():
        if market.market_ticker == market_ticker:
            return EventScope(company_symbol=market.company_symbol, event_key=market.event_ticker)

    return EventScope(company_symbol="UNKNOWN", event_key=market_ticker.rsplit("-", 1)[0])


def _serialize_market(market: WebMentionMarket) -> dict[str, Any]:
    yes_bid = Decimal(str(market.yes_bid))
    yes_ask = Decimal(str(market.yes_ask))
    return {
        "market_ticker": market.market_ticker,
        "event_ticker": market.event_ticker,
        "company_symbol": market.company_symbol,
        "title": market.title,
        "target_phrase": market.target_phrase,
        "yes_bid": str(yes_bid),
        "yes_ask": str(yes_ask),
        "spread": str((yes_ask - yes_bid).quantize(Decimal("0.01"))),
        "volume": market.volume,
    }


def _serialize_market_events(markets: list[WebMentionMarket]) -> list[dict[str, Any]]:
    grouped: dict[str, list[WebMentionMarket]] = defaultdict(list)
    for market in markets:
        grouped[market.event_ticker].append(market)

    serialized_events: list[dict[str, Any]] = []
    for event_ticker, event_markets in grouped.items():
        ordered_markets = sorted(
            event_markets,
            key=lambda market: (
                -market.volume,
                market.market_ticker,
            ),
        )
        representative = ordered_markets[0]
        serialized_events.append(
            {
                "event_ticker": event_ticker,
                "company_symbol": representative.company_symbol,
                "market_count": len(event_markets),
                "total_volume": sum(market.volume for market in event_markets),
                "representative_market_ticker": representative.market_ticker,
                "representative_phrase": representative.target_phrase,
            }
        )

    serialized_events.sort(
        key=lambda event: (
            event["company_symbol"],
            event["event_ticker"],
        )
    )
    return serialized_events


def _serialize_run(run: Any) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "market_ticker": run.market_ticker,
        "created_at": run.created_at.isoformat(),
        "status": run.status,
    }


def _serialize_job(job: Any) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "idempotency_key": job.idempotency_key,
        "market_ticker": job.market_ticker,
        "status": job.status,
        "wait_reason": job.wait_reason,
        "created_at": job.created_at.isoformat(),
    }


def run() -> None:
    uvicorn.run(
        "kalorie.webapi.main:create_app",
        host="127.0.0.1",
        port=8000,
        factory=True,
        reload=False,
    )

