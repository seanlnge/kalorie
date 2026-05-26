from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from kalorie2.market_poller import MarketPollCacheStore, default_poll_cache_root
from kalorie2.saved_models import (
    SavedModelRegistry,
    SavedModelScorer,
    read_sample_rows,
)

ExecutionMode = Literal["all", "no_only"]


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

    return app


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
