"""Build Kalshi + model snapshots and write them to S3 as yyyymmddHH.json."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import httpx

from kalorie2.market_poller import (
    ActiveMarketRow,
    CachedSavedModelMarketScorer,
    KalshiActiveMarketSource,
    OpenAIWebEvidenceSource,
    PollPredictionRow,
)


def snapshot_id_for(now: datetime) -> str:
    """UTC hour key like 2026081200."""
    return now.astimezone(UTC).strftime("%Y%m%d%H")


def _event_date_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        else:
            parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def filter_non_past_markets(
    markets: list[ActiveMarketRow],
    *,
    now: datetime | None = None,
) -> list[ActiveMarketRow]:
    """Drop markets whose event day is before today's UTC date."""
    today = (now or datetime.now(tz=UTC)).astimezone(UTC).date()
    kept: list[ActiveMarketRow] = []
    for market in markets:
        event_at = _event_date_utc(market.event_datetime)
        if event_at is not None and event_at.date() < today:
            continue
        kept.append(market)
    return kept


def prediction_to_delta_row(prediction: PollPredictionRow) -> dict[str, Any]:
    """All scored markets with model fields; no trade side / stake."""
    delta = float(prediction.residual_delta)
    return {
        "market_ticker": prediction.market_ticker,
        "event_ticker": prediction.event_ticker,
        "event_datetime": prediction.event_datetime,
        "event_title": prediction.event_title,
        "target_phrase": prediction.target_phrase,
        "model_name": prediction.model_name,
        "model_probability": float(prediction.model_probability),
        "market_probability": float(prediction.market_probability),
        "yes_bid": float(prediction.yes_bid),
        "yes_ask": float(prediction.yes_ask),
        # residual from the saved model (also exposed as delta for the desk)
        "residual_delta": delta,
        "delta": delta,
        "abs_delta": abs(delta),
        "volume": int(prediction.volume),
        "prediction_eligible": prediction.passes_risk_filter,
    }


def build_snapshot_payload(
    *,
    snapshot_id: str,
    generated_at: datetime,
    model_name: str,
    markets: list[ActiveMarketRow],
    predictions: list[PollPredictionRow],
) -> dict[str, Any]:
    delta_rows = [prediction_to_delta_row(row) for row in predictions]
    return {
        "snapshot_id": snapshot_id,
        "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "model_name": model_name,
        "market_count": len(markets),
        "prediction_count": len(delta_rows),
        "markets": [market.model_dump(mode="json") for market in markets],
        "predictions": delta_rows,
    }


def put_snapshot_json(
    *,
    bucket: str,
    snapshot_id: str,
    payload: dict[str, Any],
    s3_client: Any | None = None,
) -> str:
    client = s3_client or boto3.client("s3")
    key = f"{snapshot_id}.json"
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        CacheControl="no-cache",
    )
    # Pointer for the desk app (always the newest hour snapshot).
    client.put_object(
        Bucket=bucket,
        Key="latest.json",
        Body=body,
        ContentType="application/json",
        CacheControl="no-cache",
    )
    return key


def run_snapshot(
    *,
    model_name: str,
    models_root: Path,
    bucket: str,
    live_web_evidence: bool = True,
    web_search_model: str = "gpt-5.4-mini",
    web_search_timeout_seconds: float = 120.0,
    web_evidence_max_workers: int = 12,
    scan_all_open_markets: bool = False,
    max_pages: int | None = None,
    now: datetime | None = None,
    s3_client: Any | None = None,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    generated_at = now or datetime.now(tz=UTC)
    snapshot_id = snapshot_id_for(generated_at)
    print(
        f"snapshot start id={snapshot_id} model={model_name} "
        f"live_web={live_web_evidence} workers={web_evidence_max_workers} "
        f"scan_all_open={scan_all_open_markets}",
        flush=True,
    )
    web_evidence_source = (
        OpenAIWebEvidenceSource(
            model=web_search_model,
            timeout_seconds=web_search_timeout_seconds,
            max_workers=web_evidence_max_workers,
        )
        if live_web_evidence
        else None
    )
    scorer = CachedSavedModelMarketScorer(
        models_root=models_root,
        web_evidence_source=web_evidence_source,
    )

    owns_http = http_client is None
    client = http_client or httpx.Client(timeout=30.0)
    try:
        markets = KalshiActiveMarketSource(
            http_client=client,
            max_pages=max_pages,
            scan_all_open_markets=scan_all_open_markets,
        ).list_active_markets()
        before = len(markets)
        markets = filter_non_past_markets(markets, now=generated_at)
        print(
            f"listed markets={before} kept_non_past={len(markets)} "
            f"events={len({m.event_ticker for m in markets})}",
            flush=True,
        )
        predictions = scorer.score_active_markets(markets, model_name=model_name)
        print(f"scored predictions={len(predictions)}", flush=True)
    finally:
        if owns_http:
            client.close()

    payload = build_snapshot_payload(
        snapshot_id=snapshot_id,
        generated_at=generated_at,
        model_name=model_name,
        markets=markets,
        predictions=predictions,
    )
    key = put_snapshot_json(
        bucket=bucket,
        snapshot_id=snapshot_id,
        payload=payload,
        s3_client=s3_client,
    )
    print(f"wrote s3://{bucket}/{key} and latest.json", flush=True)
    return {
        "bucket": bucket,
        "key": key,
        "snapshot_id": snapshot_id,
        "market_count": payload["market_count"],
        "prediction_count": payload["prediction_count"],
    }


def _region_from_secret_arn(secret_arn: str) -> str | None:
    # arn:aws:secretsmanager:REGION:ACCOUNT:secret:NAME
    parts = secret_arn.split(":")
    if len(parts) >= 4 and parts[2] == "secretsmanager":
        return parts[3] or None
    return None


def load_openai_api_key_from_secrets_manager(secret_arn: str) -> str:
    region = _region_from_secret_arn(secret_arn) or os.environ.get("AWS_REGION") or os.environ.get(
        "AWS_DEFAULT_REGION"
    )
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_arn)
    raw = response.get("SecretString") or ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OPENAI secret must be JSON with OPENAI_API_KEY") from exc
    key = str(payload.get("OPENAI_API_KEY") or "").strip()
    if not key or key == "REPLACE_ME":
        raise RuntimeError(
            "OPENAI_API_KEY is missing or still REPLACE_ME; "
            "update the Secrets Manager secret before running"
        )
    return key


def ensure_openai_api_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    secret_arn = os.environ.get("OPENAI_SECRET_ARN")
    if not secret_arn:
        return
    os.environ["OPENAI_API_KEY"] = load_openai_api_key_from_secrets_manager(secret_arn)


def handler_from_event(event: dict[str, Any] | None) -> dict[str, Any]:
    event = event or {}
    ensure_openai_api_key()
    model_name = str(event.get("model_name") or os.environ.get("MODEL_NAME") or "kalorie-v6")
    models_root = Path(os.environ.get("MODELS_ROOT") or "/opt/models")
    bucket = os.environ.get("SNAPSHOT_BUCKET")
    if not bucket:
        raise RuntimeError("SNAPSHOT_BUCKET is required")
    live_web_evidence = str(
        event.get("live_web_evidence", os.environ.get("LIVE_WEB_EVIDENCE", "true"))
    ).lower() not in {"0", "false", "no"}
    web_search_model = str(
        event.get("web_search_model")
        or os.environ.get("WEB_SEARCH_MODEL")
        or "gpt-5.4-mini"
    )
    max_pages_raw = event.get("max_pages", os.environ.get("MAX_PAGES"))
    max_pages = int(max_pages_raw) if max_pages_raw not in (None, "") else None
    workers_raw = event.get(
        "web_evidence_max_workers",
        os.environ.get("WEB_EVIDENCE_MAX_WORKERS", "12"),
    )
    web_evidence_max_workers = int(workers_raw) if workers_raw not in (None, "") else 12
    scan_all_open_markets = str(
        event.get(
            "scan_all_open_markets",
            os.environ.get("KALSHI_SCAN_ALL_OPEN_MARKETS", "false"),
        )
    ).lower() in {"1", "true", "yes"}
    return run_snapshot(
        model_name=model_name,
        models_root=models_root,
        bucket=bucket,
        live_web_evidence=live_web_evidence,
        web_search_model=web_search_model,
        web_evidence_max_workers=web_evidence_max_workers,
        scan_all_open_markets=scan_all_open_markets,
        max_pages=max_pages,
    )
