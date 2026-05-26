import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from kalorie.domain.models import MentionMarketContract
from kalorie.workflows.models import WorkflowBaseModel


class KalshiEventPackCandidate(WorkflowBaseModel):
    event_ticker: str
    company_symbol: str
    company_name: str
    fiscal_year: int
    fiscal_quarter: int = Field(ge=1, le=4)
    call_start_at: datetime
    transcript_url: str | None = None
    evidence_urls: list[str] = Field(default_factory=list)

    @field_validator("company_symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.upper()


def build_event_pack(
    *,
    candidates: list[KalshiEventPackCandidate],
    kalshi_client: Any,
    output_dir: Path,
    snapshot_lead: timedelta = timedelta(minutes=10),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_summaries = []
    ready_count = 0
    for candidate in candidates:
        event_dir = output_dir / candidate.event_ticker
        event_dir.mkdir(parents=True, exist_ok=True)
        contracts = kalshi_client.get_event_mention_markets(candidate.event_ticker)
        snapshots = hydrate_t10_snapshots(
            candidate=candidate,
            contracts=contracts,
            kalshi_client=kalshi_client,
            snapshot_lead=snapshot_lead,
        )
        _write_json(event_dir / "event.json", candidate.model_dump(mode="json"))
        _write_json(
            event_dir / "contracts.json",
            [contract.model_dump(mode="json") for contract in contracts],
        )
        _write_json(event_dir / "snapshots.json", snapshots)
        _write_json(event_dir / "evidence-manifests.json", [])
        _write_readme(event_dir / "README.md", candidate=candidate, snapshots=snapshots)

        transcript_ready = (event_dir / "transcript" / "transcript.txt").exists()
        evidence_ready = bool(candidate.evidence_urls)
        ready = (
            bool(contracts)
            and len(snapshots) == len(contracts)
            and transcript_ready
            and evidence_ready
        )
        ready_count += int(ready)
        event_summaries.append(
            {
                "event_ticker": candidate.event_ticker,
                "company_symbol": candidate.company_symbol,
                "contract_count": len(contracts),
                "snapshot_count": len(snapshots),
                "ready": ready,
                "readiness_blockers": _readiness_blockers(
                    contracts=contracts,
                    snapshots=snapshots,
                    transcript_ready=transcript_ready,
                    evidence_ready=evidence_ready,
                ),
            }
        )

    summary = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "inspected_count": len(candidates),
        "ready_count": ready_count,
        "events": event_summaries,
    }
    _write_json(output_dir / "candidate-events-summary.json", summary)
    return summary


def hydrate_t10_snapshots(
    *,
    candidate: KalshiEventPackCandidate,
    contracts: list[MentionMarketContract],
    kalshi_client: Any,
    snapshot_lead: timedelta,
) -> list[dict[str, Any]]:
    cutoff = candidate.call_start_at - snapshot_lead
    snapshots = []
    for contract in contracts:
        payload = kalshi_client.get_market_candlesticks(
            series_ticker=_series_ticker(candidate.event_ticker),
            market_id=contract.market_id,
            start_ts=int((cutoff - timedelta(days=7)).timestamp()),
            end_ts=int(cutoff.timestamp()),
            period_interval=1,
        )
        candle = select_latest_pre_cutoff_candle(
            payload.get("candlesticks", []),
            cutoff=cutoff,
        )
        if candle is None:
            continue
        snapshots.append(_snapshot_from_candle(candidate, contract, candle, cutoff))
    return snapshots


def select_latest_pre_cutoff_candle(
    candles: list[dict[str, Any]],
    *,
    cutoff: datetime,
) -> dict[str, Any] | None:
    cutoff_ts = int(cutoff.timestamp())
    eligible = [
        candle
        for candle in candles
        if int(candle.get("end_period_ts", candle.get("end_ts", 0))) <= cutoff_ts
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda candle: int(candle.get("end_period_ts", candle.get("end_ts", 0))),
    )


def _snapshot_from_candle(
    candidate: KalshiEventPackCandidate,
    contract: MentionMarketContract,
    candle: dict[str, Any],
    cutoff: datetime,
) -> dict[str, Any]:
    end_ts = int(candle.get("end_period_ts", candle.get("end_ts")))
    return {
        "event_ticker": candidate.event_ticker,
        "market_id": contract.market_id,
        "target_phrase": contract.target_phrase.normalized_phrase,
        "snapshot_target_time": cutoff.isoformat(),
        "candle_end_ts": end_ts,
        "candle_end_at": datetime.fromtimestamp(end_ts, tz=UTC).isoformat(),
        "yes_bid": str(_normalize_price(candle.get("yes_bid", contract.yes_bid))),
        "yes_ask": str(_normalize_price(candle.get("yes_ask", contract.yes_ask))),
    }


def _series_ticker(event_ticker: str) -> str:
    return event_ticker.split("-", 1)[0]


def _normalize_price(value: Any) -> Decimal:
    decimal = Decimal(str(value))
    if decimal > 1:
        decimal = decimal / Decimal("100")
    return decimal.quantize(Decimal("0.01"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_readme(
    path: Path,
    *,
    candidate: KalshiEventPackCandidate,
    snapshots: list[dict[str, Any]],
) -> None:
    path.write_text(
        "\n".join(
            [
                f"# {candidate.event_ticker}",
                "",
                f"- Company: {candidate.company_name} ({candidate.company_symbol})",
                f"- Call start: {candidate.call_start_at.isoformat()}",
                f"- T-10 snapshots: {len(snapshots)}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _readiness_blockers(
    *,
    contracts: list[MentionMarketContract],
    snapshots: list[dict[str, Any]],
    transcript_ready: bool,
    evidence_ready: bool,
) -> list[str]:
    blockers = []
    if not contracts:
        blockers.append("missing_contracts")
    if len(snapshots) != len(contracts):
        blockers.append("missing_t10_snapshots")
    if not transcript_ready:
        blockers.append("missing_transcript")
    if not evidence_ready:
        blockers.append("missing_evidence")
    return blockers
