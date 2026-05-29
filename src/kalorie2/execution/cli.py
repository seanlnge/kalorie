from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from kalorie2.execution.client import KalshiExecutionClient, MarketQuote, OrderbookDepth
from kalorie2.execution.config import LiveTradingConfig
from kalorie2.execution.state import ExecutionStateStore, date_key
from kalorie2.execution.trader import LiveTrader
from kalorie2.kalshi_account import load_env_file
from kalorie2.market_poller import (
    MarketPollCacheStore,
    PollPredictionRow,
    default_models_root,
    default_poll_cache_root,
)
from kalorie2.risk_presets import RiskPreset, get_risk_preset

app = typer.Typer(help="Autonomous live trader for Kalorie2 saved-model signals.")

CacheRootOption = Annotated[Path | None, typer.Option("--cache-root")]
ExecutionRootOption = Annotated[Path | None, typer.Option("--execution-root")]
ModelsRootOption = Annotated[Path | None, typer.Option("--models-root")]
RiskPresetOption = Annotated[str, typer.Option("--risk-preset-id")]


class PreviewClient:
    """Offline stand-in client for dry-run previews.

    Serves quotes from the cached signal rows, presents a synthetic authenticated
    balance so sizing can be exercised, and makes order submission impossible. If
    anything ever tries to submit in preview, that is a bug and we raise loudly.
    """

    def __init__(self, *, rows: list[PollPredictionRow], bankroll_dollars: float) -> None:
        self._quotes = {
            row.market_ticker: MarketQuote(
                market_ticker=row.market_ticker,
                yes_bid=row.yes_bid,
                yes_ask=row.yes_ask,
                no_bid=round(1 - row.yes_ask, 4),
                no_ask=round(1 - row.yes_bid, 4),
            )
            for row in rows
        }
        # Synthetic depth at the cached top-of-book so dry-run sizing is exercised.
        self._orderbooks = {
            row.market_ticker: OrderbookDepth(
                market_ticker=row.market_ticker,
                yes_bids=[(int(round(row.yes_bid * 100)), 200)],
                no_bids=[(int(round((1 - row.yes_ask) * 100)), 200)],
            )
            for row in rows
        }
        self._bankroll_cents = int(round(bankroll_dollars * 100))

    def get_market_quote(self, ticker: str) -> MarketQuote:
        quote = self._quotes.get(ticker)
        if quote is None:
            raise KeyError(f"No cached quote for {ticker}")
        return quote

    def get_orderbook(self, ticker: str) -> OrderbookDepth:
        book = self._orderbooks.get(ticker)
        if book is None:
            raise KeyError(f"No cached orderbook for {ticker}")
        return book

    def get_balance(self) -> dict[str, Any]:
        return {
            "balance": {
                "available_balance": self._bankroll_cents,
                "portfolio_value": self._bankroll_cents,
            }
        }

    def list_positions(self) -> dict[str, Any]:
        return {"market_positions": []}

    def list_resting_orders(self, *, ticker: str | None = None) -> list[dict[str, Any]]:
        return []

    def submit_limit_order(self, **kwargs: object) -> str:
        raise AssertionError("submit_limit_order must never be called during preview")

    def cancel_order(self, order_id: str) -> bool:
        return True


@app.command("preview")
def preview_command(
    cache_root: CacheRootOption = None,
    execution_root: ExecutionRootOption = None,
    models_root: ModelsRootOption = None,
    risk_preset_id: RiskPresetOption = "balanced",
    bankroll: Annotated[float, typer.Option("--bankroll", min=0.0)] = 100.0,
    max_signals: Annotated[int, typer.Option("--max-signals", min=1)] = 50,
) -> None:
    """Dry-run the safeguards against cached signals offline. Never submits orders."""

    cache_store = MarketPollCacheStore(root=cache_root or default_poll_cache_root())
    rows = cache_store.read_latest_trades()[:max_signals]
    state = ExecutionStateStore(root=execution_root or default_execution_root())
    risk_preset = _resolve_risk_preset(models_root, risk_preset_id)

    trader = LiveTrader(
        config=LiveTradingConfig(mode="dry_run"),
        client=PreviewClient(rows=rows, bankroll_dollars=bankroll),
        state=state,
        risk_preset=risk_preset,
        signal_source=_RowsOnlySource(rows, cache_store),
    )
    summary = trader.run_once()
    typer.echo(json.dumps(summary.model_dump(mode="json"), indent=2))


@app.command("once")
def once_command(
    cache_root: CacheRootOption = None,
    execution_root: ExecutionRootOption = None,
    models_root: ModelsRootOption = None,
    risk_preset_id: RiskPresetOption = "balanced",
) -> None:
    """Run a single live/dry-run pass using KALORIE2_TRADING_MODE from the environment."""

    summary = _run_live_once(
        cache_root=cache_root,
        execution_root=execution_root,
        models_root=models_root,
        risk_preset_id=risk_preset_id,
    )
    typer.echo(json.dumps(summary.model_dump(mode="json"), indent=2))


@app.command("loop")
def loop_command(
    interval_seconds: Annotated[int, typer.Option("--interval-seconds", min=1)] = 60,
    cache_root: CacheRootOption = None,
    execution_root: ExecutionRootOption = None,
    models_root: ModelsRootOption = None,
    risk_preset_id: RiskPresetOption = "balanced",
) -> None:
    """Continuously run trading passes until interrupted."""

    while True:
        try:
            summary = _run_live_once(
                cache_root=cache_root,
                execution_root=execution_root,
                models_root=models_root,
                risk_preset_id=risk_preset_id,
            )
            typer.echo(
                f"mode={summary.mode} evaluated={summary.evaluated} "
                f"submitted={summary.submitted} rejected={summary.rejected} "
                f"halted={summary.halted} failed={summary.failed}"
            )
        except Exception as exc:  # noqa: BLE001 - keep the loop alive across transient failures
            typer.echo(f"Trading pass failed; retrying in {interval_seconds}s: {exc}", err=True)
        time.sleep(interval_seconds)


@app.command("status")
def status_command(
    execution_root: ExecutionRootOption = None,
) -> None:
    """Print mode, exposure caps, halts, daily counters, and last audit event."""

    _load_env()
    config = LiveTradingConfig.from_env(_environ())
    state = ExecutionStateStore(root=execution_root or default_execution_root())
    today = date_key(_utcnow())
    payload = {
        "mode": config.mode,
        "allows_real_orders": config.allows_real_orders(),
        "kill_switch_active": state.kill_switch_active(),
        "halted_contracts": state.halted_contracts(),
        "daily_orders": state.daily_orders(today),
        "daily_loss": state.daily_loss(today),
        "caps": {
            "max_total_exposure_fraction": config.max_total_exposure_fraction,
            "max_order_fraction": config.max_order_fraction,
            "max_event_exposure_fraction": config.max_event_exposure_fraction,
            "daily_loss_stop_fraction": config.daily_loss_stop_fraction,
            "max_total_exposure_dollars_ceiling": config.max_total_exposure_dollars,
            "max_order_dollars_ceiling": config.max_order_dollars,
            "max_event_exposure_dollars_ceiling": config.max_event_exposure_dollars,
            "daily_loss_stop_dollars_ceiling": config.daily_loss_stop_dollars,
            "event_cutoff_seconds": config.event_cutoff_seconds,
            "execution_drift_tolerance": config.execution_drift_tolerance,
            "price_swing_threshold": config.price_swing_threshold,
            "depth_anomaly_multiple": config.depth_anomaly_multiple,
            "fee_rate": config.fee_rate,
        },
        "last_audit": state.last_audit(),
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("halt")
def halt_command(
    market_ticker: Annotated[str, typer.Argument()],
    execution_root: ExecutionRootOption = None,
) -> None:
    """Manually halt a contract so the trader will not place further orders for it."""

    state = ExecutionStateStore(root=execution_root or default_execution_root())
    state.halt_contract(market_ticker, reason="manual_halt", now=_utcnow())
    typer.echo(f"Halted {market_ticker}")


@app.command("unhalt")
def unhalt_command(
    market_ticker: Annotated[str, typer.Argument()],
    execution_root: ExecutionRootOption = None,
) -> None:
    """Clear a manual or swing-triggered halt on a contract."""

    state = ExecutionStateStore(root=execution_root or default_execution_root())
    state.unhalt_contract(market_ticker)
    typer.echo(f"Unhalted {market_ticker}")


@app.command("stop")
def stop_command(
    reason: Annotated[str, typer.Option("--reason")] = "manual stop",
    execution_root: ExecutionRootOption = None,
) -> None:
    """Engage the kill switch: stop new orders and cancel resting ones next pass."""

    state = ExecutionStateStore(root=execution_root or default_execution_root())
    state.activate_kill_switch(reason=reason)
    typer.echo("Kill switch engaged")


@app.command("resume")
def resume_command(
    execution_root: ExecutionRootOption = None,
) -> None:
    """Clear the kill switch."""

    state = ExecutionStateStore(root=execution_root or default_execution_root())
    state.clear_kill_switch()
    typer.echo("Kill switch cleared")


def default_execution_root() -> Path:
    return Path(__file__).resolve().parents[3] / "artifacts" / "runtime" / "execution"


class _RowsOnlySource:
    def __init__(self, rows: list[PollPredictionRow], cache_store: MarketPollCacheStore) -> None:
        self._rows = rows
        self._cache_store = cache_store

    def read_latest_trades(self) -> list[PollPredictionRow]:
        return list(self._rows)

    def read_latest_snapshot(self):
        return self._cache_store.read_latest_snapshot()


def _run_live_once(
    *,
    cache_root: Path | None,
    execution_root: Path | None,
    models_root: Path | None,
    risk_preset_id: str,
):
    _load_env()
    config = LiveTradingConfig.from_env(_environ())
    cache_store = MarketPollCacheStore(root=cache_root or default_poll_cache_root())
    state = ExecutionStateStore(root=execution_root or default_execution_root())
    risk_preset = _resolve_risk_preset(models_root, risk_preset_id)

    with httpx.Client(timeout=15) as http_client:
        client = KalshiExecutionClient.from_env(http_client=http_client)
        if client is None:
            raise typer.BadParameter(
                "Live/once requires KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH"
            )
        trader = LiveTrader(
            config=config,
            client=client,
            state=state,
            risk_preset=risk_preset,
            signal_source=cache_store,
        )
        return trader.run_once()


def _resolve_risk_preset(models_root: Path | None, risk_preset_id: str) -> RiskPreset:
    store_path = (models_root or default_models_root()) / "risk-presets.json"
    return get_risk_preset(risk_preset_id, store_path=store_path)


def _load_env() -> None:
    load_env_file(_default_env_path())


def _default_env_path() -> Path:
    return Path(__file__).resolve().parents[3] / ".env"


def _environ() -> dict[str, str]:
    import os

    return dict(os.environ)


def _utcnow():
    from datetime import UTC, datetime

    return datetime.now(tz=UTC)


if __name__ == "__main__":
    app()
