import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from kalorie2.execution.cli import app
from kalorie2.execution.state import ExecutionStateStore

runner = CliRunner()

MARKET_TICKER = "KXEARNINGSMENTIONAAPL-26APR30-AI"
EVENT_TICKER = "KXEARNINGSMENTIONAAPL-26APR30"


def _seed_latest_trades(cache_root: Path) -> None:
    event_datetime = (datetime.now(tz=UTC) + timedelta(hours=3)).isoformat()
    row = {
        "market_ticker": MARKET_TICKER,
        "event_ticker": EVENT_TICKER,
        "event_datetime": event_datetime,
        "target_phrase": "AI",
        "model_name": "kalorie-v2",
        "model_probability": 0.31,
        "market_probability": 0.435,
        "yes_bid": 0.42,
        "yes_ask": 0.45,
        "residual_delta": -0.12,
        "side": "NO",
        "edge": 0.06,
        "cost": 0.58,
        "volume": 100,
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "latest-trades.json").write_text(json.dumps([row]), encoding="utf-8")


def test_preview_dry_runs_without_submitting(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    execution_root = tmp_path / "execution"
    _seed_latest_trades(cache_root)

    result = runner.invoke(
        app,
        [
            "preview",
            "--cache-root",
            str(cache_root),
            "--execution-root",
            str(execution_root),
            "--bankroll",
            "100",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "dry_run"
    assert payload["dry_run_approved"] == 1
    assert payload["submitted"] == 0

    state = ExecutionStateStore(root=execution_root)
    assert state.last_audit()["event"] == "dry_run_approved"


def test_cli_exposes_all_subcommands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("preview", "once", "loop", "status", "halt", "unhalt", "stop", "resume"):
        assert command in result.output


def test_halt_and_unhalt_commands_update_state(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution"

    halt = runner.invoke(app, ["halt", MARKET_TICKER, "--execution-root", str(execution_root)])
    assert halt.exit_code == 0
    assert ExecutionStateStore(root=execution_root).is_halted(MARKET_TICKER) is True

    unhalt = runner.invoke(app, ["unhalt", MARKET_TICKER, "--execution-root", str(execution_root)])
    assert unhalt.exit_code == 0
    assert ExecutionStateStore(root=execution_root).is_halted(MARKET_TICKER) is False


def test_stop_and_resume_toggle_kill_switch(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution"

    runner.invoke(app, ["stop", "--execution-root", str(execution_root)])
    assert ExecutionStateStore(root=execution_root).kill_switch_active() is True

    runner.invoke(app, ["resume", "--execution-root", str(execution_root)])
    assert ExecutionStateStore(root=execution_root).kill_switch_active() is False
