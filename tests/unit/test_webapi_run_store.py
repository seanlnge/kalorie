from datetime import UTC, datetime
from pathlib import Path

from kalorie.webapi.run_store import EventScope, RunStore


def _event_scope() -> EventScope:
    return EventScope(company_symbol="WMT", event_key="KXEARNINGSMENTIONWMT-26Q2")


def test_event_scope_paths_are_company_event_scoped(tmp_path: Path) -> None:
    store = RunStore(root=tmp_path / "runs" / "web")
    scope = _event_scope()

    data_dir = store.event_data_dir(scope)
    runs_dir = store.event_runs_dir(scope)

    assert data_dir == (
        tmp_path
        / "runs"
        / "web"
        / "events"
        / "WMT"
        / "KXEARNINGSMENTIONWMT-26Q2"
        / "data"
    )
    assert runs_dir == (
        tmp_path
        / "runs"
        / "web"
        / "events"
        / "WMT"
        / "KXEARNINGSMENTIONWMT-26Q2"
        / "runs"
    )


def test_create_run_creates_scaffold_files(tmp_path: Path) -> None:
    store = RunStore(root=tmp_path / "runs" / "web")
    run = store.create_run(
        scope=_event_scope(),
        market_ticker="KXEARNINGSMENTIONWMT-26Q2-OMNICHANNEL",
        created_at=datetime(2026, 5, 20, 22, 0, 0, tzinfo=UTC),
        options={"data_mode": "mixed_best_effort"},
    )

    assert run.run_id.startswith("20260520-220000-")
    assert run.run_dir.exists()
    assert (run.run_dir / "uploads").exists()
    assert (run.run_dir / "artifacts").exists()
    assert (run.run_dir / "inputs.json").exists()
    assert (run.run_dir / "job_status.json").exists()
    assert (run.run_dir / "summary.json").exists()
    assert (run.run_dir / "job_log.txt").exists()


def test_latest_completed_run_returns_newest_completed(tmp_path: Path) -> None:
    store = RunStore(root=tmp_path / "runs" / "web")
    scope = _event_scope()

    first = store.create_run(
        scope=scope,
        market_ticker="KXEARNINGSMENTIONWMT-26Q2-A",
        created_at=datetime(2026, 5, 20, 20, 0, 0, tzinfo=UTC),
        options={},
    )
    second = store.create_run(
        scope=scope,
        market_ticker="KXEARNINGSMENTIONWMT-26Q2-B",
        created_at=datetime(2026, 5, 20, 21, 0, 0, tzinfo=UTC),
        options={},
    )
    third = store.create_run(
        scope=scope,
        market_ticker="KXEARNINGSMENTIONWMT-26Q2-C",
        created_at=datetime(2026, 5, 20, 22, 0, 0, tzinfo=UTC),
        options={},
    )

    store.update_status(scope=scope, run_id=first.run_id, status="completed")
    store.update_status(scope=scope, run_id=second.run_id, status="failed")
    store.update_status(scope=scope, run_id=third.run_id, status="completed")

    latest = store.latest_completed_run(scope)

    assert latest is not None
    assert latest.run_id == third.run_id
