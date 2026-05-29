import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from kalorie2.execution.config import LiveTradingConfig
from kalorie2.execution.state import ExecutionStateStore
from kalorie2.execution.supervisor import TraderSpec, TraderSupervisor
from kalorie2.execution.trader import TraderRunSummary
from kalorie2.webapi.main import create_app


class FakeRunner:
    def __init__(self, spec: TraderSpec) -> None:
        self._config = LiveTradingConfig(mode="dry_run")
        self.calls = 0
        self._lock = threading.Lock()

    @property
    def config(self) -> LiveTradingConfig:
        return self._config

    def run_once(self) -> TraderRunSummary:
        with self._lock:
            self.calls += 1
        return TraderRunSummary(mode="dry_run", poll_id="p", kill_switch_active=False)


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _client(tmp_path: Path) -> tuple[TestClient, TraderSupervisor, ExecutionStateStore]:
    execution_state = ExecutionStateStore(root=tmp_path / "execution")
    supervisor = TraderSupervisor(trader_factory=lambda spec: FakeRunner(spec))
    app = create_app(
        models_root=tmp_path / "models",
        poll_cache_root=tmp_path / "cache",
        env_path=tmp_path / "missing.env",
        trader_supervisor=supervisor,
        execution_state=execution_state,
    )
    return TestClient(app), supervisor, execution_state


def test_status_reports_stopped_initially(tmp_path: Path) -> None:
    client, supervisor, _ = _client(tmp_path)
    response = client.get("/api/trader/status")
    assert response.status_code == 200
    assert response.json()["status"]["running"] is False


def test_start_stop_cycle(tmp_path: Path) -> None:
    client, supervisor, _ = _client(tmp_path)
    try:
        started = client.post(
            "/api/trader/start",
            json={"model_name": "kalorie-v2", "risk_preset_id": "balanced", "interval_seconds": 1},
        )
        assert started.status_code == 200
        assert started.json()["status"]["running"] is True
        assert _wait_for(lambda: supervisor.status().pass_count >= 1)
    finally:
        stopped = client.post("/api/trader/stop")
    assert stopped.status_code == 200
    assert stopped.json()["status"]["running"] is False


def test_restart_swaps_running_spec(tmp_path: Path) -> None:
    client, supervisor, _ = _client(tmp_path)
    client.post(
        "/api/trader/start",
        json={"model_name": "kalorie-v2", "risk_preset_id": "balanced"},
    )
    try:
        response = client.post(
            "/api/trader/restart",
            json={"model_name": "kalorie-v3", "risk_preset_id": "aggressive"},
        )
        assert response.status_code == 200
        spec = response.json()["status"]["spec"]
        assert spec["model_name"] == "kalorie-v3"
        assert spec["risk_preset_id"] == "aggressive"
    finally:
        client.post("/api/trader/stop")


def test_activity_returns_audit_records(tmp_path: Path) -> None:
    client, _, execution_state = _client(tmp_path)
    execution_state.record_audit({"event": "dry_run_approved", "market_ticker": "MKT-1"})
    execution_state.record_audit({"event": "submitted", "market_ticker": "MKT-2"})

    response = client.get("/api/trader/activity", params={"limit": 10})
    assert response.status_code == 200
    events = response.json()["activity"]
    assert [item["event"] for item in events][-2:] == ["dry_run_approved", "submitted"]


def test_default_supervisor_rescore_on_start_is_wired_to_rescorer(tmp_path: Path) -> None:
    # Build the app with its real default supervisor (no injected one) so the
    # on-start re-score hook is wired, then swap in a fake rescorer to observe it.
    app = create_app(
        models_root=tmp_path / "models",
        poll_cache_root=tmp_path / "cache",
        env_path=tmp_path / "missing.env",
        execution_state=ExecutionStateStore(root=tmp_path / "execution"),
    )

    calls: list[str] = []

    class FakeRescorer:
        def rescore_all(self, *, model_name: str) -> None:
            calls.append(model_name)

        def rescore_event(self, *, model_name: str, event_ticker: str) -> None:  # pragma: no cover
            pass

    app.state.rescorer = FakeRescorer()
    supervisor: TraderSupervisor = app.state.trader_supervisor
    supervisor._on_start(TraderSpec(model_name="kalorie-v9", risk_preset_id="balanced"))

    assert calls == ["kalorie-v9"]


def test_kill_switch_toggle(tmp_path: Path) -> None:
    client, _, execution_state = _client(tmp_path)
    engaged = client.post("/api/trader/kill", json={"reason": "panic"})
    assert engaged.status_code == 200
    assert execution_state.kill_switch_active() is True

    resumed = client.post("/api/trader/resume")
    assert resumed.status_code == 200
    assert execution_state.kill_switch_active() is False
