import threading
import time

from kalorie2.execution.config import LiveTradingConfig
from kalorie2.execution.supervisor import TraderSpec, TraderSupervisor
from kalorie2.execution.trader import TraderRunSummary


class FakeRunner:
    def __init__(self, spec: TraderSpec, *, mode: str = "dry_run") -> None:
        self.spec = spec
        self._config = LiveTradingConfig(mode=mode)
        self.calls = 0
        self._lock = threading.Lock()

    @property
    def config(self) -> LiveTradingConfig:
        return self._config

    def run_once(self) -> TraderRunSummary:
        with self._lock:
            self.calls += 1
        return TraderRunSummary(
            mode=self._config.mode,
            poll_id=f"pass-{self.calls}",
            kill_switch_active=False,
            evaluated=1,
        )


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _supervisor() -> tuple[TraderSupervisor, list[FakeRunner]]:
    runners: list[FakeRunner] = []

    def factory(spec: TraderSpec):
        runner = FakeRunner(spec)
        runners.append(runner)
        return runner

    return TraderSupervisor(trader_factory=factory), runners


def test_starts_runs_passes_and_reports_running_status() -> None:
    supervisor, runners = _supervisor()
    spec = TraderSpec(model_name="kalorie-v2", risk_preset_id="balanced", interval_seconds=1)

    supervisor.start(spec)
    try:
        assert _wait_for(lambda: supervisor.status().pass_count >= 1)
        status = supervisor.status()
        assert status.running is True
        assert status.spec == spec
        assert status.mode == "dry_run"
        assert status.pass_count >= 1
    finally:
        supervisor.stop()

    assert supervisor.status().running is False


def test_restart_with_changes_swaps_the_running_spec() -> None:
    supervisor, runners = _supervisor()
    supervisor.start(TraderSpec(model_name="kalorie-v2", risk_preset_id="balanced"))
    try:
        assert _wait_for(lambda: len(runners) == 1)
        supervisor.restart(TraderSpec(model_name="kalorie-v3", risk_preset_id="aggressive"))
        assert _wait_for(lambda: len(runners) == 2)
        status = supervisor.status()
        assert status.spec is not None
        assert status.spec.model_name == "kalorie-v3"
        assert status.spec.risk_preset_id == "aggressive"
    finally:
        supervisor.stop()


def test_stop_when_not_running_is_idempotent() -> None:
    supervisor, _ = _supervisor()
    supervisor.stop()
    assert supervisor.status().running is False


def test_double_start_is_rejected() -> None:
    supervisor, _ = _supervisor()
    supervisor.start(TraderSpec(model_name="kalorie-v2", risk_preset_id="balanced"))
    try:
        started_again = supervisor.start(
            TraderSpec(model_name="kalorie-v2", risk_preset_id="balanced")
        )
        assert started_again is False
    finally:
        supervisor.stop()


def test_on_start_hook_runs_synchronously_before_loop() -> None:
    rescored: list[str] = []
    runners: list[FakeRunner] = []

    def factory(spec: TraderSpec):
        # The rescore must have already happened before the runner is built.
        assert rescored[-1] == spec.model_name
        runner = FakeRunner(spec)
        runners.append(runner)
        return runner

    supervisor = TraderSupervisor(
        trader_factory=factory,
        on_start=lambda spec: rescored.append(spec.model_name),
    )
    spec = TraderSpec(model_name="kalorie-v2", risk_preset_id="balanced", interval_seconds=1)
    supervisor.start(spec)
    try:
        assert rescored == ["kalorie-v2"]
    finally:
        supervisor.stop()

    # Restart re-scores again with the new model.
    supervisor.restart(
        TraderSpec(model_name="kalorie-v3", risk_preset_id="aggressive", interval_seconds=1)
    )
    try:
        assert rescored == ["kalorie-v2", "kalorie-v3"]
    finally:
        supervisor.stop()


def test_on_start_failure_is_non_fatal_and_recorded() -> None:
    def factory(spec: TraderSpec):
        return FakeRunner(spec)

    def boom(_spec: TraderSpec) -> None:
        raise RuntimeError("scorer offline")

    supervisor = TraderSupervisor(trader_factory=factory, on_start=boom)
    started = supervisor.start(
        TraderSpec(model_name="kalorie-v2", risk_preset_id="balanced", interval_seconds=1)
    )
    try:
        assert started is True
        assert supervisor.status().running is True
        assert "rescore_on_start_failed" in (supervisor.status().startup_error or "")
    finally:
        supervisor.stop()
