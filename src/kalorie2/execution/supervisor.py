from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from kalorie2.execution.config import LiveTradingConfig
from kalorie2.execution.trader import TraderRunSummary


@dataclass(frozen=True)
class TraderSpec:
    """The model + risk preset the running bot is committed to.

    This is the *running* config snapshot. The UI holds its own *staged*
    selection; the bot only adopts it on start/restart, never live."""

    model_name: str
    risk_preset_id: str
    interval_seconds: int = 15


class TraderRunner(Protocol):
    @property
    def config(self) -> LiveTradingConfig: ...
    def run_once(self) -> TraderRunSummary: ...


class TraderStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    running: bool
    mode: str
    spec: TraderSpec | None
    started_at: datetime | None = None
    last_pass_at: datetime | None = None
    pass_count: int = 0
    last_error: str | None = None
    startup_error: str | None = None
    last_summary: dict[str, Any] | None = None


class TraderSupervisor:
    """Owns a single background trader thread inside the web app process.

    Trading only happens in this loop; request handlers merely start/stop it and
    read status. ``trader_factory`` builds a fresh runner for each run so a
    restart picks up the new spec without leaking state across runs."""

    def __init__(
        self,
        *,
        trader_factory: Callable[[TraderSpec], TraderRunner],
        on_start: Callable[[TraderSpec], None] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._trader_factory = trader_factory
        # Runs synchronously on every start/restart before the loop launches.
        # Used to re-score all markets with the committed model so the bot never
        # trades on stale, other-model signals.
        self._on_start = on_start
        self._now = now or (lambda: datetime.now(tz=UTC))
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._spec: TraderSpec | None = None
        self._mode = "off"
        self._started_at: datetime | None = None
        self._last_pass_at: datetime | None = None
        self._pass_count = 0
        self._last_error: str | None = None
        self._startup_error: str | None = None
        self._last_summary: dict[str, Any] | None = None

    def start(self, spec: TraderSpec) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            startup_error: str | None = None
            if self._on_start is not None:
                # Synchronous re-score before the bot can trade. Failure is
                # non-fatal: the runner's running-model guard fails closed and
                # skips stale signals until a later re-score lands.
                try:
                    self._on_start(spec)
                except Exception as exc:  # noqa: BLE001
                    startup_error = f"rescore_on_start_failed: {exc}"
            runner = self._trader_factory(spec)
            self._spec = spec
            self._mode = runner.config.mode
            self._started_at = self._now()
            self._pass_count = 0
            self._last_pass_at = None
            self._last_error = None
            self._startup_error = startup_error
            self._last_summary = None
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run_loop,
                args=(runner, spec),
                name="kalorie2-live-trader",
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            self._stop_event.set()
        if thread is not None:
            thread.join(timeout=10.0)
        with self._lock:
            self._thread = None

    def restart(self, spec: TraderSpec) -> bool:
        self.stop()
        return self.start(spec)

    def status(self) -> TraderStatus:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            return TraderStatus(
                running=running,
                mode=self._mode,
                spec=self._spec if running else None,
                started_at=self._started_at if running else None,
                last_pass_at=self._last_pass_at,
                pass_count=self._pass_count,
                last_error=self._last_error,
                startup_error=self._startup_error,
                last_summary=self._last_summary,
            )

    def _run_loop(self, runner: TraderRunner, spec: TraderSpec) -> None:
        while not self._stop_event.is_set():
            try:
                summary = runner.run_once()
            except Exception as exc:  # noqa: BLE001 - keep the loop alive across failures
                with self._lock:
                    self._last_error = str(exc)
            else:
                with self._lock:
                    self._last_summary = summary.model_dump(mode="json")
                    self._last_pass_at = self._now()
                    self._pass_count += 1
                    self._last_error = None
            self._stop_event.wait(spec.interval_seconds)
