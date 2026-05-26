from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class EventScope:
    company_symbol: str
    event_key: str

    def normalized_company_symbol(self) -> str:
        return self.company_symbol.upper().strip()


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_dir: Path
    market_ticker: str
    created_at: datetime
    status: str


class RunStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def event_root_dir(self, scope: EventScope) -> Path:
        return (
            self._root
            / "events"
            / scope.normalized_company_symbol()
            / scope.event_key.strip()
        )

    def event_data_dir(self, scope: EventScope) -> Path:
        return self.event_root_dir(scope) / "data"

    def event_runs_dir(self, scope: EventScope) -> Path:
        return self.event_root_dir(scope) / "runs"

    def create_run(
        self,
        *,
        scope: EventScope,
        market_ticker: str,
        created_at: datetime | None = None,
        options: dict[str, object] | None = None,
    ) -> RunRecord:
        created = created_at or datetime.now(tz=UTC)
        run_id = f"{created:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        run_dir = self.event_runs_dir(scope) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        self.event_data_dir(scope).mkdir(parents=True, exist_ok=True)
        (run_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        payload = {
            "run_id": run_id,
            "market_ticker": market_ticker,
            "created_at": created.isoformat(),
            "options": options or {},
            "event_scope": {
                "company_symbol": scope.normalized_company_symbol(),
                "event_key": scope.event_key.strip(),
            },
        }
        self._write_json(run_dir / "inputs.json", payload)
        self._write_json(
            run_dir / "job_status.json",
            {
                "run_id": run_id,
                "status": "queued",
                "updated_at": created.isoformat(),
            },
        )
        self._write_json(
            run_dir / "summary.json",
            {
                "run_id": run_id,
                "market_ticker": market_ticker,
                "warnings": [],
                "metrics": {},
            },
        )
        (run_dir / "job_log.txt").write_text("", encoding="utf-8")
        return RunRecord(
            run_id=run_id,
            run_dir=run_dir,
            market_ticker=market_ticker,
            created_at=created,
            status="queued",
        )

    def update_status(
        self,
        *,
        scope: EventScope,
        run_id: str,
        status: str,
        message: str | None = None,
    ) -> None:
        run_dir = self.event_runs_dir(scope) / run_id
        status_path = run_dir / "job_status.json"
        payload: dict[str, object]
        if status_path.exists():
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        else:
            payload = {"run_id": run_id}
        payload["status"] = status
        payload["updated_at"] = datetime.now(tz=UTC).isoformat()
        if message:
            payload["message"] = message
        self._write_json(status_path, payload)

    def latest_completed_run(self, scope: EventScope) -> RunRecord | None:
        completed = [run for run in self.list_runs(scope) if run.status == "completed"]
        if not completed:
            return None
        return sorted(completed, key=lambda record: record.created_at)[-1]

    def list_runs(self, scope: EventScope) -> list[RunRecord]:
        runs_dir = self.event_runs_dir(scope)
        if not runs_dir.exists():
            return []
        runs: list[RunRecord] = []
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            status_path = run_dir / "job_status.json"
            inputs_path = run_dir / "inputs.json"
            if not status_path.exists() or not inputs_path.exists():
                continue
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            inputs_payload = json.loads(inputs_path.read_text(encoding="utf-8"))
            created_at = datetime.fromisoformat(str(inputs_payload["created_at"]))
            runs.append(
                RunRecord(
                    run_id=run_dir.name,
                    run_dir=run_dir,
                    market_ticker=str(inputs_payload.get("market_ticker", "")),
                    created_at=created_at,
                    status=str(status_payload.get("status") or "queued"),
                )
            )
        return sorted(runs, key=lambda record: record.created_at, reverse=True)

    def get_run(self, scope: EventScope, run_id: str) -> RunRecord | None:
        for run in self.list_runs(scope):
            if run.run_id == run_id:
                return run
        return None

    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

