from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def date_key(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y%m%d")


def deterministic_client_order_id(
    *,
    model_name: str,
    poll_id: str,
    market_ticker: str,
    side: str,
    limit_price_cents: int,
    count: int,
    date_key: str,
) -> str:
    """Stable id for an intended order so identical intents never double-submit."""

    payload = "|".join(
        [
            model_name,
            poll_id,
            market_ticker,
            side,
            str(limit_price_cents),
            str(count),
            date_key,
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex=digest[:32]))


class ExecutionStateStore:
    """Atomic JSON state plus an append-only audit log for the live trader.

    Every mutation reloads from disk and rewrites atomically so multiple
    short-lived process invocations stay consistent. This is deliberately simple
    rather than fast; the trader runs at most once per minute.
    """

    def __init__(self, *, root: Path) -> None:
        self._root = root

    @property
    def state_path(self) -> Path:
        return self._root / "state.json"

    @property
    def audit_log_path(self) -> Path:
        return self._root / "audit-log.jsonl"

    @property
    def kill_switch_path(self) -> Path:
        return self._root / "kill-switch"

    def is_halted(self, market_ticker: str) -> bool:
        return market_ticker in self._load()["halted_contracts"]

    def halted_contracts(self) -> dict[str, Any]:
        return dict(self._load()["halted_contracts"])

    def halt_contract(self, market_ticker: str, *, reason: str, now: datetime) -> None:
        state = self._load()
        state["halted_contracts"][market_ticker] = {
            "reason": reason,
            "halted_at": now.astimezone(UTC).isoformat(),
        }
        self._save(state)

    def unhalt_contract(self, market_ticker: str) -> None:
        state = self._load()
        state["halted_contracts"].pop(market_ticker, None)
        self._save(state)

    def observed_mid(self, market_ticker: str) -> float | None:
        value = self._load()["observed_mids"].get(market_ticker)
        return float(value) if value is not None else None

    def set_observed_mid(self, market_ticker: str, mid: float) -> None:
        state = self._load()
        state["observed_mids"][market_ticker] = float(mid)
        self._save(state)

    def daily_orders(self, day: str) -> int:
        return int(self._daily(self._load(), day)["orders"])

    def daily_orders_for_contract(self, day: str, market_ticker: str) -> int:
        return int(self._daily(self._load(), day)["orders_by_contract"].get(market_ticker, 0))

    def record_order(self, day: str, market_ticker: str) -> None:
        state = self._load()
        daily = self._daily(state, day)
        daily["orders"] = int(daily["orders"]) + 1
        by_contract = daily["orders_by_contract"]
        by_contract[market_ticker] = int(by_contract.get(market_ticker, 0)) + 1
        self._save(state)

    def daily_loss(self, day: str) -> float:
        return round(float(self._daily(self._load(), day)["loss"]), 2)

    def add_daily_loss(self, day: str, amount: float) -> None:
        state = self._load()
        daily = self._daily(state, day)
        daily["loss"] = round(float(daily["loss"]) + float(amount), 2)
        self._save(state)

    def last_rescore_at(self, event_ticker: str) -> datetime | None:
        value = self._load()["rescores"]["last_at"].get(event_ticker)
        if value is None:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    def rescore_count(self, day: str) -> int:
        return int(self._load()["rescores"]["daily_counts"].get(day, 0))

    def record_rescore(self, event_ticker: str, *, now: datetime) -> None:
        state = self._load()
        rescores = state["rescores"]
        rescores["last_at"][event_ticker] = now.astimezone(UTC).isoformat()
        day = date_key(now)
        counts = rescores["daily_counts"]
        counts[day] = int(counts.get(day, 0)) + 1
        self._save(state)

    def should_rescore_event(
        self,
        event_ticker: str,
        *,
        now: datetime,
        min_interval_seconds: int,
        max_per_day: int,
    ) -> bool:
        """Debounce guard: at most one re-score per event per interval, bounded by
        a global daily cap so a noisy book can't run up unbounded scoring cost."""

        if max_per_day > 0 and self.rescore_count(date_key(now)) >= max_per_day:
            return False
        last = self.last_rescore_at(event_ticker)
        if last is not None and (now - last).total_seconds() < min_interval_seconds:
            return False
        return True

    def has_seen_client_order_id(self, client_order_id: str) -> bool:
        return client_order_id in set(self._load()["seen_client_order_ids"])

    def record_client_order_id(self, client_order_id: str) -> None:
        state = self._load()
        seen = state["seen_client_order_ids"]
        if client_order_id not in seen:
            seen.append(client_order_id)
        self._save(state)

    def record_audit(self, event: dict[str, Any]) -> None:
        record = {"recorded_at": datetime.now(tz=UTC).isoformat(), **event}
        self._root.mkdir(parents=True, exist_ok=True)
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def read_audit(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.audit_log_path.exists():
            return []
        lines = [
            line
            for line in self.audit_log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if limit is not None:
            lines = lines[-limit:]
        return [json.loads(line) for line in lines]

    def last_audit(self) -> dict[str, Any] | None:
        records = self.read_audit(limit=1)
        return records[-1] if records else None

    def kill_switch_active(self) -> bool:
        return self.kill_switch_path.exists()

    def activate_kill_switch(self, *, reason: str) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        payload = {"reason": reason, "activated_at": datetime.now(tz=UTC).isoformat()}
        self.kill_switch_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def clear_kill_switch(self) -> None:
        self.kill_switch_path.unlink(missing_ok=True)

    def _daily(self, state: dict[str, Any], day: str) -> dict[str, Any]:
        daily_map = state["daily"]
        if day not in daily_map:
            daily_map[day] = {"orders": 0, "loss": 0.0, "orders_by_contract": {}}
        return daily_map[day]

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return _empty_state()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return _empty_state()
        state = _empty_state()
        state["halted_contracts"].update(payload.get("halted_contracts", {}))
        state["observed_mids"].update(payload.get("observed_mids", {}))
        state["daily"].update(payload.get("daily", {}))
        state["seen_client_order_ids"].extend(payload.get("seen_client_order_ids", []))
        rescores = payload.get("rescores", {})
        if isinstance(rescores, dict):
            state["rescores"]["last_at"].update(rescores.get("last_at", {}))
            state["rescores"]["daily_counts"].update(rescores.get("daily_counts", {}))
        return state

    def _save(self, state: dict[str, Any]) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        temp_path = self.state_path.with_suffix(f"{self.state_path.suffix}.tmp")
        temp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.state_path)


def _empty_state() -> dict[str, Any]:
    return {
        "halted_contracts": {},
        "observed_mids": {},
        "daily": {},
        "seen_client_order_ids": [],
        "rescores": {"last_at": {}, "daily_counts": {}},
    }
