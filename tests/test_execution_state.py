from datetime import UTC, datetime, timedelta
from pathlib import Path

from kalorie2.execution.state import (
    ExecutionStateStore,
    date_key,
    deterministic_client_order_id,
)

NOW = datetime(2026, 5, 29, 12, 0, tzinfo=UTC)


def test_rescore_debounce_blocks_within_interval_then_allows(tmp_path: Path) -> None:
    store = ExecutionStateStore(root=tmp_path)

    assert store.should_rescore_event(
        "EVT-A", now=NOW, min_interval_seconds=600, max_per_day=20
    )
    store.record_rescore("EVT-A", now=NOW)

    # Within the 10-minute window the same event is debounced.
    assert not store.should_rescore_event(
        "EVT-A", now=NOW + timedelta(minutes=5), min_interval_seconds=600, max_per_day=20
    )
    # A different event is unaffected.
    assert store.should_rescore_event(
        "EVT-B", now=NOW + timedelta(minutes=5), min_interval_seconds=600, max_per_day=20
    )
    # After the interval the same event is allowed again.
    assert store.should_rescore_event(
        "EVT-A", now=NOW + timedelta(minutes=11), min_interval_seconds=600, max_per_day=20
    )


def test_rescore_daily_cap_blocks_when_reached(tmp_path: Path) -> None:
    store = ExecutionStateStore(root=tmp_path)
    day = date_key(NOW)

    for i in range(20):
        store.record_rescore(f"EVT-{i}", now=NOW)

    assert store.rescore_count(day) == 20
    # Global daily cap reached: even a brand-new event is blocked.
    assert not store.should_rescore_event(
        "EVT-NEW", now=NOW, min_interval_seconds=600, max_per_day=20
    )
    # The cap resets the next day.
    assert store.should_rescore_event(
        "EVT-NEW", now=NOW + timedelta(days=1), min_interval_seconds=600, max_per_day=20
    )


def test_rescore_state_persists_across_store_instances(tmp_path: Path) -> None:
    ExecutionStateStore(root=tmp_path).record_rescore("EVT-A", now=NOW)

    reopened = ExecutionStateStore(root=tmp_path)
    assert reopened.last_rescore_at("EVT-A") == NOW
    assert reopened.rescore_count(date_key(NOW)) == 1


def test_halt_and_unhalt_contract_roundtrip(tmp_path: Path) -> None:
    store = ExecutionStateStore(root=tmp_path)

    assert store.is_halted("MKT") is False
    store.halt_contract("MKT", reason="price_swing", now=NOW)
    assert store.is_halted("MKT") is True
    assert "MKT" in store.halted_contracts()

    store.unhalt_contract("MKT")
    assert store.is_halted("MKT") is False


def test_state_writes_are_durable_across_store_instances(tmp_path: Path) -> None:
    store = ExecutionStateStore(root=tmp_path)
    store.halt_contract("MKT", reason="price_swing", now=NOW)
    store.set_observed_mid("MKT", 0.41)

    reopened = ExecutionStateStore(root=tmp_path)

    assert reopened.is_halted("MKT") is True
    assert reopened.observed_mid("MKT") == 0.41
    assert (tmp_path / "state.json").exists()


def test_deterministic_client_order_id_is_stable_and_input_sensitive() -> None:
    first = deterministic_client_order_id(
        model_name="kalorie-v2",
        poll_id="20260529-120000",
        market_ticker="MKT",
        side="NO",
        limit_price_cents=58,
        count=8,
        date_key="20260529",
    )
    same = deterministic_client_order_id(
        model_name="kalorie-v2",
        poll_id="20260529-120000",
        market_ticker="MKT",
        side="NO",
        limit_price_cents=58,
        count=8,
        date_key="20260529",
    )
    different = deterministic_client_order_id(
        model_name="kalorie-v2",
        poll_id="20260529-120000",
        market_ticker="MKT",
        side="NO",
        limit_price_cents=58,
        count=9,
        date_key="20260529",
    )

    assert first == same
    assert first != different


def test_seen_client_order_id_prevents_duplicates(tmp_path: Path) -> None:
    store = ExecutionStateStore(root=tmp_path)

    assert store.has_seen_client_order_id("abc") is False
    store.record_client_order_id("abc")
    assert store.has_seen_client_order_id("abc") is True


def test_daily_order_counters_track_total_and_per_contract(tmp_path: Path) -> None:
    store = ExecutionStateStore(root=tmp_path)
    key = "20260529"

    assert store.daily_orders(key) == 0
    assert store.daily_orders_for_contract(key, "MKT") == 0

    store.record_order(key, "MKT")
    store.record_order(key, "MKT")
    store.record_order(key, "OTHER")

    assert store.daily_orders(key) == 3
    assert store.daily_orders_for_contract(key, "MKT") == 2
    assert store.daily_orders_for_contract(key, "OTHER") == 1


def test_daily_loss_accumulates(tmp_path: Path) -> None:
    store = ExecutionStateStore(root=tmp_path)
    key = "20260529"

    store.add_daily_loss(key, 4.5)
    store.add_daily_loss(key, 3.25)

    assert store.daily_loss(key) == 7.75


def test_audit_log_appends_and_reads_records(tmp_path: Path) -> None:
    store = ExecutionStateStore(root=tmp_path)

    store.record_audit({"event": "rejected", "market_ticker": "MKT", "reason": "event_cutoff"})
    store.record_audit({"event": "submitted", "market_ticker": "MKT", "order_id": "1"})

    records = store.read_audit()
    assert len(records) == 2
    assert records[0]["event"] == "rejected"
    assert records[1]["event"] == "submitted"
    assert store.last_audit()["event"] == "submitted"
    assert (tmp_path / "audit-log.jsonl").exists()


def test_kill_switch_activate_and_clear(tmp_path: Path) -> None:
    store = ExecutionStateStore(root=tmp_path)

    assert store.kill_switch_active() is False
    store.activate_kill_switch(reason="manual stop")
    assert store.kill_switch_active() is True

    store.clear_kill_switch()
    assert store.kill_switch_active() is False


def test_date_key_formats_utc_day() -> None:
    assert date_key(NOW) == "20260529"
