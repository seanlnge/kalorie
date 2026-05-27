import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
from typer.testing import CliRunner

from kalorie2.market_poller import (
    ActiveMarketPoller,
    ActiveMarketRow,
    CachedSavedModelMarketScorer,
    KalshiActiveMarketSource,
    MarketPollCacheStore,
    OpenAIWebEvidenceSource,
    PollPredictionRow,
    _load_env_file,
    app,
    default_poll_cache_root,
    normalize_active_market,
    preferred_model_name,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_bundle(root: Path, name: str = "unit-model") -> Path:
    model_dir = root / name
    (model_dir / "runtime").mkdir(parents=True)
    (model_dir / "training").mkdir()
    (model_dir / "README.md").write_text("# Unit Model", encoding="utf-8")
    _write_json(
        model_dir / "artifacts" / "model.json",
        {
            "model_name": name,
            "model_type": "market_anchored_linear_residual",
            "trained_at": "2026-05-26T02:39:41+00:00",
            "training_summary": {"row_count": 10, "event_count": 2, "feature_count": 3},
            "model": {"weights": {"alpha": 0.1, "beta": -0.2}},
        },
    )
    _write_json(
        model_dir / "artifacts" / "feature-schema.json",
        {"feature_names": ["alpha", "beta", "gamma"], "nonzero_weights": {"alpha": 0.1}},
    )
    _write_json(
        model_dir / "artifacts" / "training-manifest.json",
        {
            "model_name": name,
            "training_corpus": {
                "saved_csv": "training/rows.csv",
                "web_evidence_packet_count": 2,
            },
        },
    )
    _write_json(
        model_dir / "artifacts" / "evaluation-reports.json",
        {
            "full_web_backtest": {
                "summary": {"trades": 4, "total_pnl": 1.25, "roi_on_cost": 0.125}
            }
        },
    )
    (model_dir / "runtime" / "model_runtime.py").write_text(
        "print('not used in this test')\n",
        encoding="utf-8",
    )
    return model_dir


class StubMarketSource:
    def list_active_markets(self) -> list[ActiveMarketRow]:
        return [
            ActiveMarketRow(
                market_ticker="KXEARNINGSMENTIONAAPL-26APR30-AI",
                event_ticker="KXEARNINGSMENTIONAAPL-26APR30",
                series_ticker="KXEARNINGSMENTIONAAPL",
                event_title="What will Apple say during their next earnings call?",
                market_title="What will Apple say during their next earnings call? - AI",
                target_phrase="AI",
                yes_bid=0.37,
                yes_ask=0.4,
                yes_mid=0.385,
                volume=123,
            )
        ]


class StubScorer:
    def score_active_markets(
        self,
        markets: list[ActiveMarketRow],
        *,
        model_name: str,
    ) -> list[PollPredictionRow]:
        market = markets[0]
        return [
            PollPredictionRow(
                market_ticker=market.market_ticker,
                event_ticker=market.event_ticker,
                target_phrase=market.target_phrase,
                model_name=model_name,
                model_probability=0.31,
                market_probability=market.yes_mid,
                yes_bid=market.yes_bid,
                yes_ask=market.yes_ask,
                residual_delta=-0.29,
                side="NO",
                edge=0.06,
                cost=0.63,
                volume=market.volume,
            )
        ]


def test_normalize_active_market_builds_runtime_compatible_market_row() -> None:
    row = normalize_active_market(
        event_payload={
            "event_ticker": "KXEARNINGSMENTIONAAPL-26APR30",
            "series_ticker": "KXEARNINGSMENTIONAAPL",
            "event_title": "What will Apple say during their next earnings call?",
        },
        market_payload={
            "ticker": "KXEARNINGSMENTIONAAPL-26APR30-AI",
            "title": "What will Apple say during their next earnings call? - AI",
            "custom_strike": {"Word": "AI"},
            "close_time": "2026-04-30T20:00:00Z",
            "yes_bid": 37,
            "yes_ask": 40,
            "volume": 123,
        },
    )

    assert row.market_ticker == "KXEARNINGSMENTIONAAPL-26APR30-AI"
    assert row.event_ticker == "KXEARNINGSMENTIONAAPL-26APR30"
    assert row.target_phrase == "AI"
    assert row.yes_bid == 0.37
    assert row.yes_ask == 0.4
    assert row.yes_mid == 0.385
    assert row.event_datetime == "2026-04-30T20:00:00Z"
    assert row.to_runtime_row()["preclose_yes_mid"] == "0.385"
    assert row.to_runtime_row()["word_said"] == "AI"


def test_poll_cache_store_writes_latest_history_and_trade_files(tmp_path: Path) -> None:
    store = MarketPollCacheStore(root=tmp_path)
    snapshot = ActiveMarketPoller(
        market_source=StubMarketSource(),
        scorer=StubScorer(),
        cache_store=store,
        now=lambda: datetime(2026, 5, 26, 4, 0, tzinfo=UTC),
    ).run_once(model_name="unit-model")

    latest = store.read_latest_snapshot()
    trades = store.read_latest_trades()

    assert snapshot.poll_id == "20260526-040000"
    assert latest is not None
    assert latest.model_name == "unit-model"
    assert latest.market_count == 1
    assert latest.trade_count == 1
    assert trades == snapshot.trade_rows
    assert (tmp_path / "history" / "20260526-040000.json").exists()


def test_poll_snapshot_separates_all_predictions_from_trade_opportunities(tmp_path: Path) -> None:
    snapshot = ActiveMarketPoller(
        market_source=StubMarketSource(),
        scorer=StubScorer(),
        cache_store=MarketPollCacheStore(root=tmp_path),
        now=lambda: datetime(2026, 5, 26, 4, 0, tzinfo=UTC),
    ).run_once(model_name="unit-model")

    assert len(snapshot.prediction_rows) == 1
    assert len(snapshot.trade_rows) == 1
    assert snapshot.trade_rows[0].side == "NO"
    assert snapshot.trade_rows[0].edge == 0.06


def test_market_poller_cli_exposes_once_and_loop_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "once" in result.stdout
    assert "loop" in result.stdout


def test_active_market_source_falls_back_to_event_markets_when_search_has_no_markets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search/series":
            return httpx.Response(
                200,
                json={
                    "current_page": [
                        {
                            "event_ticker": "KXEARNINGSMENTIONAAPL-26APR30",
                            "series_ticker": "KXEARNINGSMENTIONAAPL",
                            "event_title": "What will Apple say during their next earnings call?",
                        }
                    ]
                },
            )
        if request.url.path == "/trade-api/v2/markets":
            return httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXEARNINGSMENTIONAAPL-26APR30-AI",
                            "event_ticker": "KXEARNINGSMENTIONAAPL-26APR30",
                            "series_ticker": "KXEARNINGSMENTIONAAPL",
                            "title": "What will Apple say during their next earnings call? - AI",
                            "custom_strike": {"Word": "AI"},
                            "yes_bid": 37,
                            "yes_ask": 40,
                            "volume": 123,
                        }
                    ]
                },
            )
        return httpx.Response(404)

    source = KalshiActiveMarketSource(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_pages=1,
    )

    rows = source.list_active_markets()

    assert [row.market_ticker for row in rows] == ["KXEARNINGSMENTIONAAPL-26APR30-AI"]


def test_active_market_source_uses_open_market_scan_for_missing_series_events() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search/series":
            return httpx.Response(200, json={"current_page": []})
        if request.url.path == "/trade-api/v2/markets":
            return httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXEARNINGSMENTIONANF-26MAY27-AI",
                            "event_ticker": "KXEARNINGSMENTIONANF-26MAY27",
                            "series_ticker": "KXEARNINGSMENTIONANF",
                            "event_title": (
                                "What will Abercrombie say during their next earnings call?"
                            ),
                            "title": (
                                "What will Abercrombie say during their next earnings call? - AI"
                            ),
                            "custom_strike": {"Word": "AI"},
                            "close_time": "2026-05-27T20:00:00Z",
                            "yes_bid": 47,
                            "yes_ask": 52,
                            "volume": 321,
                        }
                    ]
                },
            )
        return httpx.Response(404)

    source = KalshiActiveMarketSource(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_pages=1,
    )

    rows = source.list_active_markets()

    assert [row.market_ticker for row in rows] == ["KXEARNINGSMENTIONANF-26MAY27-AI"]
    assert rows[0].event_datetime == "2026-05-27T20:00:00Z"


def test_active_market_source_hydrates_event_title_and_time_from_get_event() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search/series":
            return httpx.Response(200, json={"current_page": []})
        if request.url.path == "/trade-api/v2/events/KXEARNINGSMENTIONANF-26MAY27":
            return httpx.Response(
                200,
                json={
                    "event": {
                        "event_ticker": "KXEARNINGSMENTIONANF-26MAY27",
                        "series_ticker": "KXEARNINGSMENTIONANF",
                        "title": "Abercrombie earnings call",
                        "sub_title": "On May 27, 2026",
                    }
                },
            )
        if request.url.path == "/trade-api/v2/events/KXEARNINGSMENTIONBBY-26MAY29":
            return httpx.Response(
                200,
                json={
                    "event": {
                        "event_ticker": "KXEARNINGSMENTIONBBY-26MAY29",
                        "series_ticker": "KXEARNINGSMENTIONBBY",
                        "title": "Best Buy earnings call",
                        "sub_title": "On May 29, 2026",
                    }
                },
            )
        if request.url.path == "/trade-api/v2/markets":
            return httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXEARNINGSMENTIONBBY-26MAY29-AI",
                            "event_ticker": "KXEARNINGSMENTIONBBY-26MAY29",
                            "series_ticker": "KXEARNINGSMENTIONBBY",
                            "event_title": "Stale market title",
                            "title": "Best Buy market - AI",
                            "custom_strike": {"Word": "AI"},
                            "close_time": "2026-05-29T20:00:00Z",
                            "yes_bid": 47,
                            "yes_ask": 52,
                            "volume": 500,
                        },
                        {
                            "ticker": "KXEARNINGSMENTIONANF-26MAY27-AI",
                            "event_ticker": "KXEARNINGSMENTIONANF-26MAY27",
                            "series_ticker": "KXEARNINGSMENTIONANF",
                            "event_title": "Stale market title",
                            "title": "Abercrombie market - AI",
                            "custom_strike": {"Word": "AI"},
                            "close_time": "2026-05-27T20:00:00Z",
                            "yes_bid": 47,
                            "yes_ask": 52,
                            "volume": 100,
                        },
                    ]
                },
            )
        return httpx.Response(404)

    source = KalshiActiveMarketSource(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_pages=1,
    )

    rows = source.list_active_markets()

    assert [row.event_ticker for row in rows] == [
        "KXEARNINGSMENTIONANF-26MAY27",
        "KXEARNINGSMENTIONBBY-26MAY29",
    ]
    assert rows[0].event_title == "Abercrombie earnings call"
    assert rows[0].event_datetime == "2026-05-27T00:00:00Z"


def test_default_poll_cache_root_is_inside_kalorie2_artifacts_runtime() -> None:
    assert default_poll_cache_root().parts[-3:] == ("artifacts", "runtime", "workstation")
    assert default_poll_cache_root().parent.parent.parent.name == "kalorie2"


def test_preferred_model_name_uses_newest_model_when_available(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "kalorie-v2")
    newer_dir = _write_bundle(tmp_path, "kalorie-v3")
    _write_json(
        newer_dir / "artifacts" / "model-card.json",
        {
            "model_name": "kalorie-v3",
            "model_type": "test",
            "model_version": 3,
            "default_execution_policy": "no_only",
            "default_margin": 0.02,
            "training_data": {},
            "feature_set": {},
            "evaluation_splits": [],
        },
    )

    assert preferred_model_name(tmp_path, None) == "kalorie-v3"
    assert preferred_model_name(tmp_path, "explicit-model") == "explicit-model"


def test_load_env_file_sets_missing_keys_only(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=test-key\nEXISTING=from-file\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EXISTING", "kept")

    _load_env_file(env_path)

    assert Path(env_path).exists()
    assert __import__("os").environ["OPENAI_API_KEY"] == "test-key"
    assert __import__("os").environ["EXISTING"] == "kept"


def test_openai_web_evidence_source_fetches_per_event(monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_fetch(
        *,
        event_ticker: str,
        company_name: str,
        cutoff_time_iso: str,
        target_phrases: list[str],
        model: str,
        timeout_seconds: float,
    ):
        calls.append((event_ticker, tuple(sorted(target_phrases))))
        return {
            "event_ticker": event_ticker,
            "company_name": company_name,
            "cutoff_time": cutoff_time_iso,
            "items": [],
        }

    source = OpenAIWebEvidenceSource(
        model="gpt-5.4-mini",
        timeout_seconds=60.0,
        fetch_web_evidence_packet=fake_fetch,
    )
    markets = [
        ActiveMarketRow(
            market_ticker="E1-A",
            event_ticker="E1",
            series_ticker="KXEARNINGSMENTIONE1",
            event_title="What will Apple say during their next earnings call?",
            market_title="What will Apple say during their next earnings call? - AI",
            target_phrase="AI",
            yes_bid=0.3,
            yes_ask=0.4,
            yes_mid=0.35,
        ),
        ActiveMarketRow(
            market_ticker="E1-B",
            event_ticker="E1",
            series_ticker="KXEARNINGSMENTIONE1",
            event_title="What will Apple say during their next earnings call?",
            market_title="What will Apple say during their next earnings call? - Tariff",
            target_phrase="Tariff",
            yes_bid=0.3,
            yes_ask=0.4,
            yes_mid=0.35,
        ),
        ActiveMarketRow(
            market_ticker="E2-A",
            event_ticker="E2",
            series_ticker="KXEARNINGSMENTIONE2",
            event_title="What will Costco say during their next earnings call?",
            market_title="What will Costco say during their next earnings call? - Membership",
            target_phrase="Membership",
            yes_bid=0.3,
            yes_ask=0.4,
            yes_mid=0.35,
        ),
    ]

    packets = source.fetch_packets(markets)

    assert sorted(packets) == ["E1", "E2"]
    assert ("E1", ("ai", "tariff")) in calls
    assert ("E2", ("membership",)) in calls


def test_cached_saved_model_market_scorer_passes_live_packets_to_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_bundle(tmp_path, "unit-model")
    captured: list[dict[str, object] | None] = []

    class FakeRuntimeScorer:
        def __init__(self, model_dir: Path) -> None:
            self.model_dir = model_dir

        def score_row_dict(self, row, *, web_evidence_by_event=None):
            captured.append(web_evidence_by_event)
            return type(
                "Score",
                (),
                {
                    "model_dump": lambda self: {
                        "market_ticker": row["market_ticker"],
                        "event_ticker": row["event_ticker"],
                        "probability": 0.4,
                        "market_probability": 0.5,
                        "residual_delta": -0.1,
                        "trade_decision": {"side": "NO", "cost": 0.5, "edge": 0.1},
                    }
                },
            )()

    class StubEvidence:
        def fetch_packets(self, markets):
            return {
                market.event_ticker: {
                    "event_ticker": market.event_ticker,
                    "company_name": "Example",
                    "cutoff_time": "2026-05-26T12:00:00Z",
                    "items": [],
                }
                for market in markets
            }

    monkeypatch.setattr("kalorie2.market_poller.CachedRuntimeSavedModelScorer", FakeRuntimeScorer)
    scorer = CachedSavedModelMarketScorer(
        models_root=tmp_path,
        web_evidence_source=StubEvidence(),
    )
    rows = StubMarketSource().list_active_markets()

    predictions = scorer.score_active_markets(rows, model_name="unit-model")

    assert len(predictions) == 1
    assert captured and "KXEARNINGSMENTIONAAPL-26APR30" in captured[0]
