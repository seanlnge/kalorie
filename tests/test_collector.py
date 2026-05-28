from datetime import UTC, datetime
from decimal import Decimal

import httpx

from kalorie2.collector import (
    HistoricalMentionCollector,
    KalshiMentionClient,
    classify_market_category,
    extract_target_phrase,
    is_mention_market,
    sample_snapshot_hour_offsets,
    select_preclose_snapshot,
)


def test_extract_target_phrase_prefers_custom_strike_word():
    phrase = extract_target_phrase(
        {
            "ticker": "KXEARNINGSMENTIONAAPL-26JUL30-VISI",
            "custom_strike": {"Word": "Vision Pro"},
            "yes_sub_title": "Apple headset",
            "rules_primary": "If Apple headset is said by Tim Cook.",
        }
    )

    assert phrase == "Vision Pro"


def test_is_mention_market_rejects_non_mention_contracts():
    assert is_mention_market(
        {
            "ticker": "KXEARNINGSMENTIONAAPL-26JUL30-VISI",
            "custom_strike": {"Word": "Vision Pro"},
            "title": "What will Apple say during their next earnings call?",
        }
    )
    assert not is_mention_market(
        {
            "ticker": "KXNBAFINAL-26JUN01-BOS",
            "title": "Will Boston win the NBA Finals?",
            "yes_sub_title": "Boston Celtics",
        }
    )


def test_is_mention_market_includes_say_tickers_from_mentions_category():
    assert is_mention_market(
        {
            "ticker": "KXTRUMPSAY-26MAY18-CHIN",
            "event_ticker": "KXTRUMPSAY-26MAY18",
            "title": "Donald Trump remarks",
            "custom_strike": {"Word": "China"},
        }
    )


def test_classify_market_category_uses_series_and_text_signals():
    assert (
        classify_market_category(
            series_ticker="KXEARNINGSMENTIONAAPL",
            event_title="What will Apple say during their next earnings call?",
            market_title="",
        )
        == "earnings"
    )
    assert (
        classify_market_category(
            series_ticker="KXPREZMENTION",
            event_title="What will Trump say during the debate?",
            market_title="",
        )
        == "politics"
    )
    assert (
        classify_market_category(
            series_ticker="KXVANCEMENTION",
            event_title="What will JD Vance say during his remarks in Maine?",
            market_title="",
        )
        == "politics"
    )
    assert (
        classify_market_category(
            series_ticker="KXSPORTSMENTION",
            event_title="What will the coach say after the NBA game?",
            market_title="",
        )
        == "sports"
    )
    assert (
        classify_market_category(
            series_ticker="KXMENTION",
            event_title="What will the host say during the ceremony?",
            market_title="",
        )
        == "other"
    )


def test_select_preclose_snapshot_chooses_latest_bid_ask_before_target():
    target_ts = int(datetime(2026, 5, 22, 12, tzinfo=UTC).timestamp())
    snapshot = select_preclose_snapshot(
        [
            {
                "end_period_ts": target_ts - 120,
                "yes_bid": {"close_dollars": "0.41"},
                "yes_ask": {"close_dollars": "0.47"},
            },
            {
                "end_period_ts": target_ts - 60,
                "yes_bid": {"close_dollars": "0.44"},
                "yes_ask": {"close_dollars": "0.50"},
                "volume": 1200,
                "open_interest": 340,
                "yes_bid_size": 50,
                "yes_ask_size": 65,
            },
            {
                "end_period_ts": target_ts + 60,
                "yes_bid": {"close_dollars": "0.70"},
                "yes_ask": {"close_dollars": "0.75"},
            },
        ],
        target_ts=target_ts,
        max_staleness_seconds=3600,
    )

    assert snapshot is not None
    assert snapshot.yes_bid == Decimal("0.44")
    assert snapshot.yes_ask == Decimal("0.50")
    assert snapshot.yes_mid == Decimal("0.47")
    assert snapshot.candle_end_ts == target_ts - 60
    assert snapshot.staleness_seconds == 60
    assert snapshot.volume == 1200
    assert snapshot.open_interest == 340
    assert snapshot.yes_bid_size == 50
    assert snapshot.yes_ask_size == 65


def test_select_preclose_snapshot_rejects_candles_outside_staleness_limit():
    target_ts = int(datetime(2026, 5, 22, 12, tzinfo=UTC).timestamp())
    snapshot = select_preclose_snapshot(
        [
            {
                "end_period_ts": target_ts - 7200,
                "yes_bid": {"close_dollars": "0.41"},
                "yes_ask": {"close_dollars": "0.47"},
            },
        ],
        target_ts=target_ts,
        max_staleness_seconds=3600,
    )

    assert snapshot is None


def test_select_preclose_snapshot_accepts_historical_candle_close_fields():
    target_ts = int(datetime(2025, 3, 13, 6, tzinfo=UTC).timestamp())

    snapshot = select_preclose_snapshot(
        [
            {
                "end_period_ts": target_ts,
                "yes_bid": {"close": "0.91"},
                "yes_ask": {"close": "0.96"},
            }
        ],
        target_ts=target_ts,
    )

    assert snapshot is not None
    assert snapshot.yes_bid == Decimal("0.91")
    assert snapshot.yes_ask == Decimal("0.96")


def test_sample_snapshot_hour_offsets_is_seeded_unique_and_within_bounds():
    offsets = sample_snapshot_hour_offsets(
        "KXEARNINGSMENTIONAAPL-26JUL30-VISI",
        count=5,
        min_hours=2,
        max_hours=48,
        seed=17,
    )

    assert offsets == sample_snapshot_hour_offsets(
        "KXEARNINGSMENTIONAAPL-26JUL30-VISI",
        count=5,
        min_hours=2,
        max_hours=48,
        seed=17,
    )
    assert offsets != sample_snapshot_hour_offsets(
        "KXEARNINGSMENTIONMSFT-26JUL30-AI",
        count=5,
        min_hours=2,
        max_hours=48,
        seed=17,
    )
    assert len(offsets) == 5
    assert len(set(offsets)) == 5
    assert all(2 <= offset <= 48 for offset in offsets)


def test_collector_builds_rows_from_search_series_and_historical_candles():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/search/series":
            return httpx.Response(
                200,
                json={
                    "current_page": [
                        {
                            "event_ticker": "KXEARNINGSMENTIONAAPL-26JUL30",
                            "event_title": (
                                "What will Apple say during their next earnings call?"
                            ),
                            "series_ticker": "KXEARNINGSMENTIONAAPL",
                            "markets": [
                                {
                                    "ticker": "KXEARNINGSMENTIONAAPL-26JUL30-VISI",
                                    "event_ticker": "KXEARNINGSMENTIONAAPL-26JUL30",
                                    "title": (
                                        "What will Apple say during their next earnings call?"
                                    ),
                                    "custom_strike": {"Word": "Vision Pro"},
                                    "status": "finalized",
                                    "result": "yes",
                                    "close_time": "2026-07-30T20:00:00Z",
                                    "settlement_ts": "2026-07-30T20:30:00Z",
                                }
                            ],
                        }
                    ]
                },
            )
        expected_path = (
            "/trade-api/v2/historical/markets/"
            "KXEARNINGSMENTIONAAPL-26JUL30-VISI/candlesticks"
        )
        if request.url.path == expected_path:
            assert request.url.params["period_interval"] == "1"
            return httpx.Response(
                200,
                json={
                    "candlesticks": [
                        {
                            "end_period_ts": int(
                                datetime(2026, 7, 30, 11, 59, tzinfo=UTC).timestamp()
                            ),
                            "yes_bid": {"close_dollars": "0.62"},
                            "yes_ask": {"close_dollars": "0.68"},
                            "volume": 1000,
                            "open_interest": 250,
                        }
                    ]
                },
            )
        return httpx.Response(404, json={"error": "unexpected request"})

    collector = HistoricalMentionCollector(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_pages=1,
    )

    result = collector.collect()

    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.market_ticker == "KXEARNINGSMENTIONAAPL-26JUL30-VISI"
    assert row.event_phrase == "What will Apple say during their next earnings call?"
    assert row.market_name == "What will Apple say during their next earnings call? - Vision Pro"
    assert row.word_said == "Vision Pro"
    assert row.market_category == "earnings"
    assert row.snapshot_target_time == datetime(2026, 7, 30, 12, tzinfo=UTC)
    assert row.preclose_yes_bid == Decimal("0.62")
    assert row.preclose_yes_ask == Decimal("0.68")
    assert row.preclose_yes_mid == Decimal("0.65")
    assert row.final_outcome == "yes"
    assert row.snapshot_staleness_seconds == 60
    assert row.preclose_volume == 1000
    assert row.preclose_open_interest == 250
    assert "raw_market" not in row.model_dump(mode="json")
    assert "raw_candle" not in row.model_dump(mode="json")
    assert result.stats["rows_written"] == 1
    assert any("/historical/markets/" in request.url.path for request in requests)


def test_collector_expands_market_to_seeded_random_snapshot_rows():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search/series":
            return httpx.Response(
                200,
                json={
                    "current_page": [
                        {
                            "event_ticker": "KXEARNINGSMENTIONAAPL-26JUL30",
                            "event_title": (
                                "What will Apple say during their next earnings call?"
                            ),
                            "series_ticker": "KXEARNINGSMENTIONAAPL",
                            "markets": [
                                {
                                    "ticker": "KXEARNINGSMENTIONAAPL-26JUL30-VISI",
                                    "event_ticker": "KXEARNINGSMENTIONAAPL-26JUL30",
                                    "title": (
                                        "What will Apple say during their next earnings call?"
                                    ),
                                    "custom_strike": {"Word": "Vision Pro"},
                                    "status": "finalized",
                                    "result": "yes",
                                    "close_time": "2026-07-30T20:00:00Z",
                                }
                            ],
                        }
                    ]
                },
            )
        if "/candlesticks" in request.url.path:
            end_ts = int(request.url.params["end_ts"])
            return httpx.Response(
                200,
                json={
                    "candlesticks": [
                        {
                            "end_period_ts": end_ts - 60,
                            "yes_bid": {"close_dollars": "0.42"},
                            "yes_ask": {"close_dollars": "0.48"},
                        }
                    ]
                },
            )
        return httpx.Response(404, json={"error": "unexpected request"})

    collector = HistoricalMentionCollector(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_pages=1,
        snapshot_samples_per_market=3,
        snapshot_min_hours_before_close=2,
        snapshot_max_hours_before_close=48,
        snapshot_sampling_seed=11,
    )

    result = collector.collect()

    assert len(result.rows) == 3
    assert result.stats["rows_written"] == 3
    offsets = {
        int((row.close_time - row.snapshot_target_time).total_seconds() / 3600)
        for row in result.rows
    }
    assert len(offsets) == 3
    assert all(2 <= offset <= 48 for offset in offsets)
    assert {row.market_ticker for row in result.rows} == {
        "KXEARNINGSMENTIONAAPL-26JUL30-VISI"
    }


def test_collector_searches_earnings_prefix_without_status_filter():
    seen_status_params: list[str | None] = []
    seen_query_params: list[str | None] = []
    seen_category_params: list[str | None] = []
    seen_order_by_params: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search/series":
            seen_status_params.append(request.url.params.get("status"))
            seen_query_params.append(request.url.params.get("query"))
            seen_category_params.append(request.url.params.get("category"))
            seen_order_by_params.append(request.url.params.get("order_by"))
            return httpx.Response(200, json={"current_page": []})
        return httpx.Response(404)

    collector = HistoricalMentionCollector(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_pages=1,
    )

    collector.collect()

    assert seen_status_params == [None]
    assert seen_query_params == ["KXEARNINGSMENTION"]
    assert seen_category_params == [None]
    assert seen_order_by_params == [None]


def test_search_series_retries_transient_connection_errors():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("temporary DNS failure", request=request)
        return httpx.Response(200, json={"current_page": []})

    client = KalshiMentionClient(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_sleep=lambda _: None,
    )

    assert list(client.iter_mention_series(query="KXEARNINGSMENTION")) == []
    assert attempts == 2


def test_collector_hydrates_historical_market_detail_when_search_row_lacks_close_time():
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/v1/search/series":
            return httpx.Response(
                200,
                json={
                    "current_page": [
                        {
                            "event_ticker": "KXEARNINGSMENTIONADOBE-25MAR",
                            "event_title": "What will Adobe say on their Q1 FY2025 earnings call?",
                            "series_ticker": "KXEARNINGSMENTIONADOBE",
                            "markets": [
                                {
                                    "ticker": "KXEARNINGSMENTIONADOBE-25MAR-GENAI",
                                    "event_ticker": "KXEARNINGSMENTIONADOBE-25MAR",
                                    "title": (
                                        "What will Adobe say on their Q1 FY2025 "
                                        "earnings call?"
                                    ),
                                    "custom_strike": {"Word": "Generative AI / Gen AI"},
                                    "result": "yes",
                                }
                            ],
                        }
                    ]
                },
            )
        if request.url.path.endswith("/historical/markets/KXEARNINGSMENTIONADOBE-25MAR-GENAI"):
            return httpx.Response(
                200,
                json={
                    "market": {
                        "ticker": "KXEARNINGSMENTIONADOBE-25MAR-GENAI",
                        "event_ticker": "KXEARNINGSMENTIONADOBE-25MAR",
                        "title": "What will Adobe say on their Q1 FY2025 earnings call?",
                        "custom_strike": {"Word": "Generative AI / Gen AI"},
                        "result": "yes",
                        "status": "finalized",
                        "close_time": "2025-03-13T13:59:45Z",
                    }
                },
            )
        if request.url.path.endswith(
            "/historical/markets/KXEARNINGSMENTIONADOBE-25MAR-GENAI/candlesticks"
        ):
            return httpx.Response(
                200,
                json={
                    "candlesticks": [
                        {
                            "end_period_ts": int(
                                datetime(2025, 3, 13, 5, 59, tzinfo=UTC).timestamp()
                            ),
                            "yes_bid": {"close": "0.91"},
                            "yes_ask": {"close": "0.96"},
                        }
                    ]
                },
            )
        return httpx.Response(404, json={"error": "unexpected request"})

    collector = HistoricalMentionCollector(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_pages=1,
    )

    result = collector.collect()

    assert len(result.rows) == 1
    assert result.rows[0].market_ticker == "KXEARNINGSMENTIONADOBE-25MAR-GENAI"
    assert result.rows[0].word_said == "Generative AI / Gen AI"
    assert result.rows[0].close_time == datetime(2025, 3, 13, 13, 59, 45, tzinfo=UTC)
    detail_path = "/historical/markets/KXEARNINGSMENTIONADOBE-25MAR-GENAI"
    assert any(path.endswith(detail_path) for path in requested_paths)


def test_collector_only_emits_earnings_mention_prefix_markets():
    candle_requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search/series":
            return httpx.Response(
                200,
                json={
                    "current_page": [
                        {
                            "event_ticker": "KXTRUMPSAY-26MAY18",
                            "event_title": "What will Trump say this week?",
                            "series_ticker": "KXTRUMPSAY",
                            "markets": [
                                {
                                    "ticker": "KXTRUMPSAY-26MAY18-CHIN",
                                    "event_ticker": "KXTRUMPSAY-26MAY18",
                                    "custom_strike": {"Word": "China"},
                                    "result": "yes",
                                    "close_time": "2026-05-18T14:00:00Z",
                                }
                            ],
                        },
                        {
                            "event_ticker": "KXEARNINGSMENTIONAAPL-26JUL30",
                            "event_title": (
                                "What will Apple say during their next earnings call?"
                            ),
                            "series_ticker": "KXEARNINGSMENTIONAAPL",
                            "markets": [
                                {
                                    "ticker": "KXEARNINGSMENTIONAAPL-26JUL30-VISI",
                                    "event_ticker": "KXEARNINGSMENTIONAAPL-26JUL30",
                                    "title": (
                                        "What will Apple say during their next earnings call?"
                                    ),
                                    "custom_strike": {"Word": "Vision Pro"},
                                    "result": "no",
                                    "close_time": "2026-07-30T20:00:00Z",
                                }
                            ],
                        },
                    ]
                },
            )
        if "/candlesticks" in request.url.path:
            candle_requests.append(request.url.path)
            return httpx.Response(
                200,
                json={
                    "candlesticks": [
                        {
                            "end_period_ts": int(
                                datetime(2026, 7, 30, 11, 59, tzinfo=UTC).timestamp()
                            ),
                            "yes_bid": {"close_dollars": "0.62"},
                            "yes_ask": {"close_dollars": "0.68"},
                        }
                    ]
                },
            )
        return httpx.Response(404, json={"error": "unexpected request"})

    collector = HistoricalMentionCollector(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_pages=1,
    )

    result = collector.collect()

    assert [row.market_ticker for row in result.rows] == [
        "KXEARNINGSMENTIONAAPL-26JUL30-VISI"
    ]
    assert result.stats["skip_reasons"] == {"non_earnings_mention_market": 1}
    assert all("KXTRUMPSAY" not in path for path in candle_requests)
