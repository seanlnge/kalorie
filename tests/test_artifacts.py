from datetime import UTC, datetime
from decimal import Decimal

from kalorie2.artifacts import write_collection_artifacts
from kalorie2.models import CollectionResult, HistoricalMentionMarketRow


def test_write_collection_artifacts_includes_csv_table_with_category_column(tmp_path):
    result = CollectionResult(
        rows=[
            HistoricalMentionMarketRow(
                market_ticker="KXEARNINGSMENTIONAAPL-26JUL30-VISI",
                event_ticker="KXEARNINGSMENTIONAAPL-26JUL30",
                series_ticker="KXEARNINGSMENTIONAAPL",
                market_category="earnings",
                event_phrase="What will Apple say during their next earnings call?",
                market_name="What will Apple say during their next earnings call?",
                word_said="Vision Pro",
                normalized_word_said="vision pro",
                final_outcome="yes",
                close_time=datetime(2026, 7, 30, 20, tzinfo=UTC),
                snapshot_target_time=datetime(2026, 7, 30, 12, tzinfo=UTC),
                preclose_yes_bid=Decimal("0.62"),
                preclose_yes_ask=Decimal("0.68"),
                preclose_yes_mid=Decimal("0.65"),
                candle_end_ts=1785412740,
                snapshot_staleness_seconds=60,
                source="test",
            )
        ],
        skipped_markets=[],
        stats={"rows_written": 1, "skipped_count": 0, "events_seen": 1, "markets_seen": 1},
    )

    paths = write_collection_artifacts(
        result,
        out_dir=tmp_path,
        run_date=datetime(2026, 5, 22, tzinfo=UTC),
    )

    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "market_category" in csv_text.splitlines()[0]
    assert "earnings" in csv_text
    assert "Vision Pro" in csv_text
