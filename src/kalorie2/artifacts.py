import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from kalorie2.models import CollectionResult, HistoricalMentionMarketRow

CSV_COLUMNS = [
    "market_ticker",
    "event_ticker",
    "series_ticker",
    "market_category",
    "event_phrase",
    "market_name",
    "word_said",
    "normalized_word_said",
    "final_outcome",
    "status",
    "close_time",
    "snapshot_target_time",
    "preclose_yes_bid",
    "preclose_yes_ask",
    "preclose_yes_mid",
    "candle_end_ts",
    "snapshot_staleness_seconds",
    "preclose_volume",
    "preclose_open_interest",
    "preclose_yes_bid_size",
    "preclose_yes_ask_size",
    "company_prior_call_count",
    "company_avg_call_duration_minutes_prior",
    "company_avg_qa_question_count_prior",
    "company_avg_prepared_remarks_minutes_prior",
    "company_qa_share_prior",
    "company_question_count_trend_prior",
    "company_transcript_coverage_count",
    "company_transcript_style_available",
    "company_avg_transcript_word_count_prior",
    "company_avg_phrase_mentions_prior",
    "settlement_ts",
    "source",
]


def write_collection_artifacts(
    result: CollectionResult,
    *,
    out_dir: Path,
    run_date: datetime | None = None,
) -> dict[str, Path]:
    run_date = run_date or datetime.now(tz=UTC)
    stamp = run_date.strftime("%Y%m%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rows": out_dir / f"mention-markets-historical-{stamp}.json",
        "csv": out_dir / f"mention-markets-historical-{stamp}.csv",
        "stats": out_dir / f"mention-markets-historical-{stamp}-stats.json",
        "skipped": out_dir / f"mention-markets-historical-{stamp}-skipped.json",
    }
    _write_json(paths["rows"], [row.model_dump(mode="json") for row in result.rows])
    _write_csv(paths["csv"], result.rows)
    _write_json(paths["stats"], result.stats)
    _write_json(
        paths["skipped"],
        [skip.model_dump(mode="json") for skip in result.skipped_markets],
    )
    return paths


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[HistoricalMentionMarketRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            dumped = row.model_dump(mode="json")
            writer.writerow({column: dumped.get(column, "") for column in CSV_COLUMNS})
