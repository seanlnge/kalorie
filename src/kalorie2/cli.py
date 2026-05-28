from pathlib import Path
from typing import Annotated

import httpx
import typer

from kalorie2.artifacts import write_collection_artifacts
from kalorie2.collector import DEFAULT_BASE_URL, HistoricalMentionCollector

app = typer.Typer(help="Collect historical Kalshi earnings mention-market data.")


@app.command("collect")
def collect_historical_mentions(
    out_dir: Annotated[Path, typer.Option()] = Path("artifacts"),
    base_url: Annotated[str, typer.Option()] = DEFAULT_BASE_URL,
    status: Annotated[str | None, typer.Option()] = None,
    max_pages: Annotated[int | None, typer.Option()] = None,
    max_markets: Annotated[int | None, typer.Option()] = None,
    snapshot_hours_before_close: Annotated[int, typer.Option(min=1)] = 8,
    snapshot_samples_per_market: Annotated[int, typer.Option(min=1)] = 3,
    snapshot_min_hours_before_close: Annotated[int, typer.Option(min=1)] = 2,
    snapshot_max_hours_before_close: Annotated[int, typer.Option(min=1)] = 48,
    snapshot_sampling_seed: Annotated[int, typer.Option()] = 0,
    snapshot_lookback_hours: Annotated[int, typer.Option(min=1)] = 24,
    max_snapshot_staleness_minutes: Annotated[int | None, typer.Option(min=1)] = None,
    timeout_seconds: Annotated[float, typer.Option(min=1.0)] = 30.0,
) -> None:
    """Pull finalized KXEARNINGSMENTION markets and T-minus-close bid/ask snapshots."""
    with httpx.Client(timeout=timeout_seconds) as http_client:
        result = HistoricalMentionCollector(
            http_client=http_client,
            base_url=base_url,
            status=status,
            max_pages=max_pages,
            max_markets=max_markets,
            snapshot_hours_before_close=snapshot_hours_before_close,
            snapshot_samples_per_market=snapshot_samples_per_market,
            snapshot_min_hours_before_close=snapshot_min_hours_before_close,
            snapshot_max_hours_before_close=snapshot_max_hours_before_close,
            snapshot_sampling_seed=snapshot_sampling_seed,
            snapshot_lookback_hours=snapshot_lookback_hours,
            max_snapshot_staleness_minutes=max_snapshot_staleness_minutes,
        ).collect()
    paths = write_collection_artifacts(result, out_dir=out_dir)
    typer.echo(
        f"Wrote {result.stats['rows_written']} rows, "
        f"{result.stats['skipped_count']} skipped markets"
    )
    typer.echo(f"Rows: {paths['rows']}")
    typer.echo(f"CSV: {paths['csv']}")
    typer.echo(f"Stats: {paths['stats']}")
    typer.echo(f"Skipped: {paths['skipped']}")


if __name__ == "__main__":
    app()
