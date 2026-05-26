import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from kalorie2.scoring import assign_probability_bin

app = typer.Typer(help="Backtest calibrated Kalshi earnings mention-market strategies.")


def backtest_rows(
    rows: list[dict[str, str]],
    *,
    min_bin_count: int = 20,
    margin: float = 0.0,
    probability_column: str = "preclose_yes_mid",
) -> dict:
    ordered_events = _group_rows_by_event(rows)
    bin_outcomes: dict[str, list[int]] = defaultdict(list)
    trades: list[dict] = []
    eligible_rows = 0
    skipped_not_enough_history = 0
    skipped_no_edge = 0

    for _, event_rows in ordered_events:
        event_trades: list[dict] = []
        for row in event_rows:
            probability = float(row[probability_column])
            probability_bin = assign_probability_bin(probability)
            history = bin_outcomes[probability_bin]
            if len(history) < min_bin_count:
                skipped_not_enough_history += 1
                continue
            eligible_rows += 1
            calibrated_probability = sum(history) / len(history)
            trade = _maybe_trade(
                row=row,
                probability_bin=probability_bin,
                calibrated_probability=calibrated_probability,
                margin=margin,
            )
            if trade is None:
                skipped_no_edge += 1
                continue
            event_trades.append(trade)

        trades.extend(event_trades)
        for row in event_rows:
            bin_outcomes[assign_probability_bin(float(row[probability_column]))].append(
                _parse_outcome(row["final_outcome"])
            )

    return {
        "strategy": {
            "name": "walk_forward_bin_calibration_bid_ask",
            "probability_column": probability_column,
            "min_bin_count": min_bin_count,
            "margin": margin,
            "description": (
                "For each event, estimate each probability bin's YES rate from prior events "
                "only. Buy YES when calibrated_p exceeds yes ask plus margin; buy NO when "
                "calibrated_p is below yes bid minus margin."
            ),
        },
        "summary": _summarize_trades(
            trades,
            total_rows=len(rows),
            eligible_rows=eligible_rows,
            skipped_not_enough_history=skipped_not_enough_history,
            skipped_no_edge=skipped_no_edge,
        ),
        "by_bin": _summarize_by_key(trades, "bin"),
        "by_side": _summarize_by_key(trades, "side"),
        "trades": trades,
    }


@app.command("backtest")
def backtest_csv_command(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    json_out: Annotated[Path | None, typer.Option()] = None,
    trades_out: Annotated[Path | None, typer.Option()] = None,
    probability_column: Annotated[str, typer.Option()] = "preclose_yes_mid",
    min_bin_count: Annotated[int, typer.Option(min=1)] = 20,
    margin: Annotated[float, typer.Option(min=0.0)] = 0.0,
) -> None:
    report = backtest_csv(
        input_csv,
        probability_column=probability_column,
        min_bin_count=min_bin_count,
        margin=margin,
    )
    if json_out is not None:
        _write_json(json_out, report)
    if trades_out is not None:
        _write_trades_csv(trades_out, report["trades"])
    typer.echo(
        f"Rows: {report['summary']['total_rows']} | "
        f"Trades: {report['summary']['trades']} | "
        f"PnL: {report['summary']['total_pnl']:.6f} | "
        f"ROI on cost: {report['summary']['roi_on_cost']:.6f}"
    )


def backtest_csv(
    path: Path,
    *,
    probability_column: str = "preclose_yes_mid",
    min_bin_count: int = 20,
    margin: float = 0.0,
) -> dict:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return backtest_rows(
        rows,
        probability_column=probability_column,
        min_bin_count=min_bin_count,
        margin=margin,
    )


def _maybe_trade(
    *,
    row: dict[str, str],
    probability_bin: str,
    calibrated_probability: float,
    margin: float,
) -> dict | None:
    yes_bid = float(row["preclose_yes_bid"])
    yes_ask = float(row["preclose_yes_ask"])
    outcome = _parse_outcome(row["final_outcome"])
    if calibrated_probability > yes_ask + margin:
        cost = yes_ask
        pnl = outcome - cost
        side = "YES"
        edge = calibrated_probability - yes_ask
    elif calibrated_probability < yes_bid - margin:
        cost = 1 - yes_bid
        pnl = yes_bid - outcome
        side = "NO"
        edge = yes_bid - calibrated_probability
    else:
        return None
    return {
        "event_ticker": row["event_ticker"],
        "market_ticker": row["market_ticker"],
        "close_time": row["close_time"],
        "bin": probability_bin,
        "side": side,
        "calibrated_probability": round(calibrated_probability, 6),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "outcome": outcome,
        "cost": round(cost, 6),
        "edge": round(edge, 6),
        "pnl": round(pnl, 6),
    }


def _group_rows_by_event(rows: list[dict[str, str]]) -> list[tuple[str, list[dict[str, str]]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["event_ticker"]].append(row)
    return sorted(
        grouped.items(),
        key=lambda item: (
            min(_parse_datetime(row["close_time"]) for row in item[1]),
            item[0],
        ),
    )


def _summarize_trades(
    trades: list[dict],
    *,
    total_rows: int,
    eligible_rows: int,
    skipped_not_enough_history: int,
    skipped_no_edge: int,
) -> dict:
    total_cost = sum(trade["cost"] for trade in trades)
    total_pnl = sum(trade["pnl"] for trade in trades)
    wins = sum(1 for trade in trades if trade["pnl"] > 0)
    return {
        "total_rows": total_rows,
        "eligible_rows": eligible_rows,
        "trades": len(trades),
        "yes_trades": sum(1 for trade in trades if trade["side"] == "YES"),
        "no_trades": sum(1 for trade in trades if trade["side"] == "NO"),
        "skipped_not_enough_history": skipped_not_enough_history,
        "skipped_no_edge": skipped_no_edge,
        "total_cost": round(total_cost, 6),
        "total_pnl": round(total_pnl, 6),
        "avg_pnl_per_trade": round(total_pnl / len(trades), 6) if trades else 0.0,
        "roi_on_cost": round(total_pnl / total_cost, 6) if total_cost else 0.0,
        "win_rate": round(wins / len(trades), 6) if trades else 0.0,
    }


def _summarize_by_key(trades: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for trade in trades:
        groups[str(trade[key])].append(trade)
    return [
        {
            key: value,
            **_summarize_trades(
                group,
                total_rows=len(group),
                eligible_rows=len(group),
                skipped_not_enough_history=0,
                skipped_no_edge=0,
            ),
        }
        for value, group in sorted(groups.items())
    ]


def _parse_outcome(value: str) -> int:
    normalized = value.strip().lower()
    if normalized == "yes":
        return 1
    if normalized == "no":
        return 0
    raise ValueError(f"Unsupported final_outcome: {value}")


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_trades_csv(path: Path, trades: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_ticker",
        "market_ticker",
        "close_time",
        "bin",
        "side",
        "calibrated_probability",
        "yes_bid",
        "yes_ask",
        "outcome",
        "cost",
        "edge",
        "pnl",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)


if __name__ == "__main__":
    app()
