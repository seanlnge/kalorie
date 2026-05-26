import csv
import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Score Kalshi mention-market probabilities.")

BIN_LABELS = ["0%", "1-2%"] + [f"{start}-{start + 4}%" for start in range(3, 98, 5)] + [
    "98-99%",
    "100%",
]


def assign_probability_bin(probability: float) -> str:
    percent = round(probability * 100)
    if percent <= 0:
        return "0%"
    if percent <= 2:
        return "1-2%"
    if percent >= 100:
        return "100%"
    if percent >= 98:
        return "98-99%"
    start = 3 + ((percent - 3) // 5) * 5
    return f"{start}-{start + 4}%"


def score_rows(
    rows: list[dict[str, str]],
    *,
    probability_column: str = "preclose_yes_mid",
) -> dict:
    scored_rows = []
    for row in rows:
        probability = float(row[probability_column])
        outcome = _parse_outcome(row["final_outcome"])
        squared_error = (probability - outcome) ** 2
        scored_rows.append(
            {
                "probability": probability,
                "outcome": outcome,
                "squared_error": squared_error,
                "bin": assign_probability_bin(probability),
            }
        )

    return {
        "probability_column": probability_column,
        "overall": _summarize(scored_rows),
        "bins": [
            {"bin": label, **_summarize([row for row in scored_rows if row["bin"] == label])}
            for label in BIN_LABELS
        ],
    }


@app.command("score")
def score_csv_command(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    probability_column: Annotated[str, typer.Option()] = "preclose_yes_mid",
    json_out: Annotated[Path | None, typer.Option()] = None,
    csv_out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    report = score_csv(input_csv, probability_column=probability_column)
    if json_out is not None:
        _write_json(json_out, report)
    if csv_out is not None:
        _write_bins_csv(csv_out, report["bins"])
    typer.echo(
        f"Rows: {report['overall']['count']} | "
        f"Brier: {report['overall']['brier_score']:.6f} | "
        f"Probability column: {probability_column}"
    )


def score_csv(path: Path, *, probability_column: str = "preclose_yes_mid") -> dict:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return score_rows(rows, probability_column=probability_column)


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {
            "count": 0,
            "brier_score": None,
            "mean_probability": None,
            "outcome_rate": None,
        }
    count = len(rows)
    return {
        "count": count,
        "brier_score": round(sum(row["squared_error"] for row in rows) / count, 6),
        "mean_probability": round(sum(row["probability"] for row in rows) / count, 6),
        "outcome_rate": round(sum(row["outcome"] for row in rows) / count, 6),
    }


def _parse_outcome(value: str) -> int:
    normalized = value.strip().lower()
    if normalized == "yes":
        return 1
    if normalized == "no":
        return 0
    raise ValueError(f"Unsupported final_outcome: {value}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_bins_csv(path: Path, bins: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["bin", "count", "brier_score", "mean_probability", "outcome_rate"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bins)


if __name__ == "__main__":
    app()
