import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from kalorie2.scoring import (
    app,
    assign_probability_bin,
    score_rows,
)


def test_assign_probability_bin_matches_requested_ranges():
    assert assign_probability_bin(0.0) == "0%"
    assert assign_probability_bin(0.02) == "1-2%"
    assert assign_probability_bin(0.03) == "3-7%"
    assert assign_probability_bin(0.07) == "3-7%"
    assert assign_probability_bin(0.08) == "8-12%"
    assert assign_probability_bin(0.17) == "13-17%"
    assert assign_probability_bin(0.98) == "98-99%"
    assert assign_probability_bin(1.0) == "100%"


def test_score_rows_reports_overall_and_bin_brier_scores():
    rows = [
        {"preclose_yes_mid": "0.00", "final_outcome": "no"},
        {"preclose_yes_mid": "0.05", "final_outcome": "yes"},
        {"preclose_yes_mid": "0.10", "final_outcome": "no"},
        {"preclose_yes_mid": "0.15", "final_outcome": "yes"},
    ]

    report = score_rows(rows)

    assert report["probability_column"] == "preclose_yes_mid"
    assert report["overall"]["count"] == 4
    assert report["overall"]["brier_score"] == 0.40875
    assert report["bins"][0]["bin"] == "0%"
    assert report["bins"][0]["count"] == 1
    assert report["bins"][0]["brier_score"] == 0.0
    bins_by_label = {row["bin"]: row for row in report["bins"]}
    assert bins_by_label["3-7%"]["brier_score"] == 0.9025
    assert bins_by_label["8-12%"]["brier_score"] == 0.01
    assert report["bins"][-1]["bin"] == "100%"


def test_score_rows_includes_empty_probability_bins():
    report = score_rows([{"preclose_yes_mid": "0.05", "final_outcome": "yes"}])

    zero_bin = report["bins"][0]
    assert zero_bin["bin"] == "0%"
    assert zero_bin["count"] == 0
    assert zero_bin["brier_score"] is None


def test_scoring_cli_writes_json_and_csv_reports(tmp_path: Path):
    input_csv = tmp_path / "markets.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["preclose_yes_mid", "final_outcome"])
        writer.writeheader()
        writer.writerows(
            [
                {"preclose_yes_mid": "0.05", "final_outcome": "yes"},
                {"preclose_yes_mid": "0.10", "final_outcome": "no"},
            ]
        )
    json_out = tmp_path / "brier.json"
    csv_out = tmp_path / "brier.csv"

    result = CliRunner().invoke(
        app,
        [
            str(input_csv),
            "--json-out",
            str(json_out),
            "--csv-out",
            str(csv_out),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(json_out.read_text(encoding="utf-8"))["overall"]["count"] == 2
    csv_lines = csv_out.read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].startswith("bin,count,brier_score")
    assert any(line.startswith("3-7%,1,0.9025") for line in csv_lines)
