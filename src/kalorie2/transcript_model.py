import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from kalorie2.call_structure import (
    CallStructureRecord,
    extract_call_structure,
    summarize_prior_call_structure,
)

app = typer.Typer(help="Build transcript-history probability features for earnings mentions.")

_TRANSCRIPT_RE = re.compile(
    r"^(?P<year>\d{4})_Q(?P<quarter>[1-4])_(?P<symbol>.+?)_processed\.txt$",
    re.IGNORECASE,
)
_EVENT_SYMBOL_RE = re.compile(r"^KXEARNINGSMENTION(?P<symbol>[A-Z0-9]+)-")
_EVENT_PHRASE_RE = re.compile(r"what will (?P<company>.+?) say\b", re.IGNORECASE)
_MIN_COUNT_RE = re.compile(r"^(?P<phrase>.+?)\s*\((?P<count>\d+)\+\s*times?\)$", re.IGNORECASE)


@dataclass(frozen=True)
class TranscriptRecord:
    path: Path
    company_name: str
    company_key: str
    symbol: str
    fiscal_year: int
    fiscal_quarter: int
    estimated_available_at: datetime
    published_at: datetime | None
    text: str


def transcript_contains_market_word(market_word: str, text: str) -> bool:
    phrase, min_count = _split_min_count(market_word)
    return any(count_rule_matches(option, text) >= min_count for option in _word_options(phrase))


def count_rule_matches(market_word: str, text: str) -> int:
    tokens = _word_tokens(market_word)
    if not tokens:
        return 0
    haystack = _text_tokens(text)
    if len(tokens) > len(haystack):
        return 0
    matches = 0
    for index in range(0, len(haystack) - len(tokens) + 1):
        if _tokens_match(tokens, haystack[index : index + len(tokens)]):
            matches += 1
    return matches


def fits_per_word_multiplier(
    observations: list[dict],
    *,
    min_word_observations: int = 5,
    shrinkage: float = 10.0,
    max_multiplier: float = 10.0,
) -> dict:
    global_multiplier = _fit_multiplier(observations, max_multiplier=max_multiplier)
    by_word: dict[str, list[dict]] = defaultdict(list)
    for observation in observations:
        by_word[str(observation["word"])].append(observation)

    words = {}
    for word, word_observations in by_word.items():
        raw_multiplier = _fit_multiplier(word_observations, max_multiplier=max_multiplier)
        count = len(word_observations)
        if count < min_word_observations:
            multiplier = global_multiplier
            used_global_fallback = True
        else:
            multiplier = (
                count * raw_multiplier + shrinkage * global_multiplier
            ) / (count + shrinkage)
            used_global_fallback = False
        words[word] = {
            "multiplier": round(multiplier, 6),
            "raw_multiplier": round(raw_multiplier, 6),
            "observations": count,
            "used_global_fallback": used_global_fallback,
        }
    return {
        "global": {
            "multiplier": round(global_multiplier, 6),
            "observations": len(observations),
        },
        "words": words,
    }


def build_transcript_predictions(
    rows: list[dict[str, str]],
    *,
    transcript_root: Path,
    min_word_observations: int = 5,
    shrinkage: float = 10.0,
    max_multiplier: float = 10.0,
) -> list[dict[str, str]]:
    transcript_index = _build_transcript_index(transcript_root)
    ordered_events = _group_rows_by_event(rows)
    prior_observations: list[dict] = []
    output: list[dict[str, str]] = []

    for _, event_rows in ordered_events:
        event_predictions: list[dict[str, str]] = []
        multipliers = fits_per_word_multiplier(
            prior_observations,
            min_word_observations=min_word_observations,
            shrinkage=shrinkage,
            max_multiplier=max_multiplier,
        )
        for row in event_rows:
            predicted = _predict_row(row, transcript_index, multipliers)
            event_predictions.append(predicted)
        output.extend(event_predictions)
        for predicted in event_predictions:
            rate = float(predicted["historical_transcript_rate"])
            if rate <= 0:
                continue
            prior_observations.append(
                {
                    "word": predicted["normalized_word_said"],
                    "historical_rate": rate,
                    "outcome": _parse_outcome(predicted["final_outcome"]),
                }
            )
    return output


def direct_probability_backtest_rows(
    rows: list[dict[str, str]],
    *,
    probability_column: str = "transcript_model_probability",
    margin: float = 0.0,
) -> dict:
    trades = []
    skipped_no_edge = 0
    for row in rows:
        probability = float(row[probability_column])
        yes_bid = float(row["preclose_yes_bid"])
        yes_ask = float(row["preclose_yes_ask"])
        outcome = _parse_outcome(row["final_outcome"])
        if probability > yes_ask + margin:
            side = "YES"
            cost = yes_ask
            pnl = outcome - cost
            edge = probability - yes_ask
        elif probability < yes_bid - margin:
            side = "NO"
            cost = 1 - yes_bid
            pnl = yes_bid - outcome
            edge = yes_bid - probability
        else:
            skipped_no_edge += 1
            continue
        trades.append(
            {
                "event_ticker": row["event_ticker"],
                "market_ticker": row["market_ticker"],
                "close_time": row["close_time"],
                "side": side,
                "model_probability": round(probability, 6),
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "outcome": outcome,
                "cost": round(cost, 6),
                "edge": round(edge, 6),
                "pnl": round(pnl, 6),
            }
        )
    return {
        "strategy": {
            "name": "direct_transcript_probability_bid_ask",
            "probability_column": probability_column,
            "margin": margin,
            "description": (
                "Buy YES when transcript model probability exceeds yes ask plus margin; "
                "buy NO when it is below yes bid minus margin."
            ),
        },
        "summary": _summarize_direct_trades(
            trades,
            total_rows=len(rows),
            skipped_no_edge=skipped_no_edge,
        ),
        "trades": trades,
    }


@app.command("predict")
def predict_command(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    transcript_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False),
    ] = Path("data/earnings_call_transcripts"),
    out_csv: Annotated[Path | None, typer.Option()] = None,
    report_json: Annotated[Path | None, typer.Option()] = None,
    min_word_observations: Annotated[int, typer.Option(min=1)] = 5,
    shrinkage: Annotated[float, typer.Option(min=0.0)] = 10.0,
    max_multiplier: Annotated[float, typer.Option(min=0.0)] = 10.0,
) -> None:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    predicted = build_transcript_predictions(
        rows,
        transcript_root=transcript_root,
        min_word_observations=min_word_observations,
        shrinkage=shrinkage,
        max_multiplier=max_multiplier,
    )
    report = _prediction_report(predicted)
    if out_csv is not None:
        _write_predictions_csv(out_csv, predicted)
    if report_json is not None:
        _write_json(report_json, report)
    typer.echo(
        f"Rows: {report['rows']} | "
        f"With transcript history: {report['rows_with_transcript_history']} | "
        f"Matched company: {report['rows_with_company_match']}"
    )


def _predict_row(
    row: dict[str, str],
    transcript_index: dict,
    multipliers: dict,
) -> dict[str, str]:
    event_time = _parse_datetime(row["close_time"])
    symbol = _resolve_company_symbol(row, transcript_index)
    prior_transcripts = [
        record
        for record in transcript_index["by_symbol"].get(symbol, [])
        if record.estimated_available_at < event_time
    ]
    word = (row.get("normalized_word_said") or row.get("word_said") or "").strip().lower()
    hits = sum(
        1
        for record in prior_transcripts
        if transcript_contains_market_word(row["word_said"], record.text)
    )
    phrase_mentions = [
        _market_word_match_count(row["word_said"], record.text)
        for record in prior_transcripts
    ]
    transcript_word_counts = [len(_text_tokens(record.text)) for record in prior_transcripts]
    call_structure_features = summarize_prior_call_structure(
        [
            _call_structure_record(record)
            for record in transcript_index["by_symbol"].get(symbol, [])
            if record.published_at is not None
        ],
        cutoff_time=event_time,
    )
    prior_count = len(prior_transcripts)
    historical_rate = hits / prior_count if prior_count else 0.0
    word_multiplier = multipliers["words"].get(word)
    if word_multiplier is None:
        multiplier = float(multipliers["global"]["multiplier"])
        multiplier_observations = 0
        used_global_fallback = True
    else:
        multiplier = float(word_multiplier["multiplier"])
        multiplier_observations = int(word_multiplier["observations"])
        used_global_fallback = bool(word_multiplier["used_global_fallback"])
    probability = min(1.0, max(0.0, multiplier * historical_rate))
    enriched = dict(row)
    enriched.update(
        {
            "transcript_company_symbol": symbol,
            "transcript_prior_count": str(prior_count),
            "transcript_hit_count": str(hits),
            "company_transcript_coverage_count": str(prior_count),
            "company_transcript_style_available": "1" if prior_count else "0",
            "company_avg_transcript_word_count_prior": f"{_mean(transcript_word_counts):.6f}",
            "company_avg_phrase_mentions_prior": f"{_mean(phrase_mentions):.6f}",
            **{
                key: _format_call_structure_feature(key, value)
                for key, value in call_structure_features.items()
            },
            "historical_transcript_rate": f"{historical_rate:.6f}",
            "word_multiplier": f"{multiplier:.6f}",
            "word_multiplier_observations": str(multiplier_observations),
            "word_multiplier_used_global_fallback": str(used_global_fallback).lower(),
            "transcript_model_probability": f"{probability:.6f}",
        }
    )
    return enriched


def _market_word_match_count(market_word: str, text: str) -> int:
    phrase, _ = _split_min_count(market_word)
    return max((count_rule_matches(option, text) for option in _word_options(phrase)), default=0)


def _call_structure_record(record: TranscriptRecord) -> CallStructureRecord:
    structure = extract_call_structure(record.text)
    return CallStructureRecord(
        available_at=record.published_at,
        call_duration_minutes=structure.call_duration_minutes,
        qa_question_count=structure.qa_question_count,
        prepared_remarks_minutes=structure.prepared_remarks_minutes,
    )


def _format_call_structure_feature(key: str, value: float) -> str:
    if key == "company_prior_call_count":
        return str(int(value))
    return f"{value:.6f}"


def _build_transcript_index(root: Path) -> dict:
    records = scan_transcripts(root)
    by_symbol: dict[str, list[TranscriptRecord]] = defaultdict(list)
    symbol_by_company_key: dict[str, str] = {}
    for record in records:
        by_symbol[record.symbol].append(record)
        symbol_by_company_key.setdefault(record.company_key, record.symbol)
    for symbol_records in by_symbol.values():
        symbol_records.sort(key=lambda record: record.estimated_available_at)
    return {
        "records": records,
        "by_symbol": dict(by_symbol),
        "symbol_by_company_key": symbol_by_company_key,
    }


def scan_transcripts(root: Path) -> list[TranscriptRecord]:
    records: list[TranscriptRecord] = []
    for path in root.rglob("*_processed.txt"):
        match = _TRANSCRIPT_RE.match(path.name)
        if not match:
            continue
        fiscal_year = int(match.group("year"))
        fiscal_quarter = int(match.group("quarter"))
        text = path.read_text(encoding="utf-8", errors="ignore")
        records.append(
            TranscriptRecord(
                path=path,
                company_name=path.parent.name,
                company_key=_company_key(path.parent.name),
                symbol=match.group("symbol").lower(),
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                estimated_available_at=_quarter_available_at(fiscal_year, fiscal_quarter),
                published_at=_published_at_from_text(text),
                text=text,
            )
        )
    return records


def _resolve_company_symbol(row: dict[str, str], transcript_index: dict) -> str:
    prefix_symbol = _symbol_from_event(row.get("event_ticker", "")).lower()
    if prefix_symbol in transcript_index["by_symbol"]:
        return prefix_symbol
    company = _company_from_event_phrase(row.get("event_phrase", ""))
    if company:
        mapped = transcript_index["symbol_by_company_key"].get(_company_key(company))
        if mapped:
            return mapped
    return prefix_symbol


def _symbol_from_event(event_ticker: str) -> str:
    match = _EVENT_SYMBOL_RE.match(event_ticker)
    return match.group("symbol") if match else ""


def _company_from_event_phrase(event_phrase: str) -> str:
    match = _EVENT_PHRASE_RE.search(event_phrase)
    return match.group("company").strip() if match else ""


def _company_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _quarter_available_at(fiscal_year: int, fiscal_quarter: int) -> datetime:
    month_day = {
        1: (3, 31),
        2: (6, 30),
        3: (9, 30),
        4: (12, 31),
    }[fiscal_quarter]
    return datetime(fiscal_year, month_day[0], month_day[1], 23, 59, 59, tzinfo=UTC)


def _published_at_from_text(text: str) -> datetime | None:
    match = re.search(
        r"\b(?:published|available)\s+at\s*:\s*([^\n\r]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        return _parse_datetime(match.group(1).strip())
    except ValueError:
        return None


def _split_min_count(market_word: str) -> tuple[str, int]:
    match = _MIN_COUNT_RE.match(market_word.strip())
    if not match:
        return market_word, 1
    return match.group("phrase"), int(match.group("count"))


def _word_options(market_word: str) -> list[str]:
    options = [part.strip() for part in re.split(r"\s*/\s*", market_word) if part.strip()]
    return options or [market_word]


def _word_tokens(market_word: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", market_word.lower())


def _text_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['’]s)?", text.lower())


def _tokens_match(expected: list[str], actual: list[str]) -> bool:
    if len(expected) != len(actual):
        return False
    return all(
        _token_matches(expected_token, actual_token)
        for expected_token, actual_token in zip(expected, actual, strict=True)
    )


def _token_matches(expected: str, actual: str) -> bool:
    normalized_actual = actual.removesuffix("'s").removesuffix("’s")
    if normalized_actual == expected:
        return True
    if expected.endswith("s"):
        return False
    return normalized_actual == f"{expected}s"


def _fit_multiplier(observations: list[dict], *, max_multiplier: float) -> float:
    usable = [obs for obs in observations if float(obs["historical_rate"]) > 0]
    if not usable:
        return 1.0
    numerator = sum(float(obs["historical_rate"]) * int(obs["outcome"]) for obs in usable)
    denominator = sum(float(obs["historical_rate"]) ** 2 for obs in usable)
    if denominator == 0:
        return 1.0
    return min(max_multiplier, max(0.0, numerator / denominator))


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


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


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_outcome(value: str) -> int:
    normalized = value.strip().lower()
    if normalized == "yes":
        return 1
    if normalized == "no":
        return 0
    raise ValueError(f"Unsupported final_outcome: {value}")


def _prediction_report(rows: list[dict[str, str]]) -> dict:
    return {
        "rows": len(rows),
        "rows_with_company_match": sum(1 for row in rows if row["transcript_company_symbol"]),
        "rows_with_transcript_history": sum(
            1 for row in rows if int(row["transcript_prior_count"]) > 0
        ),
        "rows_with_transcript_hits": sum(1 for row in rows if int(row["transcript_hit_count"]) > 0),
        "mean_probability": round(
            sum(float(row["transcript_model_probability"]) for row in rows) / len(rows),
            6,
        )
        if rows
        else 0.0,
    }


def _summarize_direct_trades(
    trades: list[dict],
    *,
    total_rows: int,
    skipped_no_edge: int,
) -> dict:
    total_cost = sum(trade["cost"] for trade in trades)
    total_pnl = sum(trade["pnl"] for trade in trades)
    wins = sum(1 for trade in trades if trade["pnl"] > 0)
    return {
        "total_rows": total_rows,
        "trades": len(trades),
        "yes_trades": sum(1 for trade in trades if trade["side"] == "YES"),
        "no_trades": sum(1 for trade in trades if trade["side"] == "NO"),
        "skipped_no_edge": skipped_no_edge,
        "total_cost": round(total_cost, 6),
        "total_pnl": round(total_pnl, 6),
        "avg_pnl_per_trade": round(total_pnl / len(trades), 6) if trades else 0.0,
        "roi_on_cost": round(total_pnl / total_cost, 6) if total_cost else 0.0,
        "win_rate": round(wins / len(trades), 6) if trades else 0.0,
    }


def _write_predictions_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_direct_backtest_artifacts(
    report: dict,
    *,
    json_out: Path,
    trades_out: Path,
) -> None:
    _write_json(json_out, report)
    trades_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(report["trades"][0].keys()) if report["trades"] else []
    with trades_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["trades"])


if __name__ == "__main__":
    app()
