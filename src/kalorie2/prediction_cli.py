import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from kalorie2.event_scenarios import build_event_scenario_prompt
from kalorie2.mixmcp import (
    MixMcpEventPacket,
    apply_mixmcp_to_predictions,
    build_mixmcp_prompt,
    build_openai_mixmcp_payload,
    parse_mixmcp_response,
)
from kalorie2.prediction_features import build_feature_matrix
from kalorie2.prediction_types import (
    ArtifactRetentionPolicy,
    PredictionInputRow,
    PredictionRunConfig,
)
from kalorie2.residual_engine import ResidualPrediction, walk_forward_predictions
from kalorie2.web_evidence import (
    WebEvidencePacket,
    build_openai_web_search_payload,
    build_web_evidence_prompt,
    parse_web_evidence_response,
)
from kalorie2.web_evidence_audit import audit_web_evidence_dir, write_audit_reports

app = typer.Typer(help="Build and evaluate Kalshi earnings mention prediction overlays.")

_CANONICAL_SOURCE_FILES = {
    "mention-markets-historical-20260523.csv",
    "mention-markets-historical-20260523.json",
}

_FEATURE_ABLATION_PREFIXES = {
    "none": (),
    "market": ("market_", "snapshot_"),
    "phrase": ("phrase_",),
    "semantic": ("phrase_semantic_",),
    "resolution": ("resolution_",),
    "scenario": ("scenario_",),
    "web": ("web_evidence_",),
}


@dataclass(frozen=True)
class WebEvidenceFetchResult:
    packet: WebEvidencePacket
    usage_metadata: dict[str, Any]


@app.command("build-prompts")
def build_prompts_command(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    run_id: Annotated[str, typer.Option()],
    out_dir: Annotated[Path, typer.Option()],
) -> None:
    rows = _read_rows(input_csv)
    config = _run_config(run_id)
    prompts_dir = config.validate_output_path(out_dir / "prompts", artifact_kind="temporary_prompt")
    prompts_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for event_ticker, event_rows in _group_rows_by_event(rows):
        first = event_rows[0]
        prompt = build_event_scenario_prompt(
            event={
                "event_ticker": event_ticker,
                "company_name": _company_name_from_event_phrase(first.event_phrase),
                "snapshot_target_time": first.snapshot_target_time.isoformat(),
            },
            target_phrases=sorted({row.normalized_word_said for row in event_rows}),
            evidence_snippets=[],
        )
        prompt_path = prompts_dir / f"{event_ticker}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        manifest.append({"event_ticker": event_ticker, "prompt_path": str(prompt_path)})

    _write_json_checked(
        out_dir / "prompts_manifest.json",
        {"run_id": run_id, "prompts": manifest},
        config=config,
        artifact_kind="temporary_prompt_manifest",
    )
    typer.echo(f"Wrote {len(manifest)} prompts")


@app.command("collect-web-evidence")
def collect_web_evidence_command(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    run_id: Annotated[str, typer.Option()],
    out_dir: Annotated[Path, typer.Option()],
    model: Annotated[str, typer.Option()] = "gpt-5.5",
    max_events: Annotated[int | None, typer.Option(min=1)] = None,
    dry_run: Annotated[bool, typer.Option()] = True,
    request_timeout_seconds: Annotated[float, typer.Option(min=1.0)] = 600.0,
    skip_existing: Annotated[bool, typer.Option()] = False,
    parallel_requests: Annotated[int, typer.Option(min=1)] = 1,
    max_paid_calls: Annotated[int | None, typer.Option(min=0)] = None,
    max_estimated_cost_dollars: Annotated[float | None, typer.Option(min=0.0)] = None,
    estimated_cost_per_call_dollars: Annotated[float, typer.Option(min=0.0)] = 1.0,
) -> None:
    rows = _read_rows(input_csv)
    config = _run_config(run_id)
    requests_dir = config.validate_output_path(
        out_dir / "web-evidence" / "requests",
        artifact_kind="web_evidence_request",
    )
    packets_dir = config.validate_output_path(
        out_dir / "web-evidence" / "packets",
        artifact_kind="web_evidence_packet",
    )
    usage_dir = config.validate_output_path(
        out_dir / "web-evidence" / "usage",
        artifact_kind="web_evidence_usage",
    )
    requests_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        packets_dir.mkdir(parents=True, exist_ok=True)
        usage_dir.mkdir(parents=True, exist_ok=True)

    groups = _group_rows_by_event(rows)
    if max_events is not None:
        groups = groups[:max_events]

    written_packets = 0
    skipped_packets = 0
    fetch_jobs = []
    for index, (event_ticker, event_rows) in enumerate(groups, start=1):
        packet_path = packets_dir / f"{event_ticker}.json"
        typer.echo(f"[{index}/{len(groups)}] Collecting web evidence for {event_ticker}")
        first = event_rows[0]
        prompt = build_web_evidence_prompt(
            event={
                "event_ticker": event_ticker,
                "company_name": _company_name_from_event_phrase(first.event_phrase),
                "cutoff_time": first.snapshot_target_time.isoformat(),
            },
            target_phrases=sorted({row.normalized_word_said for row in event_rows}),
        )
        payload = build_openai_web_search_payload(prompt=prompt, model=model)
        _write_json_checked(
            requests_dir / f"{event_ticker}.json",
            payload,
            config=config,
            artifact_kind="web_evidence_request",
        )
        if dry_run:
            continue
        if skip_existing and packet_path.exists():
            parse_web_evidence_response(packet_path.read_text(encoding="utf-8"))
            skipped_packets += 1
            typer.echo(f"[{index}/{len(groups)}] Skipped existing packet for {event_ticker}")
            continue
        fetch_jobs.append(
            (index, event_ticker, packet_path, usage_dir / f"{event_ticker}.json", payload)
        )

    if fetch_jobs:
        _enforce_paid_run_caps(
            paid_call_count=len(fetch_jobs),
            max_paid_calls=max_paid_calls,
            max_estimated_cost_dollars=max_estimated_cost_dollars,
            estimated_cost_per_call_dollars=estimated_cost_per_call_dollars,
        )
        typer.echo(f"Parallel requests: {parallel_requests}")
        typer.echo(f"Pending paid calls: {len(fetch_jobs)}")
    with ThreadPoolExecutor(max_workers=parallel_requests) as executor:
        future_to_job = {
            executor.submit(
                _fetch_and_write_web_evidence_packet,
                payload,
                event_ticker=event_ticker,
                packet_path=packet_path,
                usage_path=usage_path,
                config=config,
                timeout_seconds=request_timeout_seconds,
            ): (index, event_ticker)
            for index, event_ticker, packet_path, usage_path, payload in fetch_jobs
        }
        for future in as_completed(future_to_job):
            index, event_ticker = future_to_job[future]
            future.result()
            written_packets += 1
            typer.echo(
                f"[{index}/{len(groups)}] Wrote web evidence packet for {event_ticker}"
            )

    typer.echo(f"Wrote {len(groups)} web evidence requests")
    if not dry_run:
        typer.echo(f"Wrote {written_packets} web evidence packets")
        typer.echo(f"Skipped {skipped_packets} existing web evidence packets")


@app.command("collect-mixmcp")
def collect_mixmcp_command(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    run_id: Annotated[str, typer.Option()],
    out_dir: Annotated[Path, typer.Option()],
    model: Annotated[str, typer.Option()] = "gpt-5.4-mini",
    max_events: Annotated[int | None, typer.Option(min=1)] = None,
    dry_run: Annotated[bool, typer.Option()] = True,
    request_timeout_seconds: Annotated[float, typer.Option(min=1.0)] = 600.0,
    skip_existing: Annotated[bool, typer.Option()] = False,
    max_paid_calls: Annotated[int | None, typer.Option(min=0)] = None,
    max_estimated_cost_dollars: Annotated[float | None, typer.Option(min=0.0)] = None,
    estimated_cost_per_call_dollars: Annotated[float, typer.Option(min=0.0)] = 0.05,
) -> None:
    rows = _read_rows(input_csv)
    config = _run_config(run_id)
    requests_dir = config.validate_output_path(
        out_dir / "mixmcp" / "requests",
        artifact_kind="mixmcp_request",
    )
    packets_dir = config.validate_output_path(
        out_dir / "mixmcp" / "packets",
        artifact_kind="mixmcp_packet",
    )
    requests_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        packets_dir.mkdir(parents=True, exist_ok=True)

    groups = _group_rows_by_event(rows)
    if max_events is not None:
        groups = groups[:max_events]

    fetch_jobs = []
    written_packets = 0
    skipped_packets = 0
    for index, (event_ticker, event_rows) in enumerate(groups, start=1):
        packet_path = packets_dir / f"{event_ticker}.json"
        first = event_rows[0]
        prompt = build_mixmcp_prompt(
            event={
                "event_ticker": event_ticker,
                "cutoff_time": first.snapshot_target_time.isoformat(),
            },
            targets=[
                {
                    "market_ticker": row.market_ticker,
                    "target_phrase": row.normalized_word_said or row.word_said,
                    "market_probability": float(row.preclose_yes_mid),
                }
                for row in event_rows
            ],
        )
        payload = build_openai_mixmcp_payload(prompt=prompt, model=model)
        _write_json_checked(
            requests_dir / f"{event_ticker}.json",
            payload,
            config=config,
            artifact_kind="mixmcp_request",
        )
        if dry_run:
            continue
        if skip_existing and packet_path.exists():
            parse_mixmcp_response(packet_path.read_text(encoding="utf-8"))
            skipped_packets += 1
            continue
        fetch_jobs.append((index, event_ticker, packet_path, payload))

    if fetch_jobs:
        _enforce_paid_run_caps(
            paid_call_count=len(fetch_jobs),
            max_paid_calls=max_paid_calls,
            max_estimated_cost_dollars=max_estimated_cost_dollars,
            estimated_cost_per_call_dollars=estimated_cost_per_call_dollars,
        )
    for index, event_ticker, packet_path, payload in fetch_jobs:
        packet = _fetch_mixmcp_packet(payload, timeout_seconds=request_timeout_seconds)
        _write_json_checked(
            packet_path,
            packet.model_dump(mode="json"),
            config=config,
            artifact_kind="mixmcp_packet",
        )
        written_packets += 1
        typer.echo(f"[{index}/{len(groups)}] Wrote MixMCP packet for {event_ticker}")

    typer.echo(f"Wrote {len(groups)} MixMCP requests")
    if not dry_run:
        typer.echo(f"Wrote {written_packets} MixMCP packets")
        typer.echo(f"Skipped {skipped_packets} existing MixMCP packets")


def _fetch_and_write_web_evidence_packet(
    payload: dict,
    *,
    event_ticker: str,
    packet_path: Path,
    usage_path: Path,
    config: PredictionRunConfig,
    timeout_seconds: float,
) -> None:
    result = _fetch_web_evidence_packet(payload, timeout_seconds=timeout_seconds)
    packet, usage_metadata = _coerce_fetch_result(result)
    _write_json_checked(
        packet_path,
        packet.model_dump(mode="json"),
        config=config,
        artifact_kind="web_evidence_packet",
    )
    _write_json_checked(
        usage_path,
        {
            "event_ticker": event_ticker,
            "request_model": payload.get("model"),
            **usage_metadata,
        },
        config=config,
        artifact_kind="web_evidence_usage",
    )


@app.command("evaluate")
def evaluate_command(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    run_id: Annotated[str, typer.Option()],
    out_dir: Annotated[Path, typer.Option()],
    web_evidence_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    mixmcp_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    exclude_events_manifest: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False)
    ] = None,
    mixmcp_alpha: Annotated[float | None, typer.Option(min=0.0, max=1.0)] = None,
    mixmcp_alpha_mode: Annotated[str, typer.Option()] = "global",
    min_training_events: Annotated[int, typer.Option(min=1)] = 5,
    epochs: Annotated[int, typer.Option(min=1)] = 100,
    learning_rate: Annotated[float, typer.Option(min=0.0)] = 0.05,
    l2: Annotated[float, typer.Option(min=0.0)] = 0.001,
    residual_clip: Annotated[float, typer.Option(min=0.01)] = 2.0,
    target_side: Annotated[str, typer.Option()] = "yes",
    positive_label_weight: Annotated[float, typer.Option(min=0.01)] = 1.0,
    feature_ablation_group: Annotated[str, typer.Option()] = "none",
) -> None:
    rows, feature_rows, predictions = _predict_from_csv(
        input_csv,
        web_evidence_dir=web_evidence_dir,
        mixmcp_dir=mixmcp_dir,
        exclude_events_manifest=exclude_events_manifest,
        mixmcp_alpha=mixmcp_alpha,
        mixmcp_alpha_mode=mixmcp_alpha_mode,
        min_training_events=min_training_events,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        residual_clip=residual_clip,
        target_side=_validated_target_side(target_side),
        positive_label_weight=positive_label_weight,
        feature_ablation_group=_validated_feature_ablation_group(feature_ablation_group),
    )
    config = _run_config(run_id)
    _write_prediction_outputs(
        out_dir=out_dir,
        config=config,
        run_id=run_id,
        rows=rows,
        feature_rows=feature_rows,
        predictions=predictions,
        min_training_events=min_training_events,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        residual_clip=residual_clip,
        target_side=_validated_target_side(target_side),
        positive_label_weight=positive_label_weight,
        feature_ablation_group=_validated_feature_ablation_group(feature_ablation_group),
    )
    typer.echo(f"Predictions: {len(predictions)}")


@app.command("backtest")
def backtest_command(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    run_id: Annotated[str, typer.Option()],
    out_dir: Annotated[Path, typer.Option()],
    web_evidence_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    mixmcp_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    exclude_events_manifest: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False)
    ] = None,
    mixmcp_alpha: Annotated[float | None, typer.Option(min=0.0, max=1.0)] = None,
    mixmcp_alpha_mode: Annotated[str, typer.Option()] = "global",
    min_training_events: Annotated[int, typer.Option(min=1)] = 5,
    epochs: Annotated[int, typer.Option(min=1)] = 100,
    learning_rate: Annotated[float, typer.Option(min=0.0)] = 0.05,
    l2: Annotated[float, typer.Option(min=0.0)] = 0.001,
    residual_clip: Annotated[float, typer.Option(min=0.01)] = 2.0,
    margin: Annotated[float, typer.Option(min=0.0)] = 0.0,
    trade_side: Annotated[str, typer.Option()] = "all",
    target_side: Annotated[str, typer.Option()] = "yes",
    positive_label_weight: Annotated[float, typer.Option(min=0.01)] = 1.0,
    feature_ablation_group: Annotated[str, typer.Option()] = "none",
) -> None:
    rows, _, predictions = _predict_from_csv(
        input_csv,
        web_evidence_dir=web_evidence_dir,
        mixmcp_dir=mixmcp_dir,
        exclude_events_manifest=exclude_events_manifest,
        mixmcp_alpha=mixmcp_alpha,
        mixmcp_alpha_mode=mixmcp_alpha_mode,
        min_training_events=min_training_events,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        residual_clip=residual_clip,
        target_side=_validated_target_side(target_side),
        positive_label_weight=positive_label_weight,
        feature_ablation_group=_validated_feature_ablation_group(feature_ablation_group),
    )
    config = _run_config(run_id)
    trades = _build_trades(rows, predictions, margin=margin, trade_side=trade_side)
    _write_trades_csv_checked(out_dir / "trades.csv", trades, config=config)
    _write_json_checked(
        out_dir / "backtest.json",
        {
            "run_id": run_id,
            "summary": _summarize_trades(trades),
            "trades": trades,
        },
        config=config,
        artifact_kind="backtest_report",
    )
    typer.echo(f"Trades: {len(trades)}")


@app.command("sweep")
def sweep_command(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    run_id: Annotated[str, typer.Option()],
    out_dir: Annotated[Path, typer.Option()],
    web_evidence_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    mixmcp_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    exclude_events_manifest: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False)
    ] = None,
    min_training_events: Annotated[int, typer.Option(min=1)] = 5,
    epochs_grid: Annotated[str, typer.Option()] = "5,20,100",
    learning_rate_grid: Annotated[str, typer.Option()] = "0.05",
    l2_grid: Annotated[str, typer.Option()] = "0.0001,0.001,0.01,0.1",
    residual_clip_grid: Annotated[str, typer.Option()] = "0.5,1.0,2.0",
    margin_grid: Annotated[str, typer.Option()] = "0,0.005,0.01,0.02",
    trade_side_grid: Annotated[str, typer.Option()] = "all,no_only,yes_only",
    target_side_grid: Annotated[str, typer.Option()] = "yes",
    positive_label_weight_grid: Annotated[str, typer.Option()] = "1.0",
    feature_ablation_group_grid: Annotated[str, typer.Option()] = "none",
) -> None:
    config = _run_config(run_id)
    results = []
    for epochs in _parse_int_grid(epochs_grid):
        for learning_rate in _parse_float_grid(learning_rate_grid):
            for l2 in _parse_float_grid(l2_grid):
                for residual_clip in _parse_float_grid(residual_clip_grid):
                    for target_side in _parse_str_grid(target_side_grid):
                        validated_target_side = _validated_target_side(target_side)
                        for feature_ablation_group in _parse_str_grid(
                            feature_ablation_group_grid
                        ):
                            validated_ablation = _validated_feature_ablation_group(
                                feature_ablation_group
                            )
                            for positive_label_weight in _parse_float_grid(
                                positive_label_weight_grid
                            ):
                                rows, _, predictions = _predict_from_csv(
                                    input_csv,
                                    web_evidence_dir=web_evidence_dir,
                                    mixmcp_dir=mixmcp_dir,
                                    exclude_events_manifest=exclude_events_manifest,
                                    min_training_events=min_training_events,
                                    epochs=epochs,
                                    learning_rate=learning_rate,
                                    l2=l2,
                                    residual_clip=residual_clip,
                                    target_side=validated_target_side,
                                    positive_label_weight=positive_label_weight,
                                    feature_ablation_group=validated_ablation,
                                )
                                evaluation = _summarize_predictions(rows, predictions)
                                for margin in _parse_float_grid(margin_grid):
                                    for trade_side in _parse_str_grid(trade_side_grid):
                                        trades = _build_trades(
                                            rows,
                                            predictions,
                                            margin=margin,
                                            trade_side=trade_side,
                                        )
                                        results.append(
                                            {
                                                "config": {
                                                    "epochs": epochs,
                                                    "learning_rate": learning_rate,
                                                    "l2": l2,
                                                    "residual_clip": residual_clip,
                                                    "target_side": validated_target_side,
                                                    "positive_label_weight": (
                                                        positive_label_weight
                                                    ),
                                                    "feature_ablation_group": (
                                                        validated_ablation
                                                    ),
                                                    "margin": margin,
                                                    "trade_side": trade_side,
                                                },
                                                "evaluation": evaluation,
                                                "backtest": _summarize_trades(trades),
                                            }
                                        )
    results = sorted(
        results,
        key=lambda row: (
            -row["backtest"]["roi_on_cost"],
            row["evaluation"]["brier_score"] or 1.0,
        ),
    )
    _write_json_checked(
        out_dir / "sweep-summary.json",
        {"run_id": run_id, "results": results},
        config=config,
        artifact_kind="sweep_report",
    )
    _write_sweep_csv_checked(out_dir / "sweep-results.csv", results, config=config)
    typer.echo(f"Sweep results: {len(results)}")


@app.command("audit-web-evidence")
def audit_web_evidence_command(
    web_evidence_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out_dir: Annotated[Path, typer.Option()],
    run_id: Annotated[str, typer.Option()] = "web-evidence-audit",
) -> None:
    config = _run_config(run_id)
    checked_out_dir = config.validate_output_path(
        out_dir,
        artifact_kind="web_evidence_audit",
    )
    _reject_frozen_model_output(checked_out_dir)
    report = audit_web_evidence_dir(web_evidence_dir)
    write_audit_reports(report, checked_out_dir)
    typer.echo(f"Issues: {report['summary']['issue_count']}")


@app.command("cleanup")
def cleanup_command(run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    deleted = 0
    for path in run_dir.rglob("*"):
        if path.is_file() and path.suffix in {".tmp", ".scratch"}:
            path.unlink()
            deleted += 1
    typer.echo(f"Deleted {deleted} scratch artifacts")


def _predict_from_csv(
    input_csv: Path,
    *,
    web_evidence_dir: Path | None = None,
    mixmcp_dir: Path | None = None,
    exclude_events_manifest: Path | None = None,
    mixmcp_alpha: float | None = None,
    mixmcp_alpha_mode: str = "global",
    min_training_events: int,
    epochs: int,
    learning_rate: float,
    l2: float = 0.001,
    residual_clip: float = 2.0,
    target_side: str = "yes",
    positive_label_weight: float = 1.0,
    feature_ablation_group: str = "none",
) -> tuple[list[PredictionInputRow], list[dict[str, float]], list[ResidualPrediction]]:
    rows = _read_rows(input_csv)
    excluded_events = _load_excluded_events(exclude_events_manifest)
    if excluded_events:
        rows = [row for row in rows if row.event_ticker not in excluded_events]
    web_evidence_by_event = _load_web_evidence_packets(web_evidence_dir)
    mixmcp_by_event = _load_mixmcp_packets(mixmcp_dir)
    feature_rows = build_feature_matrix(rows, {}, web_evidence_by_event)
    feature_rows = _apply_feature_ablation(feature_rows, feature_ablation_group)
    predictions = walk_forward_predictions(
        rows,
        feature_rows,
        min_training_events=min_training_events,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        residual_clip=residual_clip,
        target_side=_validated_target_side(target_side),
        positive_label_weight=positive_label_weight,
    )
    predictions = apply_mixmcp_to_predictions(
        rows,
        predictions,
        mixmcp_by_event,
        alpha_mode=_validated_mixmcp_alpha_mode(mixmcp_alpha_mode),
        fixed_alpha=mixmcp_alpha,
    )
    return rows, feature_rows, predictions


def _write_prediction_outputs(
    *,
    out_dir: Path,
    config: PredictionRunConfig,
    run_id: str,
    rows: list[PredictionInputRow],
    feature_rows: list[dict[str, float]],
    predictions: list[ResidualPrediction],
    min_training_events: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    residual_clip: float,
    target_side: str,
    positive_label_weight: float,
    feature_ablation_group: str,
) -> None:
    _write_json_checked(
        out_dir / "run-config.json",
        config.model_dump(mode="json"),
        config=config,
        artifact_kind="run_config",
    )
    _write_json_checked(
        out_dir / "model-summary.json",
        {
            "model_type": "linear_residual_walk_forward",
            "min_training_events": min_training_events,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "residual_clip": residual_clip,
            "target_side": target_side,
            "positive_label_weight": positive_label_weight,
            "feature_ablation_group": feature_ablation_group,
            "prediction_count": len(predictions),
            "scenario_catalog_count": 0,
        },
        config=config,
        artifact_kind="model_summary",
    )
    _write_predictions_csv_checked(out_dir / "predictions.csv", predictions, config=config)
    _write_json_checked(
        out_dir / "feature-matrix.json",
        {
            "run_id": run_id,
            "web_evidence_count": sum(
                1
                for feature_row in feature_rows
                if feature_row.get("web_evidence_available", 0.0) > 0.0
            ),
            "feature_rows": feature_rows,
        },
        config=config,
        artifact_kind="feature_matrix",
    )
    _write_json_checked(
        out_dir / "evaluation.json",
        {
            "run_id": run_id,
            "summary": _summarize_predictions(rows, predictions),
        },
        config=config,
        artifact_kind="evaluation_report",
    )


def _read_rows(path: Path) -> list[PredictionInputRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            PredictionInputRow.model_validate(_clean_csv_row(row))
            for row in csv.DictReader(handle)
        ]


def _clean_csv_row(row: dict[str, str]) -> dict[str, str | None]:
    cleaned: dict[str, str | None] = {}
    for key, value in row.items():
        cleaned[key] = None if value == "" and key in {"status", "settlement_ts"} else value
    return cleaned


def _load_web_evidence_packets(path: Path | None) -> dict[str, WebEvidencePacket]:
    if path is None:
        return {}
    packet_dir = path / "packets" if (path / "packets").is_dir() else path
    packets = {}
    for packet_path in packet_dir.glob("*.json"):
        packet = parse_web_evidence_response(packet_path.read_text(encoding="utf-8"))
        packets[packet.event_ticker] = packet
    return packets


def _load_mixmcp_packets(path: Path | None) -> dict[str, MixMcpEventPacket]:
    if path is None:
        return {}
    packet_dir = path / "packets" if (path / "packets").is_dir() else path
    packets = {}
    for packet_path in packet_dir.glob("*.json"):
        packet = parse_mixmcp_response(packet_path.read_text(encoding="utf-8"))
        packets[packet.event_ticker] = packet
    return packets


def _validated_mixmcp_alpha_mode(value: str) -> str:
    if value not in {"global", "side"}:
        raise ValueError("mixmcp alpha mode must be 'global' or 'side'")
    return value


def _validated_target_side(value: str) -> str:
    if value not in {"yes", "no"}:
        raise ValueError("target side must be 'yes' or 'no'")
    return value


def _validated_feature_ablation_group(value: str) -> str:
    if value not in _FEATURE_ABLATION_PREFIXES:
        allowed = ", ".join(sorted(_FEATURE_ABLATION_PREFIXES))
        raise ValueError(f"feature ablation group must be one of: {allowed}")
    return value


def _apply_feature_ablation(
    feature_rows: list[dict[str, float]],
    feature_ablation_group: str,
) -> list[dict[str, float]]:
    validated_group = _validated_feature_ablation_group(feature_ablation_group)
    prefixes = _FEATURE_ABLATION_PREFIXES[validated_group]
    if not prefixes:
        return feature_rows
    return [
        {
            key: value
            for key, value in feature_row.items()
            if not key.startswith(prefixes)
        }
        for feature_row in feature_rows
    ]


def _load_excluded_events(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("event_tickers", [])
    if not isinstance(values, list):
        raise ValueError("exclusion manifest must contain an event_tickers list")
    return {str(value) for value in values}


def _reject_frozen_model_output(path: Path) -> None:
    lowered = [part.lower() for part in path.parts]
    if "models" in lowered:
        raise ValueError("generated v2 artifacts must not be written under models/")


def _enforce_paid_run_caps(
    *,
    paid_call_count: int,
    max_paid_calls: int | None,
    max_estimated_cost_dollars: float | None,
    estimated_cost_per_call_dollars: float,
) -> None:
    if max_paid_calls is not None and paid_call_count > max_paid_calls:
        raise RuntimeError(
            f"pending paid calls ({paid_call_count}) exceed max paid calls ({max_paid_calls})"
        )
    estimated_cost = paid_call_count * estimated_cost_per_call_dollars
    if max_estimated_cost_dollars is not None and estimated_cost > max_estimated_cost_dollars:
        raise RuntimeError(
            f"estimated cost ${estimated_cost:.2f} exceeds cap "
            f"${max_estimated_cost_dollars:.2f}"
        )


def _coerce_fetch_result(
    result: WebEvidenceFetchResult | WebEvidencePacket,
) -> tuple[WebEvidencePacket, dict[str, Any]]:
    if isinstance(result, WebEvidenceFetchResult):
        return result.packet, result.usage_metadata
    return result, {}


def _fetch_web_evidence_packet(
    payload: dict,
    *,
    timeout_seconds: float,
) -> WebEvidenceFetchResult:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when --no-dry-run is used")
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
    response_payload = response.json()
    packet = parse_web_evidence_response(_response_output_text(response_payload))
    return WebEvidenceFetchResult(
        packet=packet,
        usage_metadata=_openai_usage_metadata(response_payload),
    )


def _fetch_mixmcp_packet(
    payload: dict,
    *,
    timeout_seconds: float,
) -> MixMcpEventPacket:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when --no-dry-run is used")
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
    return parse_mixmcp_response(_response_output_text(response.json()))


def _openai_usage_metadata(response_payload: dict[str, Any]) -> dict[str, Any]:
    usage = response_payload.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return {
        "response_id": response_payload.get("id"),
        "response_model": response_payload.get("model"),
        "usage": usage,
    }


def _response_output_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    for output in payload.get("output", []):
        for content in output.get("content", []):
            if isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("OpenAI response did not include output text")


def _run_config(run_id: str) -> PredictionRunConfig:
    return PredictionRunConfig(
        run_id=run_id,
        decision_time_column="snapshot_target_time",
        artifact_retention_policy=ArtifactRetentionPolicy(
            canonical_source_files=_CANONICAL_SOURCE_FILES,
        ),
    )


def _group_rows_by_event(
    rows: list[PredictionInputRow],
) -> list[tuple[str, list[PredictionInputRow]]]:
    grouped: dict[str, list[PredictionInputRow]] = defaultdict(list)
    for row in rows:
        grouped[row.event_ticker].append(row)
    return sorted(
        grouped.items(),
        key=lambda item: (min(row.close_time for row in item[1]), item[0]),
    )


def _company_name_from_event_phrase(event_phrase: str) -> str:
    lowered = event_phrase.lower()
    if lowered.startswith("what will ") and " say" in lowered:
        return event_phrase[len("What will ") : lowered.index(" say")].strip()
    return event_phrase


def _write_json_checked(
    path: Path,
    payload: dict,
    *,
    config: PredictionRunConfig,
    artifact_kind: str,
) -> None:
    checked_path = config.validate_output_path(path, artifact_kind=artifact_kind)
    checked_path.parent.mkdir(parents=True, exist_ok=True)
    checked_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_predictions_csv_checked(
    path: Path,
    predictions: list[ResidualPrediction],
    *,
    config: PredictionRunConfig,
) -> None:
    checked_path = config.validate_output_path(path, artifact_kind="predictions")
    checked_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "market_ticker",
        "event_ticker",
        "probability",
        "market_probability",
        "residual_delta",
        "training_event_tickers",
        "reasons",
    ]
    with checked_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(
                {
                    "market_ticker": prediction.market_ticker,
                    "event_ticker": prediction.event_ticker,
                    "probability": str(prediction.probability),
                    "market_probability": str(prediction.market_probability),
                    "residual_delta": prediction.residual_delta,
                    "training_event_tickers": ";".join(prediction.training_event_tickers),
                    "reasons": ";".join(prediction.reasons),
                }
            )


def _write_trades_csv_checked(
    path: Path,
    trades: list[dict],
    *,
    config: PredictionRunConfig,
) -> None:
    checked_path = config.validate_output_path(path, artifact_kind="trades")
    checked_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "market_ticker",
        "event_ticker",
        "side",
        "probability",
        "yes_bid",
        "yes_ask",
        "outcome",
        "cost",
        "pnl",
    ]
    with checked_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)


def _write_sweep_csv_checked(
    path: Path,
    results: list[dict],
    *,
    config: PredictionRunConfig,
) -> None:
    checked_path = config.validate_output_path(path, artifact_kind="sweep_report")
    checked_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epochs",
        "learning_rate",
        "l2",
        "residual_clip",
        "target_side",
        "positive_label_weight",
        "feature_ablation_group",
        "margin",
        "trade_side",
        "prediction_count",
        "brier_score",
        "market_brier_score",
        "trades",
        "total_cost",
        "total_pnl",
        "roi_on_cost",
    ]
    with checked_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    **result["config"],
                    **result["evaluation"],
                    **result["backtest"],
                }
            )


def _summarize_predictions(
    rows: list[PredictionInputRow],
    predictions: list[ResidualPrediction],
) -> dict:
    rows_by_ticker = {row.market_ticker: row for row in rows}
    prediction_rows = [
        (prediction, rows_by_ticker[prediction.market_ticker])
        for prediction in predictions
        if prediction.market_ticker in rows_by_ticker
    ]
    if not prediction_rows:
        return {
            "prediction_count": 0,
            "brier_score": None,
            "market_brier_score": None,
        }
    return {
        "prediction_count": len(prediction_rows),
        "brier_score": _mean(
            (float(prediction.probability) - row.outcome_label) ** 2
            for prediction, row in prediction_rows
        ),
        "market_brier_score": _mean(
            (float(row.preclose_yes_mid) - row.outcome_label) ** 2
            for _, row in prediction_rows
        ),
    }


def _build_trades(
    rows: list[PredictionInputRow],
    predictions: list[ResidualPrediction],
    *,
    margin: float,
    trade_side: str = "all",
) -> list[dict]:
    if trade_side not in {"all", "no_only", "yes_only"}:
        raise ValueError("trade side must be all, no_only, or yes_only")
    rows_by_ticker = {row.market_ticker: row for row in rows}
    trades = []
    for prediction in predictions:
        row = rows_by_ticker.get(prediction.market_ticker)
        if row is None:
            continue
        probability = float(prediction.probability)
        yes_bid = float(row.preclose_yes_bid)
        yes_ask = float(row.preclose_yes_ask)
        outcome = row.outcome_label
        if probability > yes_ask + margin:
            if trade_side == "no_only":
                continue
            side = "YES"
            cost = yes_ask
            pnl = outcome - cost
        elif probability < yes_bid - margin:
            if trade_side == "yes_only":
                continue
            side = "NO"
            cost = 1 - yes_bid
            pnl = yes_bid - outcome
        else:
            continue
        trades.append(
            {
                "market_ticker": row.market_ticker,
                "event_ticker": row.event_ticker,
                "side": side,
                "probability": round(probability, 6),
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "outcome": outcome,
                "cost": round(cost, 6),
                "pnl": round(pnl, 6),
            }
        )
    return trades


def _summarize_trades(trades: list[dict]) -> dict:
    total_cost = sum(trade["cost"] for trade in trades)
    total_pnl = sum(trade["pnl"] for trade in trades)
    return {
        "trades": len(trades),
        "total_cost": round(total_cost, 6),
        "total_pnl": round(total_pnl, 6),
        "roi_on_cost": round(total_pnl / total_cost, 6) if total_cost else 0.0,
    }


def _mean(values) -> float:
    collected = list(values)
    return round(sum(collected) / len(collected), 6)


def _parse_float_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_int_grid(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_str_grid(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    app()
