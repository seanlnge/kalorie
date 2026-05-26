from collections import defaultdict

from kalorie.benchmarking.packs import (
    BenchmarkPack,
    BenchmarkRunMetadata,
    validate_benchmark_pack,
)
from kalorie.domain.models import FeatureVector
from kalorie.ml.market_residual import MarketResidualArtifact, predict_market_residual
from kalorie.ml.model1 import MentionModelArtifact, predict_model1


def run_model1_pack_benchmark(
    pack: BenchmarkPack,
    model: MentionModelArtifact,
    metadata: BenchmarkRunMetadata,
    *,
    company_model_map: dict[str, str] | None = None,
) -> dict:
    if metadata.model_family == "company_niched" and not company_model_map:
        raise ValueError("company_model_map is required for model_family=company_niched")

    validate_benchmark_pack(pack)
    snapshot_by_market = {snapshot.market_id: snapshot for snapshot in pack.snapshots}
    rows = []
    missing_snapshot_count = 0
    missing_quote_count = 0
    for example in pack.examples:
        snapshot = snapshot_by_market.get(example.market_id)
        if snapshot is None:
            missing_snapshot_count += 1
            continue
        market_mid = _market_mid(snapshot)
        prediction = predict_model1(
            model,
            company_symbol=example.company_symbol,
            feature_vector=FeatureVector(
                target_phrase=example.target_phrase,
                features=example.features,
            ),
            market_probability=market_mid,
        )
        rows.append(
            _benchmark_row(
                example=example,
                snapshot=snapshot,
                metadata=metadata,
                model_version=model.model_version,
                probability=prediction.probability,
                reasons=prediction.reasons,
            )
        )

    report = _metric_report(rows)
    report.update(
        {
            "model_family": metadata.model_family,
            "model_path": metadata.model_path,
            "pack_path": metadata.pack_path,
            "pack_id": pack.manifest.pack_id,
            "pack_split": pack.manifest.split,
            "excluded_events": metadata.excluded_events,
            "calibration": metadata.calibration,
            "skip_summary": {
                "total_examples": len(pack.examples),
                "evaluated_rows": len(rows),
                "skipped_rows": len(pack.examples) - len(rows),
                "missing_snapshot_count": missing_snapshot_count,
                "missing_quote_count": missing_quote_count,
            },
            "rows": rows,
        }
    )
    return report


def run_market_residual_pack_benchmark(
    pack: BenchmarkPack,
    model: MarketResidualArtifact,
    metadata: BenchmarkRunMetadata,
) -> dict:
    validate_benchmark_pack(pack)
    snapshot_by_market = {snapshot.market_id: snapshot for snapshot in pack.snapshots}
    rows = []
    missing_snapshot_count = 0
    for example in pack.examples:
        snapshot = snapshot_by_market.get(example.market_id)
        if snapshot is None:
            missing_snapshot_count += 1
            continue
        market_mid = _market_mid(snapshot)
        prediction = predict_market_residual(
            model,
            company_symbol=example.company_symbol,
            feature_vector=FeatureVector(
                target_phrase=example.target_phrase,
                features=example.features,
            ),
            market_probability=market_mid,
        )
        rows.append(
            _benchmark_row(
                example=example,
                snapshot=snapshot,
                metadata=metadata,
                model_version=model.model_version,
                probability=prediction.probability,
                reasons=prediction.reasons,
            )
        )

    report = _metric_report(rows)
    report.update(
        {
            "model_family": metadata.model_family,
            "model_path": metadata.model_path,
            "pack_path": metadata.pack_path,
            "pack_id": pack.manifest.pack_id,
            "pack_split": pack.manifest.split,
            "excluded_events": metadata.excluded_events,
            "calibration": metadata.calibration,
            "skip_summary": {
                "total_examples": len(pack.examples),
                "evaluated_rows": len(rows),
                "skipped_rows": len(pack.examples) - len(rows),
                "missing_snapshot_count": missing_snapshot_count,
                "missing_quote_count": 0,
            },
            "rows": rows,
        }
    )
    return report


def _benchmark_row(
    *,
    example,
    snapshot,
    metadata: BenchmarkRunMetadata,
    model_version: str,
    probability: float,
    reasons: list[str],
) -> dict:
    return {
        "event_id": snapshot.event_ticker,
        "market_id": example.market_id,
        "target_phrase": example.target_phrase,
        "label": example.label,
        "model_probability": probability,
        "prediction_source": f"{metadata.model_family}:{model_version}",
        "prediction_reasons": reasons,
        "kalshi_yes_bid": float(snapshot.preclose_yes_bid),
        "kalshi_yes_ask": float(snapshot.preclose_yes_ask),
        "kalshi_yes_mid": _market_mid(snapshot),
        "snapshot_target_time": snapshot.snapshot_target_time.isoformat(),
        "candle_end_ts": snapshot.candle_end_ts,
    }


def _market_mid(snapshot) -> float:
    return round(
        (float(snapshot.preclose_yes_bid) + float(snapshot.preclose_yes_ask)) / 2,
        6,
    )


def _metric_report(rows: list[dict]) -> dict:
    labels = [int(row["label"]) for row in rows]
    event_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        event_rows[str(row["event_id"])].append(row)
    report = {
        "sample_count": len(rows),
        "model": _probability_metric_block(
            [float(row["model_probability"]) for row in rows],
            labels,
        ),
        "kalshi_yes_bid": _probability_metric_block(
            [float(row["kalshi_yes_bid"]) for row in rows],
            labels,
        ),
        "kalshi_yes_ask": _probability_metric_block(
            [float(row["kalshi_yes_ask"]) for row in rows],
            labels,
        ),
        "kalshi_yes_mid": _probability_metric_block(
            [float(row["kalshi_yes_mid"]) for row in rows],
            labels,
        ),
        "per_event": {},
    }
    report["deltas_vs_kalshi"] = {
        "model_minus_yes_bid_brier": round(
            report["model"]["brier_score"] - report["kalshi_yes_bid"]["brier_score"],
            6,
        ),
        "model_minus_yes_ask_brier": round(
            report["model"]["brier_score"] - report["kalshi_yes_ask"]["brier_score"],
            6,
        ),
        "model_minus_yes_mid_brier": round(
            report["model"]["brier_score"] - report["kalshi_yes_mid"]["brier_score"],
            6,
        ),
        "model_minus_yes_mid_ece": round(
            report["model"]["expected_calibration_error"]
            - report["kalshi_yes_mid"]["expected_calibration_error"],
            6,
        ),
    }
    report["per_event"] = {
        event_id: _metric_report_for_event(event_event_rows)
        for event_id, event_event_rows in sorted(event_rows.items())
    }
    return report


def _metric_report_for_event(rows: list[dict]) -> dict:
    report = _metric_report_without_events(rows)
    report["deltas_vs_kalshi"] = {
        "model_minus_yes_mid_brier": round(
            report["model"]["brier_score"] - report["kalshi_yes_mid"]["brier_score"],
            6,
        )
    }
    return report


def _metric_report_without_events(rows: list[dict]) -> dict:
    labels = [int(row["label"]) for row in rows]
    return {
        "sample_count": len(rows),
        "model": _probability_metric_block(
            [float(row["model_probability"]) for row in rows],
            labels,
        ),
        "kalshi_yes_mid": _probability_metric_block(
            [float(row["kalshi_yes_mid"]) for row in rows],
            labels,
        ),
    }


def _probability_metric_block(probabilities: list[float], labels: list[int]) -> dict:
    return {
        "brier_score": _brier_score(probabilities, labels),
        "expected_calibration_error": _ece_score(probabilities, labels),
    }


def _brier_score(probabilities: list[float], labels: list[int]) -> float:
    if not labels:
        return 0.0
    return round(
        sum(
            (probability - label) ** 2
            for probability, label in zip(probabilities, labels, strict=True)
        )
        / len(labels),
        6,
    )


def _ece_score(probabilities: list[float], labels: list[int], *, bins: int = 10) -> float:
    if not labels:
        return 0.0
    total = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            bucket = [
                (probability, label)
                for probability, label in zip(probabilities, labels, strict=True)
                if lower <= probability <= upper
            ]
        else:
            bucket = [
                (probability, label)
                for probability, label in zip(probabilities, labels, strict=True)
                if lower <= probability < upper
            ]
        if not bucket:
            continue
        mean_probability = sum(probability for probability, _ in bucket) / len(bucket)
        positive_rate = sum(label for _, label in bucket) / len(bucket)
        total += (len(bucket) / len(labels)) * abs(mean_probability - positive_rate)
    return round(total, 6)
