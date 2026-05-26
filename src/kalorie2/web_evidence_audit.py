from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from kalorie2.web_evidence import WebEvidencePacket, parse_web_evidence_response

_LEAKAGE_TERMS = (
    "transcript",
    "earnings call transcript",
    "call audio",
    "post-call",
    "post call",
    "recap",
    "prepared remarks",
    "alphasense",
    "seeking alpha",
    "motley fool",
    "resolution",
    "kalshi",
    "prediction market",
    "final results",
)

_LEAKAGE_DOMAINS = (
    "seekingalpha.com",
    "fool.com",
    "alphasense.com",
    "kalshi.com",
    "quartr.com",
)


def audit_web_evidence_dir(path: Path) -> dict[str, Any]:
    packet_dir = path / "packets" if (path / "packets").is_dir() else path
    issues = []
    packet_count = 0
    item_count = 0
    for packet_path in sorted(packet_dir.glob("*.json")):
        packet_count += 1
        packet = parse_web_evidence_response(packet_path.read_text(encoding="utf-8"))
        for item_index, item in enumerate(packet.items):
            item_count += 1
            issues.extend(_audit_item(packet, item_index, item.model_dump(mode="json")))
    severe_events = sorted(
        {
            issue["event_ticker"]
            for issue in issues
            if issue["severity"] in {"warning", "high"}
        }
    )
    summary = {
        "packet_count": packet_count,
        "item_count": item_count,
        "issue_count": len(issues),
        "events_with_issues": len(severe_events),
    }
    return {
        "summary": summary,
        "issues": issues,
        "exclusion_manifest": {
            "event_tickers": severe_events,
            "reason": "events with warning/high web-evidence audit issues",
        },
    }


def _audit_item(
    packet: WebEvidencePacket,
    item_index: int,
    item: dict[str, Any],
) -> list[dict[str, Any]]:
    issues = []
    published_at = item.get("published_at")
    if published_at is None:
        issues.append(_issue(packet, item_index, item, "undated_item", "warning"))
    elif _parse_datetime(str(published_at)) > packet.cutoff_time:
        issues.append(_issue(packet, item_index, item, "post_cutoff_published_at", "high"))

    text = " ".join(
        str(item.get(field, "")) for field in ("title", "url", "source", "snippet")
    ).lower()
    if any(term in text for term in _LEAKAGE_TERMS) or any(
        domain in text for domain in _LEAKAGE_DOMAINS
    ):
        issues.append(
            _issue(packet, item_index, item, "transcript_or_post_call_content", "high")
        )
    return issues


def _issue(
    packet: WebEvidencePacket,
    item_index: int,
    item: dict[str, Any],
    issue_code: str,
    severity: str,
) -> dict[str, Any]:
    return {
        "event_ticker": packet.event_ticker,
        "item_index": item_index,
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "source": item.get("source", ""),
        "published_at": item.get("published_at"),
        "issue_code": issue_code,
        "severity": severity,
    }


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def write_audit_reports(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "web-evidence-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "exclusion-manifest.json").write_text(
        json.dumps(report["exclusion_manifest"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "web-evidence-audit.csv", report["issues"])


def _write_csv(path: Path, issues: list[dict[str, Any]]) -> None:
    import csv

    fieldnames = [
        "event_ticker",
        "item_index",
        "issue_code",
        "severity",
        "title",
        "url",
        "source",
        "published_at",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(issues)
