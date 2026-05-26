import json
from pathlib import Path

from kalorie2.web_evidence_audit import audit_web_evidence_dir


def test_audit_web_evidence_flags_transcript_and_post_cutoff_items(tmp_path: Path):
    packet_dir = tmp_path / "packets"
    packet_dir.mkdir()
    (packet_dir / "EVENT1.json").write_text(
        json.dumps(
            {
                "event_ticker": "EVENT1",
                "company_name": "Example Co",
                "cutoff_time": "2026-01-01T12:00:00Z",
                "items": [
                    {
                        "title": "Example Co earnings call transcript",
                        "url": "https://seekingalpha.com/article/example-transcript",
                        "source": "Seeking Alpha",
                        "published_at": "2026-01-01T10:00:00Z",
                        "snippet": "Prepared remarks from the call transcript.",
                        "target_phrases": ["tariff"],
                        "evidence_strength": 0.9,
                    },
                    {
                        "title": "Clean preview",
                        "url": "https://example.com/preview",
                        "source": "Example News",
                        "published_at": "2026-01-02T10:00:00Z",
                        "snippet": "Published after cutoff.",
                        "target_phrases": ["tariff"],
                        "evidence_strength": 0.4,
                    },
                    {
                        "title": "Undated preview",
                        "url": "https://example.com/undated",
                        "source": "Example News",
                        "published_at": None,
                        "snippet": "No publication date.",
                        "target_phrases": ["tariff"],
                        "evidence_strength": 0.4,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = audit_web_evidence_dir(packet_dir)

    issue_codes = {issue["issue_code"] for issue in report["issues"]}
    assert "transcript_or_post_call_content" in issue_codes
    assert "post_cutoff_published_at" in issue_codes
    assert "undated_item" in issue_codes
    assert report["summary"]["packet_count"] == 1
    assert report["summary"]["issue_count"] == 3
    assert report["exclusion_manifest"]["event_tickers"] == ["EVENT1"]
