import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from kalorie.domain.models import MentionMarketContract, SourceDocument, TargetPhrase
from kalorie.io.transcript_corpus import TranscriptRecord
from kalorie.ml.real_training_data import (
    DEFAULT_SYNTHETIC_TARGET_PHRASES,
    build_examples_from_transcript_records,
    build_synthetic_phrase_examples_from_transcript_records,
)


def test_build_examples_from_transcript_records_uses_press_release_and_transcript_text(
    tmp_path: Path,
):
    transcript_path = tmp_path / "2025_Q2_wmt_processed.txt"
    transcript_path.write_text("Traffic improved and automation helped.", encoding="utf-8")
    release_path = tmp_path / "release.txt"
    release_path.write_text("Traffic was strong in Walmart U.S.", encoding="utf-8")
    record = TranscriptRecord(
        company_name="Walmart",
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        path=transcript_path,
    )
    document = SourceDocument(
        source_id="WMT-2025-Q2-release",
        company_symbol="WMT",
        document_type="sec_ex_99_1_press_release",
        source_path=str(release_path),
        published_at=datetime(2025, 8, 21, tzinfo=UTC),
        content_hash="abc",
    )
    contracts = [
        MentionMarketContract(
            venue="kalshi",
            market_id="KX-WMT-TRAFFIC",
            event_ticker="KX-WMT",
            title="What will Walmart say?",
            rules_text="If traffic is said by any Walmart representative.",
            target_phrase=TargetPhrase(phrase="traffic", normalized_phrase="traffic"),
            yes_bid=Decimal("0.40"),
            yes_ask=Decimal("0.45"),
            observed_at=datetime(2025, 8, 20, tzinfo=UTC),
        )
    ]

    examples = build_examples_from_transcript_records(
        records=[record],
        documents_by_period={("WMT", 2025, 2): [document]},
        contracts=contracts,
    )

    assert len(examples) == 1
    assert examples[0].label == 1
    assert examples[0].features["exact_match_count"] == 1.0
    assert examples[0].document_ids == ["WMT-2025-Q2-release"]
    assert examples[0].market_venue == "kalshi"


def test_build_examples_from_transcript_records_drops_company_mismatched_contracts(
    tmp_path: Path,
):
    transcript_path = tmp_path / "2025_Q2_wmt_processed.txt"
    transcript_path.write_text("Traffic improved and automation helped.", encoding="utf-8")
    release_path = tmp_path / "release.txt"
    release_path.write_text("Traffic was strong in Walmart U.S.", encoding="utf-8")
    record = TranscriptRecord(
        company_name="Walmart",
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        path=transcript_path,
    )
    document = SourceDocument(
        source_id="WMT-2025-Q2-release",
        company_symbol="WMT",
        document_type="sec_ex_99_1_press_release",
        source_path=str(release_path),
        published_at=datetime(2025, 8, 21, tzinfo=UTC),
        content_hash="abc",
    )
    cava_contract = MentionMarketContract(
        venue="kalshi",
        market_id="KXEARNINGSMENTIONCAVA-26MAY19-TRAF",
        event_ticker="KXEARNINGSMENTIONCAVA-26MAY19",
        title="What will CAVA Group, Inc. say during their next earnings call?",
        rules_text="If traffic is said by any CAVA representative.",
        target_phrase=TargetPhrase(phrase="traffic", normalized_phrase="traffic"),
        yes_bid=Decimal("0.40"),
        yes_ask=Decimal("0.45"),
        observed_at=datetime(2026, 5, 19, tzinfo=UTC),
    )

    examples = build_examples_from_transcript_records(
        records=[record],
        documents_by_period={("WMT", 2025, 2): [document]},
        contracts=[cava_contract],
    )

    assert examples == []


def test_build_examples_from_transcript_records_does_not_match_symbol_inside_title_words(
    tmp_path: Path,
):
    transcript_path = tmp_path / "2025_Q2_all_processed.txt"
    transcript_path.write_text("Traffic improved.", encoding="utf-8")
    release_path = tmp_path / "release.txt"
    release_path.write_text("Traffic improved.", encoding="utf-8")
    record = TranscriptRecord(
        company_name="Allstate",
        company_symbol="ALL",
        fiscal_year=2025,
        fiscal_quarter=2,
        path=transcript_path,
    )
    document = SourceDocument(
        source_id="ALL-2025-Q2-release",
        company_symbol="ALL",
        document_type="sec_ex_99_1_press_release",
        source_path=str(release_path),
        published_at=datetime(2025, 8, 21, tzinfo=UTC),
        content_hash="abc",
    )
    cava_contract = MentionMarketContract(
        venue="kalshi",
        market_id="KXEARNINGSMENTIONCAVA-26MAY19-TRAF",
        event_ticker="KXEARNINGSMENTIONCAVA-26MAY19",
        title="What will CAVA Group, Inc. say during their next earnings call?",
        rules_text="If traffic is said by any CAVA representative.",
        target_phrase=TargetPhrase(phrase="traffic", normalized_phrase="traffic"),
        yes_bid=Decimal("0.40"),
        yes_ask=Decimal("0.45"),
        observed_at=datetime(2026, 5, 19, tzinfo=UTC),
    )

    examples = build_examples_from_transcript_records(
        records=[record],
        documents_by_period={("ALL", 2025, 2): [document]},
        contracts=[cava_contract],
    )

    assert examples == []


def test_build_synthetic_phrase_examples_without_kalshi_contracts(tmp_path: Path):
    transcript_path = tmp_path / "2025_Q2_wmt_processed.txt"
    transcript_path.write_text("Traffic improved and automation helped.", encoding="utf-8")
    release_path = tmp_path / "release.txt"
    release_path.write_text("Automation investment improved operations.", encoding="utf-8")
    record = TranscriptRecord(
        company_name="Walmart",
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        path=transcript_path,
    )
    document = SourceDocument(
        source_id="WMT-2025-Q2-release",
        company_symbol="WMT",
        document_type="sec_ex_99_1_press_release",
        source_path=str(release_path),
        published_at=datetime(2025, 8, 21, tzinfo=UTC),
        content_hash="abc",
    )

    examples = build_synthetic_phrase_examples_from_transcript_records(
        records=[record],
        documents_by_period={("WMT", 2025, 2): [document]},
        target_phrases=["traffic", "automation", "robotaxi"],
    )

    by_target = {example.target_phrase: example for example in examples}
    assert sorted(by_target) == ["automation", "robotaxi", "traffic"]
    assert by_target["traffic"].label == 1
    assert by_target["automation"].features["exact_match_count"] == 1.0
    assert by_target["robotaxi"].label == 0
    assert by_target["traffic"].market_id == "WMT-2025-Q2-traffic"
    assert by_target["traffic"].market_probability == Decimal("0.50")
    assert by_target["traffic"].market_venue == "synthetic"


def test_build_synthetic_phrase_examples_excludes_implausibly_late_baseline_evidence(
    tmp_path: Path,
):
    transcript_path = tmp_path / "2025_Q2_wmt_processed.txt"
    transcript_path.write_text("Traffic improved and automation helped.", encoding="utf-8")
    late_release_path = tmp_path / "late-release.txt"
    late_release_path.write_text("Automation investment improved operations.", encoding="utf-8")
    record = TranscriptRecord(
        company_name="Walmart",
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        path=transcript_path,
    )
    late_document = SourceDocument(
        source_id="WMT-2025-Q2-late",
        company_symbol="WMT",
        document_type="sec_ex_99_1_press_release",
        source_path=str(late_release_path),
        published_at=datetime(2035, 12, 31, tzinfo=UTC),
        content_hash="abc",
    )

    examples = build_synthetic_phrase_examples_from_transcript_records(
        records=[record],
        documents_by_period={("WMT", 2025, 2): [late_document]},
        target_phrases=["traffic", "automation", "robotaxi"],
    )

    assert examples == []


def test_build_synthetic_phrase_examples_uses_ten_minute_pre_call_cutoff(tmp_path: Path):
    transcript_path = tmp_path / "2025_Q2_wmt_processed.txt"
    transcript_path.write_text("Traffic improved and automation helped.", encoding="utf-8")
    call_start = datetime(2025, 8, 22, 16, 0, tzinfo=UTC)
    os.utime(transcript_path, (call_start.timestamp(), call_start.timestamp()))
    expected_cutoff = datetime(2025, 8, 22, 15, 50, tzinfo=UTC)

    pre_cutoff_release_path = tmp_path / "pre-cutoff-release.txt"
    pre_cutoff_release_path.write_text("Traffic was strong.", encoding="utf-8")
    post_cutoff_release_path = tmp_path / "post-cutoff-release.txt"
    post_cutoff_release_path.write_text("Automation gains accelerated.", encoding="utf-8")

    record = TranscriptRecord(
        company_name="Walmart",
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        path=transcript_path,
    )
    pre_cutoff_document = SourceDocument(
        source_id="WMT-2025-Q2-pre-cutoff",
        company_symbol="WMT",
        document_type="sec_ex_99_1_press_release",
        source_path=str(pre_cutoff_release_path),
        published_at=datetime(2025, 8, 22, 15, 45, tzinfo=UTC),
        content_hash="abc",
    )
    post_cutoff_document = SourceDocument(
        source_id="WMT-2025-Q2-post-cutoff",
        company_symbol="WMT",
        document_type="sec_ex_99_1_press_release",
        source_path=str(post_cutoff_release_path),
        published_at=datetime(2025, 8, 22, 15, 55, tzinfo=UTC),
        content_hash="def",
    )

    examples = build_synthetic_phrase_examples_from_transcript_records(
        records=[record],
        documents_by_period={("WMT", 2025, 2): [pre_cutoff_document, post_cutoff_document]},
        target_phrases=["traffic"],
    )

    assert len(examples) == 1
    assert examples[0].evidence_cutoff == expected_cutoff
    assert examples[0].document_ids == [
        "WMT-2025-Q2-pre-cutoff",
        "WMT-2025-Q2-post-cutoff",
    ]
    assert examples[0].evidence_document_roles == {
        "WMT-2025-Q2-pre-cutoff": "event_baseline",
        "WMT-2025-Q2-post-cutoff": "event_baseline",
    }


def test_build_synthetic_phrase_examples_filters_news_but_keeps_event_baseline(
    tmp_path: Path,
):
    transcript_path = tmp_path / "2025_Q2_wmt_processed.txt"
    transcript_path.write_text("Traffic improved and automation helped.", encoding="utf-8")
    call_start = datetime(2025, 8, 22, 16, 0, tzinfo=UTC)
    os.utime(transcript_path, (call_start.timestamp(), call_start.timestamp()))

    release_path = tmp_path / "release.txt"
    release_path.write_text("Traffic was strong.", encoding="utf-8")
    early_news_path = tmp_path / "early-news.txt"
    early_news_path.write_text("Analysts expected traffic strength.", encoding="utf-8")
    late_news_path = tmp_path / "late-news.txt"
    late_news_path.write_text("Automation was discussed after cutoff.", encoding="utf-8")

    record = TranscriptRecord(
        company_name="Walmart",
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        path=transcript_path,
    )
    baseline_document = SourceDocument(
        source_id="WMT-2025-Q2-release",
        company_symbol="WMT",
        document_type="sec_ex_99_1_press_release",
        source_path=str(release_path),
        published_at=datetime(2025, 8, 22, 15, 55, tzinfo=UTC),
        content_hash="abc",
    )
    early_news_document = SourceDocument(
        source_id="WMT-2025-Q2-early-news",
        company_symbol="WMT",
        document_type="news_article_reliability_080",
        source_path=str(early_news_path),
        published_at=datetime(2025, 8, 22, 15, 45, tzinfo=UTC),
        content_hash="def",
    )
    late_news_document = SourceDocument(
        source_id="WMT-2025-Q2-late-news",
        company_symbol="WMT",
        document_type="news_article_reliability_080",
        source_path=str(late_news_path),
        published_at=datetime(2025, 8, 22, 15, 55, tzinfo=UTC),
        content_hash="ghi",
    )

    examples = build_synthetic_phrase_examples_from_transcript_records(
        records=[record],
        documents_by_period={
            ("WMT", 2025, 2): [
                baseline_document,
                early_news_document,
                late_news_document,
            ]
        },
        target_phrases=["traffic"],
    )

    assert len(examples) == 1
    assert examples[0].document_ids == [
        "WMT-2025-Q2-release",
        "WMT-2025-Q2-early-news",
    ]
    assert examples[0].evidence_document_roles == {
        "WMT-2025-Q2-release": "event_baseline",
        "WMT-2025-Q2-early-news": "time_sensitive",
    }
    assert "WMT-2025-Q2-late-news" not in examples[0].document_ids


def test_build_synthetic_phrase_examples_clamps_transcript_mtime_cutoff(tmp_path: Path):
    transcript_path = tmp_path / "2025_Q2_wmt_processed.txt"
    transcript_path.write_text("Traffic improved and automation helped.", encoding="utf-8")
    future_release_path = tmp_path / "future-release.txt"
    future_release_path.write_text("Automation investment improved operations.", encoding="utf-8")
    # Simulate a very late transcript ingestion timestamp.
    os.utime(
        transcript_path,
        (
            datetime(2030, 1, 1, tzinfo=UTC).timestamp(),
            datetime(2030, 1, 1, tzinfo=UTC).timestamp(),
        ),
    )
    record = TranscriptRecord(
        company_name="Walmart",
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        path=transcript_path,
    )
    future_document = SourceDocument(
        source_id="WMT-2025-Q2-future",
        company_symbol="WMT",
        document_type="sec_ex_99_1_press_release",
        source_path=str(future_release_path),
        published_at=datetime(2026, 6, 1, tzinfo=UTC),
        content_hash="abc",
    )

    examples = build_synthetic_phrase_examples_from_transcript_records(
        records=[record],
        documents_by_period={("WMT", 2025, 2): [future_document]},
        target_phrases=["traffic", "automation"],
    )

    assert examples == []


def test_build_synthetic_phrase_examples_uses_explicit_call_start_over_transcript_mtime(
    tmp_path: Path,
):
    transcript_path = tmp_path / "2025_Q2_wmt_processed.txt"
    transcript_path.write_text("Traffic improved and automation helped.", encoding="utf-8")
    os.utime(
        transcript_path,
        (
            datetime(2030, 1, 1, tzinfo=UTC).timestamp(),
            datetime(2030, 1, 1, tzinfo=UTC).timestamp(),
        ),
    )
    late_release_path = tmp_path / "late-release.txt"
    late_release_path.write_text("Traffic was strong.", encoding="utf-8")

    record = TranscriptRecord(
        company_name="Walmart",
        company_symbol="WMT",
        fiscal_year=2025,
        fiscal_quarter=2,
        path=transcript_path,
        call_start_at=datetime(2025, 8, 22, 16, 0, tzinfo=UTC),
        call_time_source="explicit",
    )
    late_document = SourceDocument(
        source_id="WMT-2025-Q2-late-release",
        company_symbol="WMT",
        document_type="sec_ex_99_1_press_release",
        source_path=str(late_release_path),
        published_at=datetime(2025, 8, 25, 16, 1, tzinfo=UTC),
        content_hash="abc",
    )

    examples = build_synthetic_phrase_examples_from_transcript_records(
        records=[record],
        documents_by_period={("WMT", 2025, 2): [late_document]},
        target_phrases=["traffic"],
    )

    assert examples == []


def test_build_synthetic_phrase_examples_supports_company_specific_targets(tmp_path: Path):
    release_path = tmp_path / "release.txt"
    release_path.write_text(
        "Traffic improved and sweet potato menu items sold well.",
        encoding="utf-8",
    )
    records = []
    documents_by_period = {}
    for symbol in ["WMT", "COST"]:
        transcript_path = tmp_path / f"2025_Q2_{symbol.lower()}_processed.txt"
        transcript_path.write_text(
            "Traffic improved and sweet potato menu items sold well.",
            encoding="utf-8",
        )
        record = TranscriptRecord(
            company_name=symbol,
            company_symbol=symbol,
            fiscal_year=2025,
            fiscal_quarter=2,
            path=transcript_path,
        )
        records.append(record)
        documents_by_period[(symbol, 2025, 2)] = [
            SourceDocument(
                source_id=f"{symbol}-2025-Q2-release",
                company_symbol=symbol,
                document_type="sec_ex_99_1_press_release",
                source_path=str(release_path),
                published_at=datetime(2025, 8, 21, tzinfo=UTC),
                content_hash="abc",
            )
        ]

    examples = build_synthetic_phrase_examples_from_transcript_records(
        records=records,
        documents_by_period=documents_by_period,
        target_phrases=["traffic"],
        company_target_phrases={"WMT": ["sweet potato"]},
    )

    targets_by_company = {
        symbol: sorted(
            {example.target_phrase for example in examples if example.company_symbol == symbol}
        )
        for symbol in ["WMT", "COST"]
    }
    assert targets_by_company["WMT"] == ["sweet potato", "traffic"]
    assert targets_by_company["COST"] == ["traffic"]


def test_build_synthetic_phrase_examples_supports_parallel_record_processing(tmp_path: Path):
    release_path = tmp_path / "release.txt"
    release_path.write_text("Traffic improved and automation helped.", encoding="utf-8")
    records = []
    documents_by_period = {}
    for symbol in ["WMT", "COST"]:
        transcript_path = tmp_path / f"2025_Q2_{symbol.lower()}_processed.txt"
        transcript_path.write_text("Traffic improved and automation helped.", encoding="utf-8")
        record = TranscriptRecord(
            company_name=symbol,
            company_symbol=symbol,
            fiscal_year=2025,
            fiscal_quarter=2,
            path=transcript_path,
        )
        records.append(record)
        documents_by_period[(symbol, 2025, 2)] = [
            SourceDocument(
                source_id=f"{symbol}-2025-Q2-release",
                company_symbol=symbol,
                document_type="sec_ex_99_1_press_release",
                source_path=str(release_path),
                published_at=datetime(2025, 8, 21, tzinfo=UTC),
                content_hash="abc",
            )
        ]

    examples = build_synthetic_phrase_examples_from_transcript_records(
        records=records,
        documents_by_period=documents_by_period,
        target_phrases=["traffic", "automation"],
        record_concurrency=2,
    )

    assert len(examples) == 4
    assert {example.company_symbol for example in examples} == {"COST", "WMT"}


def test_build_synthetic_phrase_examples_uses_only_strictly_prior_transcripts_for_recurrence(
    tmp_path: Path,
):
    records = []
    documents_by_period = {}
    call_starts = {
        1: datetime(2025, 4, 20, 16, 0, tzinfo=UTC),
        2: datetime(2025, 7, 20, 16, 0, tzinfo=UTC),
        3: datetime(2025, 10, 20, 16, 0, tzinfo=UTC),
        4: datetime(2026, 1, 20, 16, 0, tzinfo=UTC),
    }
    transcript_texts = {
        1: "Management discussed traffic growth.",
        2: "Management discussed traffic again.",
        3: "Management avoided the phrase.",
        4: "Management discussed traffic in the future.",
    }
    for quarter in [1, 2, 3, 4]:
        transcript_path = tmp_path / f"2025_Q{quarter}_wmt_processed.txt"
        transcript_path.write_text(transcript_texts[quarter], encoding="utf-8")
        release_path = tmp_path / f"release-q{quarter}.txt"
        release_path.write_text("Operations update.", encoding="utf-8")
        records.append(
            TranscriptRecord(
                company_name="Walmart",
                company_symbol="WMT",
                fiscal_year=2025,
                fiscal_quarter=quarter,
                path=transcript_path,
                call_start_at=call_starts[quarter],
                call_time_source="explicit",
            )
        )
        documents_by_period[("WMT", 2025, quarter)] = [
            SourceDocument(
                source_id=f"WMT-2025-Q{quarter}-release",
                company_symbol="WMT",
                document_type="sec_ex_99_1_press_release",
                source_path=str(release_path),
                published_at=call_starts[quarter],
                content_hash=f"release-{quarter}",
            )
        ]

    examples = build_synthetic_phrase_examples_from_transcript_records(
        records=records,
        documents_by_period=documents_by_period,
        target_phrases=["traffic"],
    )

    by_quarter = {example.fiscal_quarter: example for example in examples}
    assert by_quarter[1].features["prior_call_count"] == 0.0
    assert by_quarter[2].features["prior_call_count"] == 1.0
    assert by_quarter[2].features["prior_mention_count"] == 1.0
    assert by_quarter[3].features["prior_call_count"] == 2.0
    assert by_quarter[3].features["prior_mention_count"] == 2.0
    assert by_quarter[3].features["prior_mention_streak"] == 2.0
    assert by_quarter[4].features["prior_call_count"] == 3.0
    assert by_quarter[4].features["prior_mention_count"] == 2.0
    assert by_quarter[4].features["prior_recent_mention_binary"] == 0.0
    assert by_quarter[4].features["prior_mention_streak"] == 0.0


def test_default_synthetic_target_phrases_include_kalshi_style_terms():
    required = {"openai", "omnichannel", "salmon"}
    assert required.issubset(set(DEFAULT_SYNTHETIC_TARGET_PHRASES))
