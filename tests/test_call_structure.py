from datetime import UTC, datetime

import pytest

from kalorie2.call_structure import (
    CallStructureRecord,
    extract_call_structure,
    summarize_prior_call_structure,
)


def test_extract_call_structure_reads_duration_prepared_remarks_and_questions():
    text = """
    Call duration: 62 minutes
    Prepared remarks: 28 minutes

    Question-and-Answer Session
    John Smith - Example Bank: How should we think about AI demand next quarter?
    Jane Doe - Example Capital: Can you discuss margin pressure?
    Operator: This concludes today's call.
    """

    structure = extract_call_structure(text)

    assert structure.call_duration_minutes == 62.0
    assert structure.prepared_remarks_minutes == 28.0
    assert structure.qa_question_count == 2
    assert structure.qa_share == pytest.approx((62.0 - 28.0) / 62.0)


def test_summarize_prior_call_structure_uses_only_records_available_before_cutoff():
    cutoff = datetime(2026, 5, 21, 7, tzinfo=UTC)
    records = [
        CallStructureRecord(
            available_at=datetime(2025, 5, 21, tzinfo=UTC),
            call_duration_minutes=50.0,
            qa_question_count=8,
            prepared_remarks_minutes=20.0,
        ),
        CallStructureRecord(
            available_at=datetime(2026, 2, 21, tzinfo=UTC),
            call_duration_minutes=70.0,
            qa_question_count=12,
            prepared_remarks_minutes=28.0,
        ),
        CallStructureRecord(
            available_at=datetime(2026, 6, 21, tzinfo=UTC),
            call_duration_minutes=90.0,
            qa_question_count=20,
            prepared_remarks_minutes=30.0,
        ),
    ]

    features = summarize_prior_call_structure(records, cutoff_time=cutoff)

    assert features["company_prior_call_count"] == 2.0
    assert features["company_avg_call_duration_minutes_prior"] == 60.0
    assert features["company_avg_qa_question_count_prior"] == 10.0
    assert features["company_avg_prepared_remarks_minutes_prior"] == 24.0
    assert features["company_qa_share_prior"] == pytest.approx(0.6)
    assert features["company_question_count_trend_prior"] == 2.0

