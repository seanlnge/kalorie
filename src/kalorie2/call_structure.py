import re
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CallStructure:
    call_duration_minutes: float = 0.0
    qa_question_count: int = 0
    prepared_remarks_minutes: float = 0.0

    @property
    def qa_share(self) -> float:
        if self.call_duration_minutes <= 0.0:
            return 0.0
        qa_minutes = max(0.0, self.call_duration_minutes - self.prepared_remarks_minutes)
        return qa_minutes / self.call_duration_minutes


@dataclass(frozen=True)
class CallStructureRecord:
    available_at: datetime
    call_duration_minutes: float = 0.0
    qa_question_count: int = 0
    prepared_remarks_minutes: float = 0.0

    @property
    def qa_share(self) -> float:
        if self.call_duration_minutes <= 0.0:
            return 0.0
        qa_minutes = max(0.0, self.call_duration_minutes - self.prepared_remarks_minutes)
        return qa_minutes / self.call_duration_minutes


def extract_call_structure(text: str) -> CallStructure:
    duration = _extract_minutes(text, r"\bcall\s+duration\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*minutes?\b")
    prepared = _extract_minutes(
        text,
        r"\bprepared\s+remarks\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*minutes?\b",
    )
    return CallStructure(
        call_duration_minutes=duration,
        prepared_remarks_minutes=prepared,
        qa_question_count=_count_qa_questions(text),
    )


def summarize_prior_call_structure(
    records: list[CallStructureRecord],
    *,
    cutoff_time: datetime,
) -> dict[str, float]:
    prior = sorted(
        (record for record in records if record.available_at < cutoff_time),
        key=lambda record: record.available_at,
    )
    if not prior:
        return _empty_features()

    question_counts = [float(record.qa_question_count) for record in prior]
    recent_question_count = question_counts[-1]
    average_question_count = _mean(question_counts)
    return {
        "company_prior_call_count": float(len(prior)),
        "company_avg_call_duration_minutes_prior": _mean(
            [record.call_duration_minutes for record in prior]
        ),
        "company_avg_qa_question_count_prior": average_question_count,
        "company_avg_prepared_remarks_minutes_prior": _mean(
            [record.prepared_remarks_minutes for record in prior]
        ),
        "company_qa_share_prior": _mean([record.qa_share for record in prior]),
        "company_question_count_trend_prior": recent_question_count - average_question_count,
    }


def _empty_features() -> dict[str, float]:
    return {
        "company_prior_call_count": 0.0,
        "company_avg_call_duration_minutes_prior": 0.0,
        "company_avg_qa_question_count_prior": 0.0,
        "company_avg_prepared_remarks_minutes_prior": 0.0,
        "company_qa_share_prior": 0.0,
        "company_question_count_trend_prior": 0.0,
    }


def _extract_minutes(text: str, pattern: str) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else 0.0


def _count_qa_questions(text: str) -> int:
    qa_section = _qa_section(text)
    count = 0
    for line in qa_section.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("operator:"):
            continue
        if "?" in stripped:
            count += 1
    return count


def _qa_section(text: str) -> str:
    match = re.search(
        r"(question\s*(?:-|and-|and\s+)?answer|q\s*&\s*a)",
        text,
        flags=re.IGNORECASE,
    )
    return text[match.start() :] if match else text


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

