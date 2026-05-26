import re
from typing import Literal

from kalorie.domain.models import DocumentChunk, MatchSpan, MentionLabel, TargetPhrase

QUOTE_TRANSLATION = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
)
ANALYST_CONTEXT_PATTERN = re.compile(
    r"(?:from the line of|analyst|question(?:\s+from)?|q:)",
    re.IGNORECASE,
)


def normalize_phrase(text: str) -> str:
    normalized = text.translate(QUOTE_TRANSLATION)
    return re.sub(r"\s+", " ", normalized).strip().lower()


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    normalized = normalize_phrase(phrase)
    escaped = re.escape(normalized).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def _kalshi_settlement_pattern(phrase: str) -> re.Pattern[str]:
    normalized = normalize_phrase(phrase)
    words = normalized.split()
    if not words:
        raise ValueError("phrase must not be empty")

    last_word = words[-1]
    plural = _pluralize(last_word)
    last_word_forms = {last_word, f"{last_word}'s", f"{last_word}s'"}
    if plural != last_word:
        last_word_forms.add(plural)
        last_word_forms.add(f"{plural}'")
        last_word_forms.add(f"{plural}'s")

    escaped_prefix = [re.escape(word) for word in words[:-1]]
    escaped_last = "|".join(
        re.escape(form) for form in sorted(last_word_forms, key=len, reverse=True)
    )
    escaped_terms = [*escaped_prefix, f"(?:{escaped_last})"]
    pattern_body = r"\s+".join(escaped_terms)
    return re.compile(rf"(?<!\w){pattern_body}(?!\w)", re.IGNORECASE)


def _pluralize(word: str) -> str:
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return f"{word[:-1]}ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return f"{word}es"
    return f"{word}s"


def find_exact_mentions(text: str, phrase: str) -> list[MatchSpan]:
    normalized_text = text.translate(QUOTE_TRANSLATION)
    return [
        MatchSpan(start=match.start(), end=match.end(), text=match.group(0), match_type="exact")
        for match in _phrase_pattern(phrase).finditer(normalized_text)
    ]


def find_kalshi_settlement_mentions(text: str, phrase: str) -> list[MatchSpan]:
    normalized_text = text.translate(QUOTE_TRANSLATION)
    return [
        MatchSpan(start=match.start(), end=match.end(), text=match.group(0), match_type="exact")
        for match in _kalshi_settlement_pattern(phrase).finditer(normalized_text)
    ]


def _filter_company_entity_mentions(text: str, matches: list[MatchSpan]) -> list[MatchSpan]:
    kept: list[MatchSpan] = []
    for span in matches:
        line_start = text.rfind("\n", 0, span.start) + 1
        line_end = text.find("\n", span.end)
        if line_end == -1:
            line_end = len(text)
        context = text[line_start:line_end]
        if ANALYST_CONTEXT_PATTERN.search(context):
            continue
        kept.append(span)
    return kept


def find_lexical_mentions(text: str, target: TargetPhrase) -> list[MatchSpan]:
    matches: list[MatchSpan] = []
    for alias in target.aliases:
        for match in _phrase_pattern(alias).finditer(text.translate(QUOTE_TRANSLATION)):
            matches.append(
                MatchSpan(
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                    match_type="lexical",
                )
            )
    return matches


def label_document_chunks(
    chunks: list[DocumentChunk],
    targets: list[TargetPhrase],
    *,
    entity_scope: Literal["all", "company_employee"] = "all",
) -> list[MentionLabel]:
    document_text = "\n".join(chunk.text for chunk in chunks)
    labels: list[MentionLabel] = []
    for target in targets:
        exact_matches = find_kalshi_settlement_mentions(document_text, target.normalized_phrase)
        if entity_scope == "company_employee":
            exact_matches = _filter_company_entity_mentions(document_text, exact_matches)
        lexical_matches = find_lexical_mentions(document_text, target)
        labels.append(
            MentionLabel(
                target_phrase=target.normalized_phrase,
                exact_mentioned=bool(exact_matches),
                lexical_mentioned=bool(lexical_matches),
                match_spans=[*exact_matches, *lexical_matches],
            )
        )
    return labels
