import re
from collections import Counter

from kalorie.data_cleaning import normalize_and_dedupe_phrases

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+]*")
STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "affected",
    "again",
    "against",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "call",
    "can",
    "company",
    "conference",
    "could",
    "details",
    "discussed",
    "during",
    "earnings",
    "for",
    "forward",
    "from",
    "had",
    "has",
    "have",
    "host",
    "in",
    "including",
    "investor",
    "is",
    "it",
    "its",
    "looking",
    "may",
    "more",
    "next",
    "not",
    "of",
    "on",
    "or",
    "our",
    "please",
    "relations",
    "remained",
    "see",
    "statements",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "we",
    "were",
    "will",
    "with",
}
BOILERPLATE_PHRASES = {
    "conference call",
    "forward looking",
    "forward looking statements",
    "investor relations",
}
ALLOWED_SHORT_TOKENS = {"ai"}


def generate_synthetic_phrase_candidates(
    texts: list[str],
    *,
    seed_phrases: list[str] | None = None,
    max_candidates: int = 200,
    max_ngram: int = 2,
) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        tokens = [
            token.lower()
            for token in TOKEN_PATTERN.findall(text)
            if _is_candidate_token(token.lower())
        ]
        for size in range(1, max_ngram + 1):
            for index in range(0, len(tokens) - size + 1):
                phrase_tokens = tokens[index : index + size]
                if not _is_candidate_phrase(phrase_tokens):
                    continue
                counts[" ".join(phrase_tokens)] += 1

    seeded = normalize_and_dedupe_phrases(seed_phrases or [])
    ranked = sorted(
        counts,
        key=lambda phrase: (
            phrase not in seeded,
            -counts[phrase],
            abs(len(phrase.split()) - 2),
            phrase,
        ),
    )
    return normalize_and_dedupe_phrases([*seeded, *ranked])[:max_candidates]


def _is_candidate_token(token: str) -> bool:
    if len(token) < 3 and token not in ALLOWED_SHORT_TOKENS:
        return False
    if token in STOPWORDS:
        return False
    return any(character.isalpha() for character in token)


def _is_candidate_phrase(tokens: list[str]) -> bool:
    if not tokens:
        return False
    phrase = " ".join(tokens)
    if phrase in BOILERPLATE_PHRASES:
        return False
    if all(token in STOPWORDS for token in tokens):
        return False
    return True
