from kalorie.market.markets import normalize_phrase


def normalize_and_dedupe_phrases(phrases: list[str]) -> list[str]:
    normalized = [normalize_phrase(phrase) for phrase in phrases if phrase.strip()]
    return list(dict.fromkeys(normalized))
