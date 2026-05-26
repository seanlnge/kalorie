import pytest

from kalorie.market.markets import MentionMarketParseError, parse_mention_market_title


@pytest.mark.parametrize(
    ("title", "phrase"),
    [
        ("Will CAVA mention traffic during earnings?", "traffic"),
        (
            'Will CAVA mention "same restaurant sales" on its earnings call?',
            "same restaurant sales",
        ),
        ("Will CAVA mention geopolitical uncertainty?", "geopolitical uncertainty"),
    ],
)
def test_parse_cava_mention_market_titles(title: str, phrase: str):
    parsed = parse_mention_market_title(title)

    assert parsed.company_symbol == "CAVA"
    assert parsed.target_phrase.phrase == phrase
    assert parsed.target_phrase.normalized_phrase == phrase.lower()


def test_unsupported_title_returns_structured_parse_error():
    with pytest.raises(MentionMarketParseError, match="Unsupported mention market title"):
        parse_mention_market_title("CAVA traffic over or under 10 mentions?")
