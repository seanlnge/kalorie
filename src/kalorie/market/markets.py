import re

from pydantic import BaseModel, ConfigDict, field_validator

from kalorie.domain.models import TargetPhrase


class MentionMarketParseError(ValueError):
    pass


class MentionMarket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_symbol: str
    target_phrase: TargetPhrase

    @field_validator("company_symbol")
    @classmethod
    def normalize_company_symbol(cls, value: str) -> str:
        return value.upper()


MENTION_TITLE_PATTERN = re.compile(
    r'^Will\s+(?P<symbol>[A-Z]+)\s+mention\s+(?P<phrase>".+?"|.+?)'
    r"(?:\s+during earnings|\s+on its earnings call)?\?$",
    re.IGNORECASE,
)
RULE_TARGET_PATTERN = re.compile(
    r"\bIf\s+(?P<phrase>.+?)\s+is\s+said\s+by\b",
    re.IGNORECASE | re.DOTALL,
)


def parse_mention_market_title(title: str) -> MentionMarket:
    match = MENTION_TITLE_PATTERN.match(title.strip())
    if not match:
        raise MentionMarketParseError(f"Unsupported mention market title: {title}")

    phrase = match.group("phrase").strip().strip('"').strip()
    normalized_phrase = normalize_phrase(phrase)
    return MentionMarket(
        company_symbol=match.group("symbol"),
        target_phrase=TargetPhrase(
            phrase=phrase,
            normalized_phrase=normalized_phrase,
            aliases=[],
        ),
    )


def parse_kalshi_rules_target_phrase(rules_text: str) -> TargetPhrase:
    match = RULE_TARGET_PATTERN.search(re.sub(r"\s+", " ", rules_text).strip())
    if not match:
        raise MentionMarketParseError("Could not extract target phrase from Kalshi rules")
    phrase = match.group("phrase").strip().strip('"').strip()
    normalized_phrase = normalize_phrase(phrase)
    return TargetPhrase(phrase=phrase, normalized_phrase=normalized_phrase, aliases=[])


def parse_mention_market_target(title: str, rules_text: str) -> TargetPhrase:
    if rules_text.strip():
        try:
            return parse_kalshi_rules_target_phrase(rules_text)
        except MentionMarketParseError:
            pass
    return parse_mention_market_title(title).target_phrase


def normalize_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()
