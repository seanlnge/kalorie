import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_METADATA_PATH = Path(__file__).resolve().parents[2] / "data" / "company_metadata.json"

_SECTORS = (
    "technology",
    "communication_services",
    "consumer_cyclical",
    "consumer_defensive",
    "financials",
    "healthcare",
    "industrials",
    "energy",
    "utilities",
    "materials",
    "real_estate",
    "other",
)
_MARKET_CAP_BUCKETS = ("mega", "large", "mid", "small", "unknown")
_BUSINESS_MODELS = ("consumer", "enterprise", "mixed", "other")


class CompanyMetadataModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompanyMetadataProvenance(CompanyMetadataModel):
    source_model: str = Field(min_length=1)
    generated_at: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class CompanyMetadata(CompanyMetadataModel):
    symbol: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    sector: str = Field(min_length=1)
    industry: str = Field(min_length=1)
    market_cap_bucket: str = Field(min_length=1)
    business_model: str = Field(min_length=1)
    is_sp500: bool = False
    is_megacap: bool = False
    tags: list[str] = Field(default_factory=list)
    provenance: CompanyMetadataProvenance


def load_company_metadata(path: Path = DEFAULT_METADATA_PATH) -> dict[str, CompanyMetadata]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    companies = payload.get("companies", [])
    return {
        metadata.symbol.upper(): metadata
        for metadata in (CompanyMetadata.model_validate(company) for company in companies)
    }


@lru_cache(maxsize=1)
def default_company_metadata() -> dict[str, CompanyMetadata]:
    return load_company_metadata(DEFAULT_METADATA_PATH)


def default_company_metadata_features(series_ticker: str) -> dict[str, float]:
    return company_metadata_features(series_ticker, default_company_metadata())


def company_metadata_features(
    series_ticker: str,
    registry: dict[str, CompanyMetadata],
) -> dict[str, float]:
    symbol = _symbol_from_series(series_ticker)
    metadata = registry.get(symbol)
    features = _empty_features()
    if metadata is None:
        return features

    features["company_metadata_available"] = 1.0
    sector = _normalized_choice(metadata.sector, _SECTORS, "other")
    cap_bucket = _normalized_choice(
        metadata.market_cap_bucket,
        _MARKET_CAP_BUCKETS,
        "unknown",
    )
    business_model = _normalized_choice(metadata.business_model, _BUSINESS_MODELS, "other")
    features[f"company_sector_{sector}"] = 1.0
    features[f"company_market_cap_{cap_bucket}"] = 1.0
    features[f"company_business_model_{business_model}"] = 1.0
    features["company_is_sp500"] = 1.0 if metadata.is_sp500 else 0.0
    features["company_is_megacap"] = 1.0 if metadata.is_megacap else 0.0
    features["company_metadata_confidence"] = metadata.provenance.confidence
    return features


def _empty_features() -> dict[str, float]:
    features = {"company_metadata_available": 0.0}
    features.update({f"company_sector_{sector}": 0.0 for sector in _SECTORS})
    features.update({f"company_market_cap_{bucket}": 0.0 for bucket in _MARKET_CAP_BUCKETS})
    features.update({f"company_business_model_{model}": 0.0 for model in _BUSINESS_MODELS})
    features.update(
        {
            "company_is_sp500": 0.0,
            "company_is_megacap": 0.0,
            "company_metadata_confidence": 0.0,
        }
    )
    return features


def _symbol_from_series(series_ticker: str) -> str:
    cleaned = series_ticker.upper().strip()
    cleaned = cleaned.removeprefix("KXEARNINGSMENTION")
    return re.split(r"[^A-Z0-9]", cleaned, maxsplit=1)[0]


def _normalized_choice(value: str, choices: tuple[str, ...], fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized if normalized in choices else fallback

