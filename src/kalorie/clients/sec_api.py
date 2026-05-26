import re
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict


class SecApiError(RuntimeError):
    pass


class SecApiAuthError(SecApiError):
    pass


class SecApiRateLimitError(SecApiError):
    pass


class SecApiExhibit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: str
    description: str | None = None
    document_url: str


class SecApiFiling(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    cik: str
    filed_at: datetime
    exhibit_url: str
    exhibits: list[SecApiExhibit]
    form_type: str = "8-K"


class SecCompanyMapping(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    ticker: str
    cik: str
    is_delisted: bool = False
    category: str = ""


class SecApiClient:
    def __init__(
        self,
        api_key: str,
        http_client: httpx.Client,
        base_url: str = "https://api.sec-api.io",
    ) -> None:
        self._api_key = api_key
        self._http = http_client
        self._base_url = base_url.rstrip("/")

    def query_ex99_1_filings(
        self,
        *,
        query: str,
        start: int = 0,
        size: int = 50,
    ) -> list[SecApiFiling]:
        response = self._http.post(
            self._base_url,
            params={"token": self._api_key},
            json={
                "query": query,
                "from": str(start),
                "size": str(size),
                "sort": [{"filedAt": {"order": "desc"}}],
            },
        )
        if response.status_code == 401:
            raise SecApiAuthError("SEC API authentication failed")
        if response.status_code == 429:
            raise SecApiRateLimitError("SEC API rate limit exceeded")
        response.raise_for_status()
        payload = response.json()
        filings = []
        for row in payload.get("filings", []):
            try:
                filings.append(_parse_filing(row))
            except SecApiError:
                continue
        return filings

    def resolve_mapping(self, *, resolve_by: str, value: str) -> list[SecCompanyMapping]:
        response = self._http.get(
            f"{self._base_url}/mapping/{resolve_by}/{value}",
            params={"token": self._api_key},
        )
        if response.status_code == 401:
            raise SecApiAuthError("SEC API authentication failed")
        if response.status_code == 429:
            raise SecApiRateLimitError("SEC API rate limit exceeded")
        response.raise_for_status()
        return [SecCompanyMapping.model_validate(row) for row in response.json()]


def select_best_company_mapping(mappings: list[SecCompanyMapping]) -> SecCompanyMapping:
    if not mappings:
        raise SecApiError("SEC API mapping returned no candidates")
    active = [mapping for mapping in mappings if not mapping.is_delisted]
    candidates = active or mappings
    for mapping in candidates:
        if "common stock" in mapping.category.lower():
            return mapping
    return candidates[0]


def _parse_filing(row: dict[str, Any]) -> SecApiFiling:
    exhibits = _find_exhibits(row.get("documentFormatFiles", []))
    return SecApiFiling(
        ticker=str(row.get("ticker") or row.get("tickerSymbol") or ""),
        cik=str(row.get("cik") or ""),
        filed_at=datetime.fromisoformat(str(row["filedAt"]).replace("Z", "+00:00")),
        exhibit_url=exhibits[0].document_url,
        exhibits=exhibits,
        form_type=str(row.get("formType") or "8-K"),
    )


def _find_exhibits(files: list[dict[str, Any]]) -> list[SecApiExhibit]:
    exhibits = []
    for document in files:
        document_type = str(document.get("type", "")).upper()
        description = str(document.get("description") or "")
        document_url = str(document.get("documentUrl") or "")
        if _is_ex99_supplement(document_type, description, document_url):
            exhibits.append(
                SecApiExhibit(
                    document_type=document_type,
                    description=description or None,
                    document_url=document_url,
                )
            )
    if not exhibits:
        raise SecApiError("filing did not include EX-99 supplemental exhibits")
    return exhibits


def _is_ex99_supplement(document_type: str, description: str, document_url: str) -> bool:
    haystack = " ".join([document_type, description, document_url]).upper()
    return re.search(r"\bEX-99(?:\.\d+)?\b", haystack) is not None
