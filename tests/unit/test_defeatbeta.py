import pandas as pd
import pytest

from kalorie.clients.defeatbeta import (
    DefeatBetaApiError,
    DefeatBetaNewsClient,
)


def test_search_stock_news_reads_filtered_parquet_and_parses_news(monkeypatch):
    seen = {}

    def fake_read_parquet(url, columns, filters):
        seen["url"] = url
        seen["columns"] = columns
        seen["filters"] = filters
        return pd.DataFrame(
            [
                {
                    "uuid": "db-1",
                    "report_date": "2024-05-10",
                    "title": "Walmart pre-earnings expectations rise",
                    "publisher": "Reuters",
                    "type": "STORY",
                    "link": "https://example.com/wmt-earnings",
                    "related_symbols": "WMT",
                    "news": [{"paragraph_number": 1, "paragraph": "First paragraph text."}],
                }
            ]
        )

    monkeypatch.setattr("kalorie.clients.defeatbeta.pd.read_parquet", fake_read_parquet)

    client = DefeatBetaNewsClient(dataset_url="https://example.com/stock_news.parquet")
    rows = client.search_stock_news(
        symbol="WMT",
        start_date="2024-05-01",
        end_date="2024-05-14",
        max_rows=10,
    )

    assert seen["url"] == "https://example.com/stock_news.parquet"
    assert ("related_symbols", "==", "WMT") in seen["filters"]
    assert ("report_date", ">=", "2024-05-01") in seen["filters"]
    assert len(rows) == 1
    assert rows[0].article_id == "db-1"
    assert rows[0].content == "First paragraph text."


def test_search_stock_news_wraps_transport_errors():
    def fake_read_parquet(url, columns, filters):
        raise RuntimeError("broken parquet request")

    client = DefeatBetaNewsClient(dataset_url="https://example.com/stock_news.parquet")
    with pytest.raises(DefeatBetaApiError, match="parquet query failed"):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("kalorie.clients.defeatbeta.pd.read_parquet", fake_read_parquet)
            client.search_stock_news(
                symbol="WMT",
                start_date="2024-05-01",
                end_date="2024-05-14",
            )


def test_search_stock_news_normalizes_related_symbol_string(monkeypatch):
    def fake_read_parquet(url, columns, filters):
        return pd.DataFrame(
            [
                {
                    "uuid": "db-2",
                    "report_date": "2024-05-12",
                    "title": "Walmart guidance update",
                    "publisher": "MT Newswires",
                    "type": "STORY",
                    "link": "https://example.com/wmt-guidance",
                    "related_symbols": "WMT",
                    "news": [{"paragraph_number": 1, "paragraph": "Guidance paragraph."}],
                }
            ]
        )

    monkeypatch.setattr("kalorie.clients.defeatbeta.pd.read_parquet", fake_read_parquet)
    client = DefeatBetaNewsClient(dataset_url="https://example.com/stock_news.parquet")
    rows = client.search_stock_news(
        symbol="WMT",
        start_date="2024-05-01",
        end_date="2024-05-14",
        max_rows=10,
    )
    assert rows[0].tickers == ["WMT"]


def test_search_stock_news_raises_when_invalid_max_rows():
    client = DefeatBetaNewsClient(dataset_url="https://example.com/stock_news.parquet")
    with pytest.raises(ValueError):
        client.search_stock_news(
            symbol="WMT",
            start_date="2024-05-01",
            end_date="2024-05-14",
            max_rows=0,
        )
