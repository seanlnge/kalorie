from pathlib import Path

from kalorie2.company_metadata import company_metadata_features, load_company_metadata


def test_load_company_metadata_reads_seeded_local_artifact():
    registry = load_company_metadata(Path("data/company_metadata.json"))

    apple = registry["AAPL"]

    assert apple.symbol == "AAPL"
    assert apple.sector == "technology"
    assert apple.market_cap_bucket == "mega"
    assert apple.is_sp500
    assert apple.provenance.source_model


def test_company_metadata_features_are_numeric_and_missing_safe():
    registry = load_company_metadata(Path("data/company_metadata.json"))

    features = company_metadata_features("KXEARNINGSMENTIONAAPL", registry)
    missing = company_metadata_features("KXEARNINGSMENTIONUNKNOWN", registry)

    assert features["company_metadata_available"] == 1.0
    assert features["company_sector_technology"] == 1.0
    assert features["company_market_cap_mega"] == 1.0
    assert features["company_business_model_consumer"] == 1.0
    assert features["company_is_sp500"] == 1.0
    assert missing["company_metadata_available"] == 0.0
    assert missing["company_sector_technology"] == 0.0

