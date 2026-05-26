from pathlib import Path

import pytest

from kalorie.domain.config import Settings


def test_model_dump_and_redacted_dict_do_not_expose_secret_values(tmp_path: Path):
    private_key_path = tmp_path / "placeholder.key"
    private_key_path.write_text("", encoding="utf-8")

    settings = Settings(
        api_ninjas_api_key="api-secret",
        financial_modeling_prep_api_key="fmp-secret",
        newsdata_api_key="news-secret",
        tiingo_api_key="tiingo-secret",
        defeat_beta_api_key="defeatbeta-secret",
        sec_api_key="sec-secret",
        kalshi_api_key_id="kalshi-secret",
        kalshi_private_key_path=private_key_path,
        openai_api_key="openai-secret",
    )

    assert "api-secret" not in str(settings.model_dump())
    assert "fmp-secret" not in str(settings.model_dump())
    assert "news-secret" not in str(settings.model_dump())
    assert "tiingo-secret" not in str(settings.model_dump())
    assert "defeatbeta-secret" not in str(settings.model_dump())
    assert "sec-secret" not in str(settings.model_dump())
    assert "kalshi-secret" not in str(settings.model_dump())
    assert "openai-secret" not in str(settings.model_dump())
    assert settings.redacted_dict() == {
        "api_ninjas_configured": True,
        "financial_modeling_prep_configured": True,
        "newsdata_configured": True,
        "tiingo_configured": True,
        "defeat_beta_configured": True,
        "sec_api_configured": True,
        "kalshi_api_key_id_configured": True,
        "kalshi_private_key_path_configured": True,
        "openai_configured": True,
        "openai_embedding_model": "text-embedding-3-small",
        "openai_embedding_dimensions": None,
        "embedding_batch_size": 100,
        "embedding_cache_path": ".kalorie-cache/embeddings.json",
        "kalshi_base_url": "https://api.elections.kalshi.com/trade-api/v2",
        "mode": "local",
    }


def test_missing_vendor_credentials_are_allowed_for_local_and_public_modes(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SEC_API_KEY", raising=False)
    settings = Settings()

    settings.validate_for_mode("local")
    settings.validate_for_mode("kalshi_public")
    assert settings.redacted_dict()["openai_configured"] is False
    assert settings.redacted_dict()["sec_api_configured"] is False
    assert settings.redacted_dict()["financial_modeling_prep_configured"] is False
    assert settings.redacted_dict()["newsdata_configured"] is False
    assert settings.redacted_dict()["tiingo_configured"] is False
    assert settings.redacted_dict()["defeat_beta_configured"] is False


def test_embedding_settings_are_validated():
    with pytest.raises(ValueError, match="embedding_batch_size"):
        Settings(embedding_batch_size=0)

    with pytest.raises(ValueError, match="openai_embedding_dimensions"):
        Settings(openai_embedding_dimensions=0)


def test_authorized_kalshi_mode_requires_key_id_and_existing_path(tmp_path: Path):
    private_key_path = tmp_path / "kalshi.key"
    private_key_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="KALSHI_API_KEY_ID"):
        Settings(kalshi_private_key_path=private_key_path).validate_for_mode("kalshi_authorized")

    with pytest.raises(ValueError, match="KALSHI_PRIVATE_KEY_PATH"):
        Settings(kalshi_api_key_id="key-id").validate_for_mode("kalshi_authorized")

    with pytest.raises(ValueError, match="does not exist"):
        Settings(
            kalshi_api_key_id="key-id",
            kalshi_private_key_path=tmp_path / "missing.key",
        ).validate_for_mode("kalshi_authorized")

    Settings(
        kalshi_api_key_id="key-id",
        kalshi_private_key_path=private_key_path,
    ).validate_for_mode("kalshi_authorized")


def test_authorized_validation_checks_presence_without_reading_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    private_key_path = tmp_path / "kalshi.key"
    private_key_path.write_text("", encoding="utf-8")

    def fail_on_read(*args, **kwargs):
        raise AssertionError("private key content must not be read")

    monkeypatch.setattr(Path, "read_text", fail_on_read)
    monkeypatch.setattr(Path, "read_bytes", fail_on_read)

    Settings(
        kalshi_api_key_id="key-id",
        kalshi_private_key_path=private_key_path,
    ).validate_for_mode("kalshi_authorized")
