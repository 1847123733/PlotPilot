"""LLM Provider factory tests."""
import pytest
from unittest.mock import sentinel, patch

from infrastructure.ai.providers.factory import build_llm_provider


def test_build_anthropic_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    with patch("infrastructure.ai.providers.factory.AnthropicProvider", return_value=sentinel.provider) as mock_cls:
        provider = build_llm_provider(require_key=False)
        assert provider is sentinel.provider
        mock_cls.assert_called_once()


def test_build_openapi_alias_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openapi")
    monkeypatch.setenv("OPENAPI_API_KEY", "test-key")

    with patch(
        "infrastructure.ai.providers.factory.OpenAICompatibleProvider",
        return_value=sentinel.provider,
    ) as mock_cls:
        provider = build_llm_provider(require_key=False)
        assert provider is sentinel.provider
        mock_cls.assert_called_once()


def test_build_mock_when_no_credentials(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    with patch("infrastructure.ai.providers.factory.MockProvider", return_value=sentinel.mock) as mock_cls:
        provider = build_llm_provider(require_key=False)
        assert provider is sentinel.mock
        mock_cls.assert_called_once()


def test_default_provider_falls_back_to_openai_compatible_when_only_ark_key(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ARK_API_KEY", "test-key")

    with patch(
        "infrastructure.ai.providers.factory.OpenAICompatibleProvider",
        return_value=sentinel.provider,
    ) as mock_cls:
        provider = build_llm_provider(require_key=False)
        assert provider is sentinel.provider
        mock_cls.assert_called_once()


def test_require_key_raises_when_missing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAPI_API_KEY", raising=False)
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="No LLM credentials found"):
        build_llm_provider(require_key=True)
