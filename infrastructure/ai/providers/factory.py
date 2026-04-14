"""LLM provider 工厂：根据环境变量自动选择供应商。"""
import os
from typing import Optional

from domain.ai.services.llm_service import LLMService
from infrastructure.ai.config.settings import Settings
from .anthropic_provider import AnthropicProvider
from .mock_provider import MockProvider
from .openai_compatible_provider import OpenAICompatibleProvider


def _clean_env(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return None


def _clean_float_env(*keys: str, default: float) -> float:
    raw = _clean_env(*keys)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _provider_name() -> str:
    raw = (_clean_env("LLM_PROVIDER") or "anthropic").lower()
    aliases = {
        "anthropic": "anthropic",
        "claude": "anthropic",
        "openai": "openai_compatible",
        "openapi": "openai_compatible",
        "openai_compatible": "openai_compatible",
        "ark": "openai_compatible",
        "deepseek": "openai_compatible",
        "doubao": "openai_compatible",
    }
    return aliases.get(raw, raw)


def _anthropic_settings() -> Optional[Settings]:
    api_key = _clean_env("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        return None
    return Settings(
        provider="anthropic",
        api_key=api_key,
        base_url=_clean_env("ANTHROPIC_BASE_URL"),
        default_model=_clean_env("ANTHROPIC_MODEL") or "claude-sonnet-4-6",
        request_timeout=_clean_float_env("ANTHROPIC_TIMEOUT", default=300.0),
    )


def _openai_compatible_settings() -> Optional[Settings]:
    api_key = _clean_env("OPENAI_API_KEY", "OPENAPI_API_KEY", "ARK_API_KEY")
    if not api_key:
        return None
    return Settings(
        provider="openai_compatible",
        api_key=api_key,
        base_url=_clean_env("OPENAI_BASE_URL", "OPENAPI_BASE_URL", "ARK_BASE_URL"),
        default_model=_clean_env("OPENAI_MODEL", "OPENAPI_MODEL", "ARK_MODEL") or "gpt-4o-mini",
        request_timeout=_clean_float_env("OPENAI_TIMEOUT", "OPENAPI_TIMEOUT", "ARK_TIMEOUT", default=300.0),
    )


def build_llm_provider(require_key: bool = False) -> LLMService:
    explicit_provider = _clean_env("LLM_PROVIDER")
    provider_name = _provider_name()

    if provider_name == "openai_compatible":
        settings = _openai_compatible_settings()
        if settings:
            return OpenAICompatibleProvider(settings)
    else:
        settings = _anthropic_settings()
        if settings:
            return AnthropicProvider(settings)
        # 兼容旧配置：未显式设置 LLM_PROVIDER 时，如果只配了 OPENAI/ARK，也可自动启用。
        if not explicit_provider:
            openai_settings = _openai_compatible_settings()
            if openai_settings:
                return OpenAICompatibleProvider(openai_settings)

    if require_key:
        raise ValueError(
            "No LLM credentials found. Configure ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN "
            "or OPENAI_API_KEY/OPENAPI_API_KEY/ARK_API_KEY."
        )
    return MockProvider()
