"""LLM 配置 API（支持前端读取与保存）。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from application.paths import AITEXT_ROOT

router = APIRouter(prefix="/api/v1/system", tags=["system-config"])

_OPENAI_COMPAT_ALIASES = {"openai", "openapi", "openai_compatible", "ark", "deepseek", "doubao"}


def _provider_name(raw: Optional[str]) -> str:
    v = (raw or "anthropic").strip().lower()
    if v in {"anthropic", "claude"}:
        return "anthropic"
    if v in _OPENAI_COMPAT_ALIASES:
        return "openai_compatible"
    return v


def _mask_secret(value: Optional[str]) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}***{value[-4:]}"


def _clean(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = v.strip()
    return s or None


def _env_key_group(raw_provider: str) -> Dict[str, str]:
    p = raw_provider.strip().lower()
    if p in {"ark", "deepseek", "doubao"}:
        return {
            "api_key": "ARK_API_KEY",
            "base_url": "ARK_BASE_URL",
            "model": "ARK_MODEL",
            "timeout": "ARK_TIMEOUT",
        }
    if p == "openai":
        return {
            "api_key": "OPENAI_API_KEY",
            "base_url": "OPENAI_BASE_URL",
            "model": "OPENAI_MODEL",
            "timeout": "OPENAI_TIMEOUT",
        }
    if p == "openapi":
        return {
            "api_key": "OPENAPI_API_KEY",
            "base_url": "OPENAPI_BASE_URL",
            "model": "OPENAPI_MODEL",
            "timeout": "OPENAPI_TIMEOUT",
        }
    return {
        "api_key": "OPENAPI_API_KEY",
        "base_url": "OPENAPI_BASE_URL",
        "model": "OPENAPI_MODEL",
        "timeout": "OPENAPI_TIMEOUT",
    }


def _read_llm_config_from_env() -> Dict[str, Optional[str]]:
    raw_provider = _clean(os.getenv("LLM_PROVIDER")) or "anthropic"
    provider_kind = _provider_name(raw_provider)

    if provider_kind == "anthropic":
        api_key = _clean(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
        base_url = _clean(os.getenv("ANTHROPIC_BASE_URL"))
        model = _clean(os.getenv("ANTHROPIC_MODEL"))
        timeout = _clean(os.getenv("ANTHROPIC_TIMEOUT"))
    else:
        keys = _env_key_group(raw_provider)
        fallback_api_key = _clean(os.getenv("OPENAI_API_KEY") or os.getenv("OPENAPI_API_KEY") or os.getenv("ARK_API_KEY"))
        fallback_base_url = _clean(os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAPI_BASE_URL") or os.getenv("ARK_BASE_URL"))
        fallback_model = _clean(os.getenv("OPENAI_MODEL") or os.getenv("OPENAPI_MODEL") or os.getenv("ARK_MODEL"))
        fallback_timeout = _clean(os.getenv("OPENAI_TIMEOUT") or os.getenv("OPENAPI_TIMEOUT") or os.getenv("ARK_TIMEOUT"))
        api_key = _clean(os.getenv(keys["api_key"])) or fallback_api_key
        base_url = _clean(os.getenv(keys["base_url"])) or fallback_base_url
        model = _clean(os.getenv(keys["model"])) or fallback_model
        timeout = _clean(os.getenv(keys["timeout"])) or fallback_timeout

    timeout_value: Optional[float] = None
    if timeout:
        try:
            timeout_value = float(timeout)
        except ValueError:
            timeout_value = None

    return {
        "provider": raw_provider,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "timeout": timeout_value,
    }


def _provider_for_ui(raw_provider: str) -> str:
    p = (raw_provider or "anthropic").strip().lower()
    if p in {"anthropic", "claude"}:
        return "anthropic"
    if p in {"openai", "openapi", "ark", "deepseek"}:
        return p
    if p in {"doubao"}:
        return "ark"
    if _provider_name(p) == "openai_compatible":
        return "openapi"
    return "anthropic"


def _update_env_file(env_updates: Dict[str, Optional[str]]) -> None:
    env_path = Path(AITEXT_ROOT) / ".env"
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)

    remaining = dict(env_updates)
    output_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in line:
            output_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key not in remaining:
            output_lines.append(line)
            continue
        value = remaining.pop(key)
        if value is not None and value != "":
            output_lines.append(f"{key}={value}\n")

    for key, value in remaining.items():
        if value is not None and value != "":
            output_lines.append(f"{key}={value}\n")

    env_path.write_text("".join(output_lines), encoding="utf-8")


def _apply_env_updates(env_updates: Dict[str, Optional[str]]) -> None:
    for key, value in env_updates.items():
        if value is None or value == "":
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)


class LLMConfigResponse(BaseModel):
    provider: str
    has_api_key: bool
    api_key_masked: str = ""
    base_url: Optional[str] = None
    model: Optional[str] = None
    timeout: Optional[float] = None


class LLMConfigUpdateRequest(BaseModel):
    provider: str = Field(..., description="anthropic/openapi/openai/ark/deepseek")
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    timeout: Optional[float] = Field(default=None, gt=0)
    persist_to_env: bool = True


@router.get("/llm-config", response_model=LLMConfigResponse)
async def get_llm_config() -> LLMConfigResponse:
    cfg = _read_llm_config_from_env()
    api_key = cfg.get("api_key")
    return LLMConfigResponse(
        provider=_provider_for_ui(str(cfg["provider"])),
        has_api_key=bool(api_key),
        api_key_masked=_mask_secret(api_key),
        base_url=cfg.get("base_url"),
        model=cfg.get("model"),
        timeout=cfg.get("timeout"),
    )


@router.put("/llm-config", response_model=LLMConfigResponse)
async def update_llm_config(request: LLMConfigUpdateRequest) -> LLMConfigResponse:
    raw_provider = (request.provider or "").strip().lower()
    if raw_provider not in {"anthropic", "claude", "openai", "openapi", "openai_compatible", "ark", "deepseek", "doubao"}:
        raise HTTPException(status_code=400, detail="Unsupported provider")

    provider_kind = _provider_name(raw_provider)
    provider_to_save = "anthropic" if provider_kind == "anthropic" else raw_provider

    env_updates: Dict[str, Optional[str]] = {"LLM_PROVIDER": provider_to_save}

    # 先清理所有候选键，避免多源冲突
    for key in [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_TIMEOUT",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "OPENAI_TIMEOUT",
        "OPENAPI_API_KEY",
        "OPENAPI_BASE_URL",
        "OPENAPI_MODEL",
        "OPENAPI_TIMEOUT",
        "ARK_API_KEY",
        "ARK_BASE_URL",
        "ARK_MODEL",
        "ARK_TIMEOUT",
    ]:
        env_updates[key] = None

    if provider_kind == "anthropic":
        env_updates["ANTHROPIC_API_KEY"] = _clean(request.api_key)
        env_updates["ANTHROPIC_BASE_URL"] = _clean(request.base_url)
        env_updates["ANTHROPIC_MODEL"] = _clean(request.model)
        env_updates["ANTHROPIC_TIMEOUT"] = str(request.timeout) if request.timeout else None
    else:
        key_group = _env_key_group(provider_to_save)
        env_updates[key_group["api_key"]] = _clean(request.api_key)
        env_updates[key_group["base_url"]] = _clean(request.base_url)
        env_updates[key_group["model"]] = _clean(request.model)
        env_updates[key_group["timeout"]] = str(request.timeout) if request.timeout else None

    _apply_env_updates(env_updates)
    if request.persist_to_env:
        _update_env_file(env_updates)

    cfg = _read_llm_config_from_env()
    api_key = cfg.get("api_key")
    return LLMConfigResponse(
        provider=_provider_for_ui(str(cfg["provider"])),
        has_api_key=bool(api_key),
        api_key_masked=_mask_secret(api_key),
        base_url=cfg.get("base_url"),
        model=cfg.get("model"),
        timeout=cfg.get("timeout"),
    )
