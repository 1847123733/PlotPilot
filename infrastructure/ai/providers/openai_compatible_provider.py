"""OpenAI 兼容 LLM 提供商实现（支持 OpenAI/DeepSeek/Ark 等）。"""
import logging
from typing import AsyncIterator, List, Dict

from openai import AsyncOpenAI

from domain.ai.value_objects.prompt import Prompt
from domain.ai.value_objects.token_usage import TokenUsage
from domain.ai.services.llm_service import GenerationConfig, GenerationResult
from infrastructure.ai.config.settings import Settings
from .base import BaseProvider

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(BaseProvider):
    """OpenAI 兼容接口提供商。"""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        if not settings.api_key:
            raise ValueError("API key is required for OpenAICompatibleProvider")

        self.client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.request_timeout,
            max_retries=5,
        )

    def _messages(self, prompt: Prompt) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

    async def generate(self, prompt: Prompt, config: GenerationConfig) -> GenerationResult:
        try:
            response = await self.client.chat.completions.create(
                model=config.model or self.settings.default_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                messages=self._messages(prompt),
            )

            content = (response.choices[0].message.content or "").strip() if response.choices else ""
            if not content:
                raise RuntimeError("API returned empty content")

            usage = response.usage
            token_usage = TokenUsage(
                input_tokens=(usage.prompt_tokens if usage else 0),
                output_tokens=(usage.completion_tokens if usage else 0),
            )
            return GenerationResult(content=content, token_usage=token_usage)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to generate text: {str(e)}") from e

    async def stream_generate(self, prompt: Prompt, config: GenerationConfig) -> AsyncIterator[str]:
        try:
            stream = await self.client.chat.completions.create(
                model=config.model or self.settings.default_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                messages=self._messages(prompt),
                stream=True,
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = delta.content if hasattr(delta, "content") else None
                if isinstance(content, str) and content:
                    yield content
                elif isinstance(content, list):
                    text = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                    if text:
                        yield text
        except Exception as e:
            logger.error("[OpenAICompatibleProvider][Stream] Failed: %s", e)
            raise RuntimeError(f"Failed to stream text: {str(e)}") from e
