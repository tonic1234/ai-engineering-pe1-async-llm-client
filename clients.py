"""clients.py — Implementaciones concretas de los proveedores.

APUNTE DE CLASE (la parte que más me costó):
La clase base es ABSTRACTA. Eso significa que no se puede instanciar directamente:
solo sirve como "contrato". Las clases hijas (OpenAI, Anthropic) están obligadas a
implementar generate() y stream(), y si se olvidan de una, Python protesta con
TypeError al instanciarlas. Es justo la idea de la rúbrica: la lógica de negocio
habla contra BaseLLMClient y no sabe qué SDK hay del otro lado.

Detalle que tomé del profe: los SDKs traen versión síncrona y asíncrona. Hay que usar
SIEMPRE la asíncrona (AsyncOpenAI / AsyncAnthropic) adentro de una función async.
Si uso la síncrona, bloqueo el event loop y se cae toda la app (el anti-patrón que
marcó la clase de asyncio).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from schemas import ChatMessage, ModelConfig, ModelResponse

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Error controlado del cliente.

    La idea es NO dejar escapar la excepción cruda del SDK (una API key inválida, un
    429 de rate limit, un timeout de red). Se envuelve en este error propio para que
    quien use el cliente decida qué hacer, y el loop principal nunca se rompa.
    """


class BaseLLMClient(ABC):
    """Contrato común para cualquier proveedor de LLM."""

    provider: str = "base"

    def __init__(self, api_key: str, model: str, config: ModelConfig | None = None) -> None:
        self.model = model
        self.config = config or ModelConfig()

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Devuelve la respuesta completa (no bloqueante)."""

        raise NotImplementedError

    @abstractmethod
    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Devuelve los tokens a medida que llegan (generador asíncrono)."""

        raise NotImplementedError

    @staticmethod
    def _payload(messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Convierte los ChatMessage a la lista de dicts que esperan los SDKs."""

        return [m.model_dump(mode="json") for m in messages]


class OpenAIClient(BaseLLMClient):
    provider = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", config: ModelConfig | None = None) -> None:
        super().__init__(api_key, model, config)
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=self._payload(messages),
            )
        except Exception as exc:  # red, 401, 429, timeout...
            logger.error("Fallo OpenAI en generate(): %s", exc)
            raise LLMError(f"openai: {exc}") from exc

        choice = response.choices[0]
        return ModelResponse(
            content=choice.message.content or "",
            model=self.model,
            provider=self.provider,
            finish_reason=choice.finish_reason,
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=self._payload(messages),
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:  # ojo: hay chunks vacíos (el primero y el último suelen serlo)
                    yield delta
        except Exception as exc:
            logger.error("Fallo OpenAI en stream(): %s", exc)
            raise LLMError(f"openai: {exc}") from exc


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-latest", config: ModelConfig | None = None) -> None:
        super().__init__(api_key, model, config)
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            response = await self.client.messages.create(
                model=self.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=self._payload(messages),
            )
        except Exception as exc:
            logger.error("Fallo Anthropic en generate(): %s", exc)
            raise LLMError(f"anthropic: {exc}") from exc

        # Diferencia con OpenAI: acá content es una LISTA de bloques, no un string.
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        return ModelResponse(
            content=text,
            model=self.model,
            provider=self.provider,
            finish_reason=response.stop_reason,
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            async with self.client.messages.stream(
                model=self.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=self._payload(messages),
            ) as stream:
                async for token in stream.text_stream:
                    yield token
        except Exception as exc:
            logger.error("Fallo Anthropic en stream(): %s", exc)
            raise LLMError(f"anthropic: {exc}") from exc
