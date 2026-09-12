"""clients.py — Implementaciones concretas de los proveedores.

APUNTE DE CLASE (la parte que más me costó):
La clase base es ABSTRACTA. No se puede instanciar directamente: sirve como contrato.
Las clases hijas están obligadas a implementar generate() y generate_stream(), y si se
olvidan de una, Python protesta al instanciarlas. La lógica de negocio habla siempre
contra BaseLLMClient y no sabe qué SDK hay del otro lado.

Dos detalles que tomé de la clase:
  - Los SDKs traen versión síncrona y asíncrona: hay que usar SIEMPRE la async
    (AsyncOpenAI / AsyncAnthropic / genai.Client(...).aio). La síncrona bloquea el
    event loop y congela la app entera.
  - Ninguna excepción se deja escapar: se captura y se devuelve un ModelResponse con
    el campo `error` (o un chunk de aviso en el streaming). Nunca un crash.

Lo que cambia entre proveedores (lo anoté en la tabla de la clase):
  - OpenAI:      response.choices[0].message.content
  - Anthropic:   response.content[0].text  (content es LISTA, no string)
  - Gemini:      response.text  (y el rol del asistente se llama "model", no "assistant")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List

from anthropic import (
    APIConnectionError as AnthropicConnectionError,
    APIError as AnthropicAPIError,
    AsyncAnthropic,
    RateLimitError as AnthropicRateLimitError,
)
from google import genai
from google.genai import types
from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    RateLimitError,
)

from schemas import ChatMessage, ModelResponse, Provider

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Contrato que todo cliente de LLM debe cumplir, sin importar el proveedor."""

    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa (modo normal, no streaming)."""

        raise NotImplementedError

    @abstractmethod
    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """Genera la respuesta token a token (modo streaming)."""

        raise NotImplementedError
        yield  # nunca se ejecuta: solo le indica a Python que es un generador

    @staticmethod
    def _payload(messages: List[ChatMessage]) -> List[dict]:
        return [m.model_dump(mode="json") for m in messages]


class OpenAIClient(BaseLLMClient):
    provider = Provider.OPENAI

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int) -> None:
        super().__init__(model, temperature, max_tokens)
        self._client = AsyncOpenAI(api_key=api_key)

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=self._payload(messages),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.model,
                content=response.choices[0].message.content or "",
            )
        except RateLimitError as e:
            return ModelResponse(provider=Provider.OPENAI, model=self.model,
                                 error=f"Límite de cuota excedido: {e}")
        except APIConnectionError as e:
            return ModelResponse(provider=Provider.OPENAI, model=self.model,
                                 error=f"Error de conexión: {e}")
        except APIError as e:
            return ModelResponse(provider=Provider.OPENAI, model=self.model,
                                 error=f"Error de la API de OpenAI: {e}")

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=self._payload(messages),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:  # hay chunks vacíos (el primero y el último suelen serlo)
                    yield delta
        except (RateLimitError, APIConnectionError, APIError) as e:
            yield f"\n[⚠️ Error durante el streaming: {e}]"


class AnthropicClient(BaseLLMClient):
    provider = Provider.ANTHROPIC

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int) -> None:
        super().__init__(model, temperature, max_tokens)
        self._client = AsyncAnthropic(api_key=api_key)

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,  # obligatorio en Anthropic, a diferencia de OpenAI
                temperature=self.temperature,
                messages=self._payload(messages),
            )
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model,
                content=response.content[0].text,  # ojo: content es una LISTA
            )
        except AnthropicRateLimitError as e:
            return ModelResponse(provider=Provider.ANTHROPIC, model=self.model,
                                 error=f"Límite de cuota excedido: {e}")
        except AnthropicConnectionError as e:
            return ModelResponse(provider=Provider.ANTHROPIC, model=self.model,
                                 error=f"Error de conexión: {e}")
        except AnthropicAPIError as e:
            return ModelResponse(provider=Provider.ANTHROPIC, model=self.model,
                                 error=f"Error de la API de Anthropic: {e}")

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=self._payload(messages),
            ) as stream:
                async for texto in stream.text_stream:
                    yield texto
        except (AnthropicRateLimitError, AnthropicConnectionError, AnthropicAPIError) as e:
            yield f"\n[⚠️ Error durante el streaming: {e}]"


class GeminiClient(BaseLLMClient):
    """Gemini: es el que tiene free tier (no pide tarjeta), así que lo uso para probar."""

    provider = Provider.GEMINI

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int) -> None:
        super().__init__(model, temperature, max_tokens)
        self._client = genai.Client(api_key=api_key)

    def _convertir_mensajes(self, messages: List[ChatMessage]):
        """Gemini separa el system prompt del resto y llama 'model' al rol del asistente."""

        contents = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                rol_gemini = "model" if m.role == "assistant" else "user"
                contents.append(types.Content(role=rol_gemini, parts=[types.Part(text=m.content)]))
        return contents, system_instruction

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            contents, system_instruction = self._convertir_mensajes(messages)
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    system_instruction=system_instruction,
                ),
            )
            return ModelResponse(provider=Provider.GEMINI, model=self.model, content=response.text or "")
        except Exception as e:  # el SDK de Google no expone las mismas clases de error
            return ModelResponse(provider=Provider.GEMINI, model=self.model,
                                 error=f"Error de la API de Gemini: {e}")

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            contents, system_instruction = self._convertir_mensajes(messages)
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    system_instruction=system_instruction,
                ),
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n[⚠️ Error durante el streaming: {e}]"
