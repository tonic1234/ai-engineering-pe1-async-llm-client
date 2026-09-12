"""manager.py — El punto único de entrada al cliente.

APUNTE DE CLASE:
Este es el patrón Factory que vimos: el manager mira una variable de entorno
(LLM_PROVIDER) y construye el cliente que corresponde. La lógica de negocio le pide
el manager y listo; nunca importa openai ni anthropic directamente.

Lo bueno de esta separación es que si mañana aparece otro proveedor, lo único que
cambio es acá adentro. El resto del código ni se entera.
"""

from __future__ import annotations

import os
from typing import AsyncIterator

from clients import AnthropicClient, BaseLLMClient, LLMError, OpenAIClient
from schemas import ChatMessage, ModelConfig, ModelResponse

# Modelos por defecto de cada proveedor (se pueden pisar con la env LLM_MODEL).
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
}


class AsyncLLMManager:
    """Elige el proveedor según configuración y expone una interfaz única."""

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider)

        if provider == "openai":
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("Falta OPENAI_API_KEY (definila en el .env)")
            self._client: BaseLLMClient = OpenAIClient(key, model, config)
        elif provider == "anthropic":
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("Falta ANTHROPIC_API_KEY (definila en el .env)")
            self._client = AnthropicClient(key, model, config)
        else:
            raise ValueError(f"Proveedor no soportado: {provider!r} (opciones: openai, anthropic)")

    @property
    def provider(self) -> str:
        return self._client.provider

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        return await self._client.generate(messages)

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        # Delegamos el generador async al cliente concreto.
        async for token in self._client.stream(messages):
            yield token

    # Alias cómodo para cuando hay que comparar varios modelos en paralelo
    # (lo usamos en el ejercicio de gather + semáforo).
    async def safe_generate(self, messages: list[ChatMessage]) -> tuple[str, str]:
        """Igual que generate() pero nunca lanza: devuelve (proveedor, texto|error)."""

        try:
            response = await self.generate(messages)
            return self.provider, response.content
        except (LLMError, ValueError) as exc:
            return self.provider, f"[error controlado] {exc}"
