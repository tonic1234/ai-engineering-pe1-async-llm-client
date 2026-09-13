"""manager.py — Factory que construye el cliente según la configuración.

NOTA:
Este es el patrón Factory que vimos: el manager mira `config.provider` y arma el
cliente que corresponde. La lógica de negocio pide el manager y listo; nunca importa
openai, anthropic ni google directamente.

Lo bueno: si mañana piden "agregá Gemini", solo se crea GeminiClient y se suma una rama
acá adentro. Nada del resto del código cambia. Eso es justamente lo que evita el
acoplamiento.

Detalle: la key se guarda en SecretStr, así que para pasarla al SDK hay que llamar a
.get_secret_value() (es lo único que la expone, y no queda en los logs).
"""

from __future__ import annotations

import os
from typing import AsyncGenerator, List

from pydantic import SecretStr

from clients import AnthropicClient, BaseLLMClient, GeminiClient, OpenAIClient
from schemas import ChatMessage, LLMConfig, ModelResponse, Provider

# Modelos por defecto de cada proveedor.
DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    Provider.GEMINI: "gemini-flash-latest",  # alias estable; tiene free tier
}


class AsyncLLMManager:
    """Elige el proveedor según la configuración y expone una interfaz única."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client: BaseLLMClient = self._crear_cliente()

    def _crear_cliente(self) -> BaseLLMClient:
        cfg = self.config

        if cfg.provider == Provider.OPENAI:
            if not cfg.openai_api_key:
                raise ValueError("Falta openai_api_key en la configuración")
            return OpenAIClient(
                api_key=cfg.openai_api_key.get_secret_value(),
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )

        if cfg.provider == Provider.ANTHROPIC:
            if not cfg.anthropic_api_key:
                raise ValueError("Falta anthropic_api_key en la configuración")
            return AnthropicClient(
                api_key=cfg.anthropic_api_key.get_secret_value(),
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )

        if cfg.provider == Provider.GEMINI:
            if not cfg.google_api_key:
                raise ValueError("Falta google_api_key en la configuración")
            return GeminiClient(
                api_key=cfg.google_api_key.get_secret_value(),
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )

        raise ValueError(f"Proveedor no soportado: {cfg.provider}")

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        return await self._client.generate(messages)

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        # Delegamos el generador async al cliente concreto.
        async for chunk in self._client.generate_stream(messages):
            yield chunk


def config_desde_entorno(provider: str | None = None) -> LLMConfig:
    """Arma la LLMConfig leyendo las variables de entorno.

    Lo agregué para no tener que hardcodear las keys en el script: se cargan del .env.
    """

    provider_enum = Provider((provider or os.getenv("LLM_PROVIDER", "gemini")).lower())
    modelo = os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider_enum]

    def _key(nombre: str) -> SecretStr | None:
        valor = os.getenv(nombre)
        return SecretStr(valor) if valor else None

    return LLMConfig(
        provider=provider_enum,
        model=modelo,
        openai_api_key=_key("OPENAI_API_KEY"),
        anthropic_api_key=_key("ANTHROPIC_API_KEY"),
        google_api_key=_key("GOOGLE_API_KEY"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
    )
