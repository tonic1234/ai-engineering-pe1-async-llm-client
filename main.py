"""main.py — Script de prueba del cliente unificado.

Prueba, en este orden:
  1. Modo normal (generate) con el proveedor configurado.
  2. Modo streaming (generate_stream): los tokens tienen que ir apareciendo de a uno.
  3. Resiliencia: con una API key inválida el programa NO se cae, devuelve el error
     adentro del ModelResponse.

Uso:
    cp .env.example .env   # y completar la key del proveedor
    python main.py
"""

from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv
from pydantic import SecretStr

from manager import AsyncLLMManager, config_desde_entorno
from schemas import ChatMessage, LLMConfig, Provider, Role

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

PREGUNTA = [ChatMessage(role=Role.USER, content="¿Qué es la entropía? Respondé en 2 líneas.")]


async def main() -> None:
    load_dotenv()

    config = config_desde_entorno()
    manager = AsyncLLMManager(config)
    logger.info("Proveedor activo: %s (modelo=%s)", config.provider.value, config.model)

    print("\n--- Modo normal ---")
    resultado = await manager.generate(PREGUNTA)
    print("🟢 OK:", resultado.content if not resultado.error else f"❌ {resultado.error}")

    print("\n--- Modo streaming ---")
    async for chunk in manager.generate_stream(PREGUNTA):
        print(chunk, end="", flush=True)
    print()

    print("\n--- Prueba de resiliencia (key inválida a propósito) ---")
    config_rota = LLMConfig(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        openai_api_key=SecretStr("sk-key-invalida-a-proposito"),
    )
    resultado = await AsyncLLMManager(config_rota).generate(PREGUNTA)
    print("¿El programa siguió vivo?: ✅ Sí")
    print("Error capturado (sin crash):", (resultado.error or "")[:120])


if __name__ == "__main__":
    asyncio.run(main())
