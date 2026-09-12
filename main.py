"""main.py — Script de prueba del cliente unificado.

Corre una pregunta corta en modo normal y en modo streaming, para verificar que
ambos caminos funcionan y que el streaming realmente "tira" los tokens de a uno.

Uso:
    cp .env.example .env   # y completar la key
    python main.py
"""

from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig, Role

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

PREGUNTA = "¿Qué es la entropía? Respondé en 3 líneas."


async def main() -> None:
    load_dotenv()
    manager = AsyncLLMManager(config=ModelConfig(temperature=0.3, max_tokens=300))
    logger.info("Proveedor activo: %s", manager.provider)

    messages = [ChatMessage(role=Role.USER, content=PREGUNTA)]

    print("\n--- Modo normal ---")
    response = await manager.generate(messages)
    print(response.content)
    print(f"(modelo={response.model} finish_reason={response.finish_reason})")

    print("\n--- Modo streaming ---")
    async for token in manager.stream(messages):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
