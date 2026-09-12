"""tests/test_cliente.py — Pruebas del cliente unificado.

APUNTE: para probar sin gastar tokens (y sin necesitar una API key real) reemplazo el
cliente del SDK por un doble ("fake"). Así puedo verificar el comportamiento del
CÓDIGO PROPIO: que el streaming respete el orden, que los errores se envuelvan en
LLMError y que las validaciones de Pydantic salten cuando tienen que saltar.

Correr:  pytest -q
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from clients import BaseLLMClient, LLMError, OpenAIClient
from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig, Role


# --------------------------------------------------------------------------
# Dobles de prueba (fakes)
# --------------------------------------------------------------------------
def _chunk(text: str | None):
    """Arma un chunk con la misma forma que devuelve el SDK de OpenAI."""

    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class _AsyncIter:
    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        async def gen():
            for item in self._items:
                yield item

        return gen()


class FakeCompletions:
    """Imita client.chat.completions, pero devuelve datos fijos."""

    def __init__(self, *, fail: bool = False, tokens=("Hola", ", ", "mundo")):
        self.fail = fail
        self.tokens = tokens

    async def create(self, **kwargs):
        if self.fail:
            raise RuntimeError("429 Too Many Requests (simulado)")
        if kwargs.get("stream"):
            # incluyo un chunk vacío a propósito: el SDK real los manda y hay que saltarlos
            return _AsyncIter([_chunk(None), *[_chunk(t) for t in self.tokens], _chunk("")])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="respuesta simulada"),
                    finish_reason="stop",
                )
            ]
        )


def make_client(fail: bool = False) -> OpenAIClient:
    client = OpenAIClient(api_key="test-key", model="gpt-4o-mini")
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(fail=fail)))
    return client


MSG = [ChatMessage(role=Role.USER, content="¿Qué es la entropía?")]


# --------------------------------------------------------------------------
# 1. Validación Pydantic (criterio "Validación" de la rúbrica)
# --------------------------------------------------------------------------
def test_temperatura_fuera_de_rango_falla():
    with pytest.raises(ValidationError):
        ModelConfig(temperature=5)


def test_temperatura_en_el_limite_es_valida():
    assert ModelConfig(temperature=2).temperature == 2


def test_role_invalido_falla():
    with pytest.raises(ValidationError):
        ChatMessage(role="root", content="hola")


def test_contenido_vacio_falla():
    with pytest.raises(ValidationError):
        ChatMessage(role=Role.USER, content="")


# --------------------------------------------------------------------------
# 2. Abstracción (criterio "Abstracción y gestión de proveedores")
# --------------------------------------------------------------------------
def test_la_base_es_abstracta():
    with pytest.raises(TypeError):
        BaseLLMClient(api_key="x", model="y")  # type: ignore[abstract]


def test_manager_elige_proveedor_por_entorno(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    manager = AsyncLLMManager(api_key="test-key")
    assert manager.provider == "anthropic"


def test_manager_falla_sin_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        AsyncLLMManager(provider="openai")


# --------------------------------------------------------------------------
# 3. Streaming asíncrono (criterio "Streaming asíncrono")
# --------------------------------------------------------------------------
def test_generate_devuelve_respuesta():
    client = make_client()
    response = asyncio.run(client.generate(MSG))
    assert response.content == "respuesta simulada"
    assert response.provider == "openai"


def test_stream_entrega_tokens_en_orden():
    client = make_client()

    async def run():
        return [token async for token in client.stream(MSG)]

    assert asyncio.run(run()) == ["Hola", ", ", "mundo"]


def test_stream_salta_chunks_vacios():
    client = make_client()

    async def run():
        return [t async for t in client.stream(MSG)]

    tokens = asyncio.run(run())
    assert "" not in tokens and None not in tokens


# --------------------------------------------------------------------------
# 4. Resiliencia (criterio "Resiliencia": error controlado, no crash)
# --------------------------------------------------------------------------
def test_error_de_red_se_envuelve_en_llm_error():
    client = make_client(fail=True)
    with pytest.raises(LLMError):
        asyncio.run(client.generate(MSG))


def test_safe_generate_no_propaga_el_error():
    manager = AsyncLLMManager(provider="openai", api_key="test-key")
    manager._client = make_client(fail=True)

    provider, text = asyncio.run(manager.safe_generate(MSG))
    assert provider == "openai"
    assert text.startswith("[error controlado]")
