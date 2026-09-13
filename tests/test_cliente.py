"""tests/test_cliente.py — Pruebas del cliente unificado.

APUNTE: para probar sin gastar tokens (y sin necesitar una API key real) reemplazo el
cliente interno del SDK por un doble ("fake"). Así verifico el comportamiento del
CÓDIGO PROPIO: que el streaming respete el orden, que los errores viajen dentro del
ModelResponse (sin crash) y que las validaciones de Pydantic salten cuando corresponde.

Correr:  pytest -q
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr, ValidationError

from clients import BaseLLMClient, GeminiClient, OpenAIClient
from manager import AsyncLLMManager, config_desde_entorno
from schemas import ChatMessage, LLMConfig, ModelConfig, Provider, Role


# --------------------------------------------------------------------------
# Dobles de prueba
# --------------------------------------------------------------------------
def _chunk(text):
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
    def __init__(self, *, fail: Exception | None = None, tokens=("Hola", ", ", "mundo")):
        self.fail = fail
        self.tokens = tokens

    async def create(self, **kwargs):
        if self.fail:
            raise self.fail
        if kwargs.get("stream"):
            # incluyo un chunk vacío a propósito: el SDK real los manda
            return _AsyncIter([_chunk(None), *[_chunk(t) for t in self.tokens], _chunk("")])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="respuesta simulada"))]
        )


def make_client(fail=None) -> OpenAIClient:
    client = OpenAIClient(api_key="test-key", model="gpt-4o-mini", temperature=0.7, max_tokens=100)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(fail=fail)))
    return client


MSG = [ChatMessage(role=Role.USER, content="¿Qué es la entropía?")]


def _config(provider=Provider.OPENAI, **kwargs) -> LLMConfig:
    campos = {
        "provider": provider,
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 100,
        "openai_api_key": SecretStr("test-key"),
        "anthropic_api_key": SecretStr("test-key"),
        "google_api_key": SecretStr("test-key"),
    }
    campos.update(kwargs)
    return LLMConfig(**campos)


# --------------------------------------------------------------------------
# 1. Validación Pydantic
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


def test_secretstr_oculta_la_key():
    cfg = _config()
    assert "test-key" not in repr(cfg)  # SecretStr NO expone el valor


# --------------------------------------------------------------------------
# 2. Abstracción y factory
# --------------------------------------------------------------------------
def test_la_base_es_abstracta():
    with pytest.raises(TypeError):
        BaseLLMClient(model="x", temperature=0.5, max_tokens=10)  # type: ignore[abstract]


@pytest.mark.parametrize(
    "provider",
    [Provider.OPENAI, Provider.ANTHROPIC, Provider.GEMINI],
)
def test_manager_construye_los_tres_proveedores(provider):
    manager = AsyncLLMManager(_config(provider))
    assert manager._client.provider == provider


def test_manager_falla_si_falta_la_key():
    with pytest.raises(ValueError):
        AsyncLLMManager(LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini"))


def test_config_desde_entorno(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    cfg = config_desde_entorno()
    assert cfg.provider is Provider.GEMINI
    assert cfg.model == "gemini-flash-latest"


# --------------------------------------------------------------------------
# 3. Streaming asíncrono
# --------------------------------------------------------------------------
def test_generate_devuelve_respuesta():
    response = asyncio.run(make_client().generate(MSG))
    assert response.content == "respuesta simulada"
    assert response.provider is Provider.OPENAI
    assert response.error is None


def test_stream_entrega_tokens_en_orden():
    async def run():
        return [t async for t in make_client().generate_stream(MSG)]

    assert asyncio.run(run()) == ["Hola", ", ", "mundo"]


def test_stream_salta_chunks_vacios():
    async def run():
        return [t async for t in make_client().generate_stream(MSG)]

    tokens = asyncio.run(run())
    assert "" not in tokens and None not in tokens


def test_manager_delega_el_streaming(monkeypatch):
    manager = AsyncLLMManager(_config())
    manager._client = make_client()

    async def run():
        return [t async for t in manager.generate_stream(MSG)]

    assert asyncio.run(run()) == ["Hola", ", ", "mundo"]


# --------------------------------------------------------------------------
# 4. Resiliencia: el error viaja en el ModelResponse, nunca hay crash
# --------------------------------------------------------------------------
def test_error_de_api_no_crashea():
    import openai

    class FakeRespuesta:
        status_code = 401
        request = None
        headers = {}

    error = openai.APIError("incorrect api key", FakeRespuesta(), body=None)
    resultado = asyncio.run(make_client(fail=error).generate(MSG))

    assert isinstance(resultado, object)
    assert resultado.error is not None
    assert "Error de la API de OpenAI" in resultado.error
    assert resultado.content == ""


def test_error_en_streaming_avisa_dentro_del_stream():
    import openai

    class FakeRespuesta:
        status_code = 429
        request = None
        headers = {}

    error = openai.APIError("rate limited", FakeRespuesta(), body=None)

    async def run():
        return [t async for t in make_client(fail=error).generate_stream(MSG)]

    texto = "".join(asyncio.run(run()))
    assert "Error durante el streaming" in texto


def test_gemini_client_existe():
    # El tercer proveedor soportado.
    cliente = GeminiClient(api_key="test-key", model="gemini-flash-latest",
                           temperature=0.7, max_tokens=100)
    assert cliente.provider is Provider.GEMINI
