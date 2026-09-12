# Cliente de LLM unificado y asíncrono

Pre-entrega 1 del curso **AI Engineering** (Coderhouse).
Implementa un cliente que permite intercambiar entre **OpenAI** y **Anthropic** con la
misma interfaz, todo asíncrono y con soporte de *streaming* de tokens.

## Qué hay adentro

| Archivo | Qué hace |
|---|---|
| `schemas.py` | Modelos Pydantic: `ChatMessage`, `Role`, `ModelConfig`, `ModelResponse`. |
| `clients.py` | `BaseLLMClient` (clase abstracta) + `OpenAIClient` y `AnthropicClient`. |
| `manager.py` | `AsyncLLMManager`: elige el proveedor según la variable de entorno. |
| `main.py` | Script de prueba: hace una pregunta en modo normal y en streaming. |
| `tests/` | Pruebas que corren **sin API key** (el cliente del SDK se reemplaza por un doble). |

## Cómo correrlo

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # completar la API key del proveedor elegido
python main.py
```

## Variables de entorno

| Variable | Descripción |
|---|---|
| `LLM_PROVIDER` | `openai` o `anthropic` (por defecto `openai`). |
| `OPENAI_API_KEY` | Requerida si el proveedor es OpenAI. |
| `ANTHROPIC_API_KEY` | Requerida si el proveedor es Anthropic. |
| `LLM_MODEL` | Opcional: sobreescribe el modelo por defecto del proveedor. |

## Ejemplo de salida

```
INFO main: Proveedor activo: openai

--- Modo normal ---
La entropía es una medida del desorden de un sistema...
(modelo=gpt-4o-mini finish_reason=stop)

--- Modo streaming ---
La entropía es una medida del desorden de un sistema...
```

## Decisiones de diseño

- **Abstracción**: `BaseLLMClient` es una ABC con `@abstractmethod`; el resto del código
  no importa los SDK de OpenAI ni de Anthropic, solo el manager.
- **Asincronía**: se usan las versiones `AsyncOpenAI` / `AsyncAnthropic`. Nunca se llama a
  la variante síncrona dentro de una función `async`, porque bloquearía el event loop.
- **Streaming**: `stream()` es un generador asíncrono (`yield` dentro de un `async for`).
  Se saltean los chunks vacíos que mandan los SDK.
- **Validación**: `ModelConfig` restringe `temperature` a [0, 2]; `Role` es un `Enum`, así
  que un rol inválido falla antes de llegar a la API.
- **Resiliencia**: los errores del SDK se envuelven en `LLMError`, de modo que un 429 o una
  key inválida no tumban el programa.

## Tests

```bash
pytest -q
```

Corren sin conexión y sin API key: reemplazan el cliente del SDK por un `FakeCompletions`
para poder verificar el comportamiento del código propio (orden de los tokens, manejo de
errores, validaciones).
