# Cliente de LLM unificado y asíncrono

Pre-entrega 1 del curso **AI Engineering** (Coderhouse).
Implementa un cliente que permite intercambiar entre **OpenAI**, **Anthropic** y **Gemini**
con la misma interfaz, todo asíncrono y con soporte de *streaming* de tokens.

## Qué hay adentro

| Archivo | Qué hace |
|---|---|
| `schemas.py` | Modelos Pydantic: `Provider`, `Role`, `ChatMessage`, `ModelConfig`, `ModelResponse`, `LLMConfig`. |
| `clients.py` | `BaseLLMClient` (ABC) + `OpenAIClient`, `AnthropicClient` y `GeminiClient`. |
| `manager.py` | `AsyncLLMManager` (Factory): elige el proveedor según la configuración. |
| `main.py` | Prueba: modo normal, modo streaming y resiliencia (key inválida). |
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
| `LLM_PROVIDER` | `openai`, `anthropic` o `gemini` (por defecto `gemini`). |
| `GOOGLE_API_KEY` | Requerida si el proveedor es Gemini. |
| `OPENAI_API_KEY` | Requerida si el proveedor es OpenAI. |
| `ANTHROPIC_API_KEY` | Requerida si el proveedor es Anthropic. |
| `LLM_MODEL` | Opcional: sobreescribe el modelo por defecto del proveedor. |

> **Gemini tiene free tier y no pide tarjeta**: la key se saca gratis en
> <https://aistudio.google.com/apikey>. Es el proveedor con el que está probado el repo
> de punta a punta, así no hace falta cargar tarjeta en OpenAI/Anthropic.

## Ejemplo de salida

```
INFO main: Proveedor activo: gemini (modelo=gemini-flash-latest)

--- Modo normal ---
🟢 OK: La entropía es una medida del desorden de un sistema...

--- Modo streaming ---
La entropía es una medida del desorden de un sistema...

--- Prueba de resiliencia (key inválida a propósito) ---
¿El programa siguió vivo?: ✅ Sí
Error capturado (sin crash): Error de la API de OpenAI: Error code: 401 - Incorrect API key...
```

## Qué cambia entre proveedores

| Característica | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| SDK async | `AsyncOpenAI` | `AsyncAnthropic` | `genai.Client(...).aio` |
| Método de chat | `chat.completions.create(...)` | `messages.create(...)` | `aio.models.generate_content(...)` |
| Dónde está el texto | `choices[0].message.content` | `content[0].text` | `response.text` |
| Streaming | `stream=True` + `async for` | `messages.stream(...)` (context manager) | `generate_content_stream(...)` + `async for` |
| Rol del asistente | `assistant` | `assistant` | `model` |

## Decisiones de diseño

- **Abstracción**: `BaseLLMClient` es una ABC con `@abstractmethod`; el resto del código no
  importa los SDK, solo el manager.
- **Asincronía**: se usan las versiones async de los tres SDK. Nunca la síncrona dentro de
  una función `async`, porque bloquearía el event loop.
- **Streaming**: `generate_stream()` es un generador asíncrono (`yield` dentro de un
  `async for`). Se saltean los chunks vacíos que mandan los SDK.
- **Validación**: `temperature` restringida a [0, 2]; `Role` es un `Enum`; las API keys van
  en `SecretStr`, así que nunca se imprimen ni quedan en los logs.
- **Resiliencia**: las excepciones **no se propagan**. Cada error se devuelve dentro del
  `ModelResponse` (campo `error`), o como un aviso dentro del stream. Una key inválida o un
  429 no tumban el programa (es la "fuga de excepciones" que menciona la consigna).

## Tests

```bash
pytest -q
```

Corren sin conexión y sin API key: reemplazan el cliente del SDK por un `FakeCompletions`
para verificar el comportamiento del código propio (orden de los tokens, manejo de errores,
validaciones, y que el factory construya los tres proveedores).
