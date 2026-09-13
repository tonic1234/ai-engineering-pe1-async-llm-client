"""schemas.py — Modelos de datos (contratos) del cliente unificado de LLMs.

APUNTE DE CLASE:
En la unidad de Pydantic vimos que la idea de un "esquema" es definir de antemano la
FORMA de los datos que entran y salen del sistema. Así, si algo viene mal (por ejemplo
una temperatura de 5), falla ANTES de llamar a la API y no en medio de la petición.

El profe insistió en dos cosas:
  1. Definir esto PRIMERO evita el "error de diccionarios anidados" típico de
     principiante (armar a mano {"role": ..., "content": ...} y equivocarse en la clave).
  2. Las API keys van en SecretStr: si imprimo o logueo la config, sale '**********' en
     lugar de la clave. Es lo que vimos en el repaso de seguridad.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, SecretStr


class Provider(str, Enum):
    """Proveedores soportados.

    Heredo de str para que el valor se serialice directo a JSON.
    (Sumo Gemini además de OpenAI y Anthropic: tiene free tier y no pide tarjeta.)
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class Role(str, Enum):
    """Roles válidos de un mensaje."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Un mensaje de la conversación.

    Al ser Role un Enum, un rol inválido (por ejemplo "root") lanza ValidationError
    antes de llegar a la API.
    """

    role: Role
    content: str = Field(min_length=1, description="Texto del mensaje")


class ModelConfig(BaseModel):
    """Parámetros de generación (validaciones de negocio).

    La temperatura de los LLM va de 0 a 2: permitir 5 no tiene sentido y es mejor que
    lo ataje Pydantic.
    """

    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)


class ModelResponse(BaseModel):
    """Lo que devuelve un cliente: el texto, la metadata y el error si hubo.

    Decisión de diseño importante: los errores NO se propagan como excepción, viajan
    adentro del ModelResponse en el campo `error`. Así una cuota agotada o una key
    inválida nunca tumban el loop principal de la app (la "fuga de excepciones" que
    menciona la consigna como error común).
    """

    provider: Provider
    model: str
    content: str = ""
    error: Optional[str] = None


class LLMConfig(BaseModel):
    """Configuración completa del manager.

    Cada proveedor tiene su propia key (opcional) porque el manager elige una sola
    según `provider`. Con SecretStr, la clave nunca se imprime.
    """

    provider: Provider
    model: str
    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    google_api_key: Optional[SecretStr] = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)
