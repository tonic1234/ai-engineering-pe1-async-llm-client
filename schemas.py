"""schemas.py — Modelos de datos (contratos) del cliente unificado de LLMs.

APUNTE DE CLASE:
En la unidad de Pydantic vimos que la idea de un "esquema" es definir de antemano
la FORMA que tienen los datos que entran y salen del sistema. Así, si algo viene mal
(por ejemplo un rol que no existe), falla ANTES de llegar al LLM y no en medio de la
llamada a la API.

El profe insistió en un punto: definir esto PRIMERO evita el famoso "error de
diccionarios anidados". En vez de andar armando
    {"role": "user", "content": [...]}   # y equivocarse en la clave
armamos objetos ChatMessage y listo.
"""

from enum import Enum

from pydantic import BaseModel, Field, SecretStr


class Role(str, Enum):
    """Roles válidos de un mensaje.

    Heredo de str para que el valor se pueda serializar directo a JSON cuando el
    SDK de turno lo necesite (si no, tendría que hacer .value a cada rato).
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Un mensaje de la conversación."""

    role: Role
    content: str = Field(min_length=1, description="Texto del mensaje")


class ModelConfig(BaseModel):
    """Parámetros de generación.

    Acá están las validaciones "de negocio": la temperatura de los LLMs va de 0 a 2,
    no tiene sentido permitir 5. Si alguien pasa 5, Pydantic lanza ValidationError.
    """

    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, gt=0, le=8192)


class ModelResponse(BaseModel):
    """Lo que devuelve un cliente: el texto más la metadata útil."""

    content: str
    model: str
    provider: str
    finish_reason: str | None = None


class ProviderSettings(BaseModel):
    """Configuración leída del entorno.

    Uso SecretStr para la API key: de esta forma, si imprimo el objeto o lo logueo,
    Pydantic muestra '**********' en lugar de la clave real. Es la recomendación que
    quedó del repaso de seguridad.
    """

    provider: str = Field(default="openai", pattern="^(openai|anthropic)$")
    api_key: SecretStr
    model: str
