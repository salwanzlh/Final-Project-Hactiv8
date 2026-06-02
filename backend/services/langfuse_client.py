from langfuse import Langfuse
from backend.config import settings

_enabled = settings.langfuse_enabled and bool(
    settings.langfuse_public_key and settings.langfuse_secret_key
)

langfuse = Langfuse(
    public_key=settings.langfuse_public_key or "disabled",
    secret_key=settings.langfuse_secret_key or "disabled",
    host=settings.langfuse_host,
    enabled=_enabled,
)
