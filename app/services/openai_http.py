"""Shared HTTP plumbing for OpenAI-compatible APIs (OpenAI, Mistral, Ollama, ...)."""

from typing import Any

import httpx

from app.config import Settings


class LLMError(Exception):
    """Any failure talking to the model provider, with a message safe to show."""


class OpenAICompatibleHTTP:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        if not settings.llm_api_key:
            raise LLMError("LLM_API_KEY is required when LLM_PROVIDER=openai")
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url,
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=settings.llm_timeout_seconds,
            transport=transport,
        )

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Model provider returned HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Model provider request failed: {exc.__class__.__name__}") from exc
        except ValueError as exc:
            raise LLMError("Model provider returned invalid JSON") from exc

    async def aclose(self) -> None:
        await self._client.aclose()
