"""Chat providers that write the final answer from the retrieved passages."""

from typing import Protocol

import httpx

from app.config import Settings
from app.services.openai_http import LLMError, OpenAICompatibleHTTP


class ChatClient(Protocol):
    async def answer(self, system_prompt: str, user_prompt: str) -> str: ...


class FakeChatClient:
    """Offline client: returns the first retrieved passage as an extractive answer."""

    async def answer(self, system_prompt: str, user_prompt: str) -> str:
        for line in user_prompt.splitlines():
            if line.startswith("[1]"):
                passage = line.split(":", 1)[-1].strip()
                return f"According to the documents: {passage[:300]} [1]"
        return "I could not find the answer in the documents."


class OpenAIChatClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self._model = settings.chat_model
        self._http = OpenAICompatibleHTTP(settings, transport)

    async def answer(self, system_prompt: str, user_prompt: str) -> str:
        data = await self._http.post(
            "/chat/completions",
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
            },
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected chat response format") from exc
        if not content or not content.strip():
            raise LLMError("Model returned an empty answer")
        return content.strip()

    async def aclose(self) -> None:
        await self._http.aclose()


def build_chat_client(settings: Settings) -> ChatClient:
    if settings.llm_provider == "openai":
        return OpenAIChatClient(settings)
    return FakeChatClient()
