"""Embedding providers: turn texts into vectors whose cosine similarity reflects meaning."""

import hashlib
import math
import re
from typing import Protocol

import httpx

from app.config import Settings
from app.services.openai_http import LLMError, OpenAICompatibleHTTP

_WORD = re.compile(r"\w+", re.UNICODE)

# Frequent words that carry no meaning; ignoring them keeps unrelated texts far apart.
_STOPWORDS_EN = (
    "the and for are was were with that this from what which who whom how why when where "
    "can could does did has have had will would should may must not but you your our their"
)
_STOPWORDS_FR = (
    "les des une est sont pour dans avec que qui quoi quel quelle comment pourquoi quand "
    "par sur pas plus ont aux ces cette mon mes vos nos leur"
)
_STOPWORDS = frozenset(f"{_STOPWORDS_EN} {_STOPWORDS_FR}".split())  # noqa: SIM905


class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Offline embedder based on hashed word counts (no model, no network).

    It only captures shared vocabulary, not meaning, but it is deterministic and good
    enough to exercise the whole pipeline in development, tests and CI.
    """

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions
        self.name = f"hashing-{dimensions}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in _WORD.findall(text.lower()):
            if len(word) < 3 or word in _STOPWORDS:
                continue
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector


class OpenAIEmbedder:
    """Calls POST /embeddings on any OpenAI-compatible API, in batches."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.name = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._http = OpenAICompatibleHTTP(settings, transport)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            data = await self._http.post("/embeddings", {"model": self.name, "input": batch})
            try:
                items = sorted(data["data"], key=lambda item: item["index"])
                vectors.extend(item["embedding"] for item in items)
            except (KeyError, TypeError) as exc:
                raise LLMError("Unexpected embeddings response format") from exc
            if len(items) != len(batch):
                raise LLMError("Embeddings response does not match the number of inputs")
        return vectors

    async def aclose(self) -> None:
        await self._http.aclose()


def build_embedder(settings: Settings) -> Embedder:
    if settings.llm_provider == "openai":
        return OpenAIEmbedder(settings)
    return HashingEmbedder()
