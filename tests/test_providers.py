import json

import httpx
import pytest

from app.config import Settings
from app.services.chat import OpenAIChatClient
from app.services.embeddings import HashingEmbedder, OpenAIEmbedder
from app.services.openai_http import LLMError


def _settings(**kwargs) -> Settings:
    base = {"llm_provider": "openai", "llm_api_key": "test-key", "embedding_batch_size": 2}
    return Settings(**(base | kwargs))


async def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(dimensions=64)
    [a], [b] = await embedder.embed(["Notice period"]), await embedder.embed(["Notice period"])
    assert a == b
    assert abs(sum(v * v for v in a) - 1.0) < 1e-6
    [empty] = await embedder.embed(["the and of"])  # only stopwords
    assert not any(empty)


async def test_openai_embedder_batches_and_keeps_order():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body["input"])
        # return items out of order to check that we sort by index
        data = [{"index": i, "embedding": [float(len(t))]} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": list(reversed(data))})

    embedder = OpenAIEmbedder(_settings(), transport=httpx.MockTransport(handler))
    vectors = await embedder.embed(["a", "bb", "ccc"])
    await embedder.aclose()

    assert requests == [["a", "bb"], ["ccc"]]
    assert vectors == [[1.0], [2.0], [3.0]]


async def test_openai_embedder_rejects_wrong_count():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    embedder = OpenAIEmbedder(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="number of inputs"):
        await embedder.embed(["a", "b"])
    await embedder.aclose()


async def test_openai_chat_sends_prompts_with_zero_temperature():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": " Answer [1] "}}]})

    chat = OpenAIChatClient(_settings(chat_model="m"), transport=httpx.MockTransport(handler))
    assert await chat.answer("sys", "user") == "Answer [1]"
    await chat.aclose()
    assert captured["model"] == "m"
    assert captured["temperature"] == 0
    assert captured["messages"][0] == {"role": "system", "content": "sys"}


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(429), "HTTP 429"),
        (httpx.Response(200, content=b"not json"), "invalid JSON"),
        (httpx.Response(200, json={"choices": []}), "Unexpected chat response"),
        (httpx.Response(200, json={"choices": [{"message": {"content": ""}}]}), "empty answer"),
    ],
)
async def test_openai_chat_errors(response, expected):
    chat = OpenAIChatClient(_settings(), transport=httpx.MockTransport(lambda _: response))
    with pytest.raises(LLMError, match=expected):
        await chat.answer("sys", "user")
    await chat.aclose()


def test_api_key_required():
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        OpenAIChatClient(Settings(llm_provider="openai", llm_api_key=""))
