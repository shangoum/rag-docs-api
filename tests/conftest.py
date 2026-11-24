from collections.abc import AsyncIterator

import httpx
import pytest
from helpers import make_pdf

from app.config import Settings
from app.main import create_app
from app.services.chat import FakeChatClient
from app.services.embeddings import HashingEmbedder

HANDBOOK_PAGES = [
    "The employee is entitled to 25 days of paid holiday per year. "
    "Holiday requests must be submitted to the manager at least two weeks in advance. "
    "Unused holiday days can be carried over until March 31 of the following year.",
    "The notice period is one month during the first year of employment and two months after. "
    "Resignation must be sent by registered letter or by email to human resources.",
    "Remote work is allowed up to two days per week after the trial period. "
    "Employees working remotely receive a monthly allowance of 30 euros for equipment costs.",
]


@pytest.fixture
def handbook_pdf() -> bytes:
    return make_pdf(HANDBOOK_PAGES)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}", llm_provider="fake")


@pytest.fixture
def make_client(settings):
    """Build a test client, optionally with custom model clients or settings overrides."""

    async def factory(chat=None, embedder=None, **overrides) -> AsyncIterator[httpx.AsyncClient]:
        app = create_app(
            settings=settings.model_copy(update=overrides),
            embedder=embedder or HashingEmbedder(),
            chat=chat or FakeChatClient(),
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client

    return factory


@pytest.fixture
async def client(make_client) -> AsyncIterator[httpx.AsyncClient]:
    async for c in make_client():
        yield c
