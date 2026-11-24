import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import router
from app.config import Settings, get_settings
from app.db import Database
from app.services.chat import ChatClient, build_chat_client
from app.services.embeddings import Embedder, build_embedder
from app.services.rag import RagService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    chat: ChatClient | None = None,
) -> FastAPI:
    """App factory. Tests inject their own settings and model clients."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db = Database(settings.database_url)
        await db.create_tables()
        embedder_client = embedder or build_embedder(settings)
        chat_client = chat or build_chat_client(settings)
        app.state.settings = settings
        app.state.db = db
        app.state.rag = RagService(settings, embedder_client, chat_client)
        yield
        for client in (embedder_client, chat_client):
            if hasattr(client, "aclose"):
                await client.aclose()
        await db.dispose()

    app = FastAPI(
        title="RAG Docs API",
        description="Upload documents, then ask questions answered with cited sources.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
