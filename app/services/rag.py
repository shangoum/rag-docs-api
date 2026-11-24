"""The two RAG workflows: ingest a document, and answer a question from stored chunks."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import Chunk, Document
from app.schemas import AskIn, AskOut, Source
from app.services.chat import ChatClient
from app.services.chunking import chunk_pages
from app.services.embeddings import Embedder
from app.services.extract import extract_pages
from app.services.search import Hit, search_chunks

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You answer questions using only the numbered passages provided. "
    "Cite the passages you use with their number in brackets, like [1] or [2][3]. "
    "If the passages do not contain the answer, say that you could not find it in the "
    "documents. Never invent facts. Answer in the language of the question."
)

NOT_FOUND = "I could not find the answer in the documents."


class RagService:
    def __init__(self, settings: Settings, embedder: Embedder, chat: ChatClient) -> None:
        self._settings = settings
        self._embedder = embedder
        self._chat = chat

    async def ingest(
        self, session: AsyncSession, filename: str, data: bytes
    ) -> tuple[Document, int]:
        pages = extract_pages(filename, data)
        drafts = chunk_pages(pages, self._settings.chunk_size, self._settings.chunk_overlap)
        vectors = await self._embedder.embed([d.text for d in drafts])

        document = Document(
            filename=filename, pages=len(pages), embedding_model=self._embedder.name
        )
        document.chunks = [
            Chunk(page=d.page, position=d.position, text=d.text, embedding=vector)
            for d, vector in zip(drafts, vectors, strict=True)
        ]
        session.add(document)
        await session.commit()
        logger.info("Ingested %s: %d pages, %d chunks", filename, len(pages), len(drafts))
        return document, len(drafts)

    async def ask(self, session: AsyncSession, request: AskIn) -> AskOut:
        [query_vector] = await self._embedder.embed([request.question])
        hits = await search_chunks(
            session,
            query_vector,
            embedding_model=self._embedder.name,
            top_k=request.top_k or self._settings.default_top_k,
            min_score=self._settings.min_score,
            document_ids=request.document_ids,
        )

        # Guardrail: with nothing relevant, don't ask the model (it would guess).
        if not hits:
            return AskOut(answer=NOT_FOUND, found=False, sources=[])

        answer = await self._chat.answer(SYSTEM_PROMPT, build_user_prompt(request.question, hits))
        return AskOut(answer=answer, found=True, sources=[_to_source(hit) for hit in hits])


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    passages = "\n".join(
        f"[{i}] ({hit.filename}, p. {hit.chunk.page}): {hit.chunk.text}"
        for i, hit in enumerate(hits, start=1)
    )
    return f"Passages:\n{passages}\n\nQuestion: {question}"


def _to_source(hit: Hit) -> Source:
    text = hit.chunk.text
    excerpt = text if len(text) <= 240 else text[:240].rsplit(" ", 1)[0] + "..."
    return Source(
        document_id=hit.chunk.document_id,
        filename=hit.filename,
        page=hit.chunk.page,
        score=round(hit.score, 3),
        excerpt=excerpt,
    )
