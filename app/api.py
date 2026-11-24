from collections.abc import AsyncIterator
from pathlib import PurePath
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document
from app.schemas import AskIn, AskOut, DocumentCreated, DocumentOut
from app.services.extract import ExtractionError, UnsupportedFileError
from app.services.openai_http import LLMError
from app.services.rag import RagService

router = APIRouter()


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.db.session_factory() as session:
        yield session


def get_rag(request: Request) -> RagService:
    return request.app.state.rag


SessionDep = Annotated[AsyncSession, Depends(get_session)]
RagDep = Annotated[RagService, Depends(get_rag)]


@router.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/documents",
    response_model=DocumentCreated,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
    summary="Upload a PDF, .txt or .md file and index it",
)
async def upload_document(
    request: Request,
    session: SessionDep,
    rag: RagDep,
    file: Annotated[UploadFile, File()],
) -> DocumentCreated:
    max_bytes = request.app.state.settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, "File too large")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")

    # Keep only the base name: never trust a path sent by the client.
    filename = PurePath(file.filename or "document").name

    try:
        document, chunk_count = await rag.ingest(session, filename, data)
    except UnsupportedFileError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except ExtractionError as exc:
        raise HTTPException(422, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    return DocumentCreated(
        id=document.id,
        filename=document.filename,
        pages=document.pages,
        created_at=document.created_at,
        chunks=chunk_count,
    )


@router.get("/documents", response_model=list[DocumentOut], tags=["documents"])
async def list_documents(session: SessionDep) -> list[Document]:
    result = await session.execute(select(Document).order_by(Document.created_at.desc()))
    return list(result.scalars())


@router.delete(
    "/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"]
)
async def delete_document(document_id: str, session: SessionDep) -> None:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    await session.delete(document)
    await session.commit()


@router.post("/ask", response_model=AskOut, tags=["questions"])
async def ask(question: AskIn, session: SessionDep, rag: RagDep) -> AskOut:
    try:
        return await rag.ask(session, question)
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
