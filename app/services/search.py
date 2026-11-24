"""Vector search: cosine similarity between the question and stored chunks.

Similarity is computed with NumPy in the application. That is simple and portable
(SQLite or PostgreSQL) and fast enough for thousands of chunks. For larger volumes the
next step is a vector index in the database (pgvector).
"""

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    filename: str
    score: float


async def search_chunks(
    session: AsyncSession,
    query_vector: list[float],
    embedding_model: str,
    top_k: int,
    min_score: float,
    document_ids: list[str] | None = None,
) -> list[Hit]:
    query = (
        select(Chunk, Document.filename)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.embedding_model == embedding_model)
    )
    if document_ids:
        query = query.where(Document.id.in_(document_ids))
    rows = (await session.execute(query)).all()
    if not rows:
        return []

    matrix = np.array([chunk.embedding for chunk, _ in rows], dtype=np.float32)
    question = np.array(query_vector, dtype=np.float32)
    scores = _cosine(matrix, question)

    best = np.argsort(-scores)[:top_k]
    return [
        Hit(chunk=rows[i][0], filename=rows[i][1], score=float(scores[i]))
        for i in best
        if scores[i] >= min_score
    ]


def _cosine(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    norms[norms == 0] = 1.0  # avoid dividing by zero for empty vectors
    return (matrix @ vector) / norms
