from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    pages: int
    created_at: datetime


class DocumentCreated(DocumentOut):
    chunks: int


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["What is the notice period?"])
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[str] | None = Field(
        default=None, description="Restrict the search to these documents."
    )


class Source(BaseModel):
    document_id: str
    filename: str
    page: int
    score: float
    excerpt: str


class AskOut(BaseModel):
    answer: str
    found: bool = Field(description="False when no passage was relevant enough to answer.")
    sources: list[Source]
