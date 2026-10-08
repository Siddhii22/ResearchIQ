"""Pydantic request and response schemas for research queries."""

from pydantic import BaseModel, Field, field_validator

from app.config import settings


class ResearchQueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=settings.final_top_k, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Question must not be blank.")
        return question


class ResearchSourceResponse(BaseModel):
    citation_id: str
    chunk_id: str
    paper_id: str
    title: str
    authors: list[str]
    section: str
    page: int
    text: str
    similarity_score: float
    retrieval_score: float | None = None
    retrieval_method: str = "dense"


class ResearchQueryResponse(BaseModel):
    answer: str
    sources: list[ResearchSourceResponse]