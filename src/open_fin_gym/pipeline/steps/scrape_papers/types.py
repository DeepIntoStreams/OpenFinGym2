from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from open_fin_gym.pipeline.db.tables import PaperStatus, SourceName


class PaperRecord(BaseModel):
    paper_id: str
    scope_id: str
    title: str
    abstract: str
    source: SourceName = SourceName.ARXIV
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    updated_at: datetime | None = None
    arxiv_url: Optional[str] = None
    pdf_url: str | None = None
    primary_category: str | None = None
    doi: str | None = None
    citation_count: int | None = None
    semantic_scholar_id: str | None = None
    influential_citation_count: int | None = None
    venue: str | None = None
    journal_name: str | None = None
    publication_types: list[str] = Field(default_factory=list)
    peer_reviewed: bool = False
    prefilter_score: float = 0.0
    prefilter_passed: bool = False
    status: PaperStatus = PaperStatus.SCRAPED
