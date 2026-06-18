from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from open_fin_gym.pipeline.steps.scrape_papers.collection.config import (
    ArxivConfig,
    CrossrefConfig,
    SemanticScholarConfig,
)


class JudgeLabel(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class JudgeDecision(BaseModel):
    paper_id: str
    scope_id: str
    label: JudgeLabel
    score_0_10: float = Field(ge=0.0, le=10.0)
    reasons: str = ""
    confidence_0_1: float = Field(ge=0.0, le=1.0)
    model: str
    raw_response: dict[str, Any] = Field(default_factory=dict)


class PaperStatus(str, Enum):
    SCRAPED = "scraped"
    EXTRACTED = "extracted"
    ERRORED = "errored"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMPLETE = "complete"


class SourceName(str, Enum):
    ARXIV = "arxiv"
    MANUAL = "manual"


@dataclass
class Scope:
    id: str
    name: str
    description: str
    enabled: bool = True
    queries: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)


def scope_context(scope: Scope) -> str:
    query_block = "\n".join(f"- {q}" for q in scope.queries[:10]) or "- (none)"
    categories = ", ".join(scope.categories) or "(none)"
    return (
        f"Scope name: {scope.name}\n"
        f"Scope description: {scope.description}\n"
        f"Scope categories: {categories}\n"
        f"Scope queries:\n{query_block}"
    )


@dataclass
class ScrapingConfig:
    root_dir: str
    scopes: list[Scope]
    arxiv: ArxivConfig
    semantic_scholar: SemanticScholarConfig
    crossref: CrossrefConfig
    max_papers_per_scope: int
    max_accepts_per_scope: int
    since: str
    until: str
    overwrite: bool = False
    max_accepts: Optional[int] = None


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
