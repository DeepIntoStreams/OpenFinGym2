from enum import Enum as EnumType

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.sqlite.json import JSON
from sqlalchemy.orm import declarative_base

from open_fin_gym.pipeline.steps.scrape_papers.types import (
    PaperStatus,
    SourceName,
)

Base = declarative_base()


class RejectionReason(str, EnumType):
    NoPaperURL = "no_paper_url"


class Paper(Base):
    __tablename__ = "papers"
    paper_id = Column(String, primary_key=True)
    scope_id = Column(String, primary_key=True)
    title = Column(String)
    abstract = Column(String)
    source = Column(Enum(SourceName))
    authors = Column(JSON)
    categories = Column(JSON)
    published_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)
    arxiv_url = Column(String, nullable=True)
    pdf_url = Column(String, nullable=True)
    primary_category = Column(String, nullable=True)
    doi = Column(String, nullable=True)
    citation_count = Column(Integer)
    semantic_scholar_id = Column(String, nullable=True)
    influential_citation_count = Column(Integer)
    venue = Column(String, nullable=True)
    journal_name = Column(String, nullable=True)
    publication_types = Column(JSON)
    peer_reviewed = Column(Boolean)
    prefilter_score = Column(Float)
    prefilter_passed = Column(Boolean)
    status = Column(Enum(PaperStatus), index=True)
    rejection_reason = Column(Enum(RejectionReason), nullable=True)
    __table_args__ = (
        UniqueConstraint("paper_id", "scope_id", name="paper_constraint"),
    )


class Chunk(Base):
    __tablename__ = "chunks"
    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(String, index=True)
    chunk_index = Column(Integer, index=True)
    header = Column(String)
    text = Column(String)
