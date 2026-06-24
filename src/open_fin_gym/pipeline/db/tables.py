from enum import Enum as EnumType
from typing import Any, List

from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.sqlite.json import JSON
from sqlalchemy.orm import (
    Mapped,
    declarative_base,
    mapped_column,
    relationship,
)

Base = declarative_base()


class JudgeLabel(str, EnumType):
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


class PaperStatus(str, EnumType):
    SCRAPED = "scraped"
    EXTRACTED = "extracted"
    ERRORED = "errored"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMPLETE = "complete"


class SourceName(str, EnumType):
    ARXIV = "arxiv"
    MANUAL = "manual"


class RejectionReason(str, EnumType):
    NoPaperURL = "no_paper_url"
    RetrievalError = "retrieval_error"
    PreFiltered = "pre_filtered"
    JudgeRejected = "judge_rejected"
    JudgeCutoff = "judge_cutoff"
    LLMError = "llm_error"


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


class TaskType(str, EnumType):
    forecasting = "forecasting"
    generative = "generative"


class TaskCandidate(Base):
    __tablename__ = "task_candidates"
    task_id = Column(Integer, primary_key=True, autoincrement=True)
    scope_id = Column(String, index=True)
    paper_id = Column(String, index=True)
    new = Column(Boolean, default=True)
    task_name = Column(String)
    ml_task_summary = Column(String)
    experiments = Column(String)
    links = Column(JSON)
    task_family = Column(Enum(TaskType))
    datasets: Mapped[List["DatasetCandidate"]] = relationship(
        "DatasetCandidate", backref="task_candidate"
    )
    metrics: Mapped[List["MetricCandidate"]] = relationship(
        "MetricCandidate", backref="task_candidate"
    )


class DatasetType(str, EnumType):
    real = "real"
    synthetic = "synthetic"


class DatasetCandidate(Base):
    __tablename__ = "dataset_candidates"
    dataset_id = Column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("task_candidates.task_id"))
    name = Column(String)
    description = Column(String)
    dataset_type = Column(Enum(DatasetType))


class MetricCandidate(Base):
    __tablename__ = "metric_candidates"
    metric_id = Column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("task_candidates.task_id"))
    name = Column(String)
    description = Column(String)
