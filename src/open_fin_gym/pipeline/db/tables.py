from enum import StrEnum
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


class TaskType(StrEnum):
    FORECASTING = "forecasting"
    GENERATION = "generation"


class JudgeLabel(StrEnum):
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


class PaperStatus(StrEnum):
    SCRAPED = "scraped"
    EXTRACTED = "extracted"
    ERRORED = "errored"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TASK_EXTRACTED = "task_extracted"
    TASK_EXTRACTION_FAILED = "task_extraction_failed"


class SourceName(StrEnum):
    ARXIV = "arxiv"
    MANUAL = "manual"


class SourceFormat(StrEnum):
    HTML = "html"
    PDF = "pdf"


class RejectionReason(StrEnum):
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
    source_format = Column(Enum(SourceFormat))


class TaskCandidateStatus(StrEnum):
    NEW = "new"
    PROCESSED = "processed"
    FAILED = "failed"


class TaskCandidate(Base):
    __tablename__ = "task_candidates"
    task_id = Column(Integer, primary_key=True, autoincrement=True)
    scope_id = Column(String, index=True)
    paper_id = Column(String, index=True)
    new = Column(Boolean, default=True)
    status = Column(Enum(TaskCandidateStatus), index=True)
    task_name = Column(String)
    description = Column(String)
    training_inputs: Mapped[List["TrainInputDatasetCandidate"]] = relationship(
        "TrainInputDatasetCandidate", backref="task_candidate"
    )
    training_targets: Mapped[List["TrainTargetDatasetCandidate"]] = relationship(
        "TrainTargetDatasetCandidate", backref="task_candidate"
    )
    test_inputs: Mapped[List["TestInputDatasetCandidate"]] = relationship(
        "TestInputDatasetCandidate", backref="task_candidate"
    )
    test_outputs: Mapped[List["TestOutputDatasetCandidate"]] = relationship(
        "TestOutputDatasetCandidate", backref="task_candidate"
    )
    test_targets: Mapped[List["TestTargetDatasetCandidate"]] = relationship(
        "TestTargetDatasetCandidate", backref="task_candidate"
    )
    assessment_metrics: Mapped[List["MetricCandidate"]] = relationship(
        "MetricCandidate", backref="task_candidate"
    )


class BaseDatasetCandidate(Base):
    __abstract__ = True
    dataset_id = Column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("task_candidates.task_id"))
    name = Column(String)
    filename = Column(String)
    description = Column(String)
    source = Column(String)
    relevant_urls = Column(JSON)
    download_link = Column(String, nullable=True)


class TrainInputDatasetCandidate(BaseDatasetCandidate):
    __tablename__ = "train_input_dataset_candidates"


class TrainTargetDatasetCandidate(BaseDatasetCandidate):
    __tablename__ = "train_target_dataset_candidates"


class TestInputDatasetCandidate(BaseDatasetCandidate):
    __tablename__ = "test_input_dataset_candidates"


class TestOutputDatasetCandidate(BaseDatasetCandidate):
    __tablename__ = "test_output_dataset_candidates"


class TestTargetDatasetCandidate(BaseDatasetCandidate):
    __tablename__ = "test_target_dataset_candidates"


class MetricCandidate(Base):
    __tablename__ = "metric_candidates"
    metric_id = Column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("task_candidates.task_id"))
    name = Column(String)
    description = Column(String)
    input_datasets = Column(JSON)


class TaskStatus(StrEnum):
    NEW = "new"
    EXPORT_FAILED = "export_failed"
    EXPORTED = "exported"


class Task(Base):
    __tablename__ = "task"
    task_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    task_candidate_id = Column(Integer, index=True)
    status = Column(Enum(TaskStatus))
    train_script = Column(String)
    test_script = Column(String)
    assessment_script = Column(String)
    requirements = Column(JSON)
    instructions = Column(String)
    short_description = Column(String)
    difficulty_explanation = Column(String)
