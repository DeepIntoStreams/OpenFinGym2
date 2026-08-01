from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from open_fin_gym.pipeline.config import Scope
from open_fin_gym.pipeline.db.tables import (
    Base,
    Chunk,
    JudgeLabel,
    Paper,
    PaperStatus,
    RejectionReason,
    SourceName,
)
from open_fin_gym.pipeline.steps.judge.config import JudgeConfig
from open_fin_gym.pipeline.steps.judge.pipeline import Judge
from open_fin_gym.pipeline.steps.judge.prompts import Evidence, SiftJudgement


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def scope():
    return Scope(id="s1", name="S1", task_type="forecasting", description="")


def make_paper(db, scope_id, paper_id="p1"):
    with Session(db) as session:
        session.add(
            Paper(
                paper_id=paper_id,
                scope_id=scope_id,
                title="t",
                abstract="a",
                source=SourceName.ARXIV,
                authors=[],
                categories=[],
                citation_count=0,
                influential_citation_count=0,
                peer_reviewed=False,
                prefilter_score=8.0,
                prefilter_passed=True,
                status=PaperStatus.EXTRACTED,
            )
        )
        session.add(
            Chunk(paper_id=paper_id, chunk_index=0, header="Data", text="excerpt")
        )
        session.commit()


def make_judge(db, threshold=7.0):
    cfg = JudgeConfig(
        sift_budget=5,
        prefilter_enabled=False,
        prefilter_llm_score_min=0.0,
        prefilter_llm_confidence_min=0.0,
        ranking_citation_boost=0.5,
        ranking_recency_weight=0.1,
        ranking_recency_half_life_days=365,
        threshold_default=threshold,
        llm=None,
    )
    judge = Judge.__new__(Judge)  # bypass __init__ so we skip real LLM instantiation
    judge.db = db
    judge.cfg = cfg
    judge.llm = MagicMock()
    return judge


def _get_paper(db, paper_id="p1"):
    with Session(db) as session:
        return session.execute(
            select(Paper).where(Paper.paper_id == paper_id)
        ).scalar_one()


def test_rejects_when_data_not_public(db, scope, tmp_path):
    make_paper(db, scope.id)
    judge = make_judge(db)
    judge.llm.with_structured_output.return_value.invoke.return_value = SiftJudgement(
        evidence=Evidence(experiments="e", datasets="d", metrics="m"),
        data_publicly_available=JudgeLabel.REJECTED,
        data_availability_reasoning="proprietary broker feed, no public source",
        reasons="otherwise relevant",
        label=JudgeLabel.ACCEPTED,
        score=9.0,
        confidence=0.9,
    )

    judge.judge_scope(tmp_path, scope)

    paper = _get_paper(db)
    assert paper.status == PaperStatus.REJECTED
    assert paper.rejection_reason == RejectionReason.DataNotPublic


def test_accepts_when_data_public_and_score_high(db, scope, tmp_path):
    make_paper(db, scope.id)
    judge = make_judge(db)
    judge.llm.with_structured_output.return_value.invoke.return_value = SiftJudgement(
        evidence=Evidence(experiments="e", datasets="d", metrics="m"),
        data_publicly_available=JudgeLabel.ACCEPTED,
        data_availability_reasoning="reconstructible via yfinance",
        reasons="good paper",
        label=JudgeLabel.ACCEPTED,
        score=9.0,
        confidence=0.9,
    )

    judge.judge_scope(tmp_path, scope)

    paper = _get_paper(db)
    assert paper.status == PaperStatus.ACCEPTED
    assert paper.rejection_reason is None


def test_llm_error_does_not_crash_and_sets_llm_error_reason(db, scope, tmp_path):
    make_paper(db, scope.id)
    judge = make_judge(db)
    judge.llm.with_structured_output.return_value.invoke.side_effect = RuntimeError(
        "boom"
    )

    judge.judge_scope(tmp_path, scope)  # must not raise

    paper = _get_paper(db)
    assert paper.status == PaperStatus.REJECTED
    assert paper.rejection_reason == RejectionReason.LLMError
