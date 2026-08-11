from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from open_fin_gym.pipeline.config import Scope
from open_fin_gym.pipeline.db.tables import (
    Base,
    JudgeLabel,
    MetricCandidate,
    TaskCandidate,
    TaskCandidateStatus,
    TestInputDatasetCandidate,
    TestOutputDatasetCandidate,
    TestTargetDatasetCandidate,
    TrainInputDatasetCandidate,
    TrainTargetDatasetCandidate,
)
from open_fin_gym.pipeline.steps.task_critic.config import TaskCriticConfig
from open_fin_gym.pipeline.steps.task_critic.pipeline import TaskCritic
from open_fin_gym.pipeline.steps.task_critic.prompts import TaskCritique


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def scope():
    return Scope(id="s1", name="S1", task_type="forecasting", description="")


def _dataset(cls, name: str, filename: str):
    return cls(
        name=name,
        filename=filename,
        description="d",
        source="s",
        relevant_urls=[],
        download_link=None,
    )


def make_task_candidate(
    db, scope_id: str, paper_id: str = "p1", *, include_test_outputs: bool = True
) -> int:
    with Session(db) as session:
        candidate = TaskCandidate(
            scope_id=scope_id,
            paper_id=paper_id,
            task_name="task",
            description="desc",
            status=TaskCandidateStatus.NEW,
        )
        candidate.training_inputs.append(
            _dataset(TrainInputDatasetCandidate, "train_x", "train_x.csv")
        )
        candidate.training_targets.append(
            _dataset(TrainTargetDatasetCandidate, "train_y", "train_y.csv")
        )
        candidate.test_inputs.append(
            _dataset(TestInputDatasetCandidate, "test_x", "test_x.csv")
        )
        candidate.test_targets.append(
            _dataset(TestTargetDatasetCandidate, "test_y", "test_y.csv")
        )
        if include_test_outputs:
            candidate.test_outputs.append(
                _dataset(TestOutputDatasetCandidate, "pred_y", "pred_y.csv")
            )
        candidate.assessment_metrics.append(
            MetricCandidate(
                name="mse",
                description="mean squared error",
                input_datasets=["pred_y.csv", "test_y.csv"],
            )
        )
        session.add(candidate)
        session.commit()
        return candidate.task_id


def make_critic(db, threshold: float = 6.0) -> TaskCritic:
    cfg = TaskCriticConfig(threshold_default=threshold, llm=None)
    critic = TaskCritic.__new__(
        TaskCritic
    )  # bypass __init__, skip real LLM instantiation
    critic.db = db
    critic.cfg = cfg
    critic.llm = MagicMock()
    return critic


def _get_status(db, task_id: int) -> TaskCandidateStatus:
    with Session(db) as session:
        return (
            session.execute(
                select(TaskCandidate).where(TaskCandidate.task_id == task_id)
            )
            .scalar_one()
            .status
        )


def test_approves_well_formed_candidate(db, scope, tmp_path):
    task_id = make_task_candidate(db, scope.id)
    critic = make_critic(db)
    critic.llm.with_structured_output.return_value.invoke.return_value = TaskCritique(
        consistency_assessment="matches description",
        completeness_assessment="datasets are specific",
        issues="",
        label=JudgeLabel.ACCEPTED,
        score=8.0,
        confidence=0.9,
    )

    critic.critique_scope(tmp_path, scope)

    assert _get_status(db, task_id) == TaskCandidateStatus.APPROVED


def test_rejects_when_llm_labels_rejected(db, scope, tmp_path):
    task_id = make_task_candidate(db, scope.id)
    critic = make_critic(db)
    critic.llm.with_structured_output.return_value.invoke.return_value = TaskCritique(
        consistency_assessment="description doesn't match datasets",
        completeness_assessment="vague",
        issues="dataset period unspecified",
        label=JudgeLabel.REJECTED,
        score=3.0,
        confidence=0.8,
    )

    critic.critique_scope(tmp_path, scope)

    assert _get_status(db, task_id) == TaskCandidateStatus.REJECTED


def test_rejects_when_score_below_threshold_despite_accepted_label(db, scope, tmp_path):
    task_id = make_task_candidate(db, scope.id)
    critic = make_critic(db, threshold=6.0)
    critic.llm.with_structured_output.return_value.invoke.return_value = TaskCritique(
        consistency_assessment="mostly fine",
        completeness_assessment="borderline",
        issues="minor ambiguity in test window",
        label=JudgeLabel.ACCEPTED,
        score=5.0,
        confidence=0.6,
    )

    critic.critique_scope(tmp_path, scope)

    assert _get_status(db, task_id) == TaskCandidateStatus.REJECTED


def test_llm_error_does_not_crash_and_fails_closed(db, scope, tmp_path):
    task_id = make_task_candidate(db, scope.id)
    critic = make_critic(db)
    critic.llm.with_structured_output.return_value.invoke.side_effect = RuntimeError(
        "boom"
    )

    critic.critique_scope(tmp_path, scope)  # must not raise

    assert _get_status(db, task_id) == TaskCandidateStatus.REJECTED


def test_structural_issue_short_circuits_llm_call(db, scope, tmp_path):
    task_id = make_task_candidate(db, scope.id, include_test_outputs=False)
    critic = make_critic(db)

    critic.critique_scope(tmp_path, scope)

    critic.llm.with_structured_output.assert_not_called()
    assert _get_status(db, task_id) == TaskCandidateStatus.REJECTED
