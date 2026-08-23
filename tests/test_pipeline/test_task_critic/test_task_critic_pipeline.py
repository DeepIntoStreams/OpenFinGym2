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
    TaskType,
    TestInputDatasetCandidate,
    TestOutputDatasetCandidate,
    TestTargetDatasetCandidate,
    TrainInputDatasetCandidate,
    TrainTargetDatasetCandidate,
)
from open_fin_gym.pipeline.steps.task_critic.config import TaskCriticConfig
from open_fin_gym.pipeline.steps.task_critic.pipeline import TaskCritic
from open_fin_gym.pipeline.steps.task_critic.prompts import TaskCritique
from open_fin_gym.pipeline.steps.task_extraction.utils import (
    build_task_specification,
)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def scope():
    return Scope(id="s1", name="S1", task_type="forecasting", description="")


def _dataset(
    cls,
    name: str,
    filename: str,
    *,
    source: str | None = "s",
    download_link: str | None = None,
):
    return cls(
        name=name,
        filename=filename,
        description="d",
        source=source,
        relevant_urls=[],
        download_link=download_link,
    )


def make_task_candidate(
    db,
    scope_id: str,
    paper_id: str = "p1",
    *,
    task_type: TaskType = TaskType.FORECASTING,
    include_test_outputs: bool = True,
    missing_source_field: str | None = None,  # new
) -> int:
    with Session(db) as session:
        candidate = TaskCandidate(
            scope_id=scope_id,
            paper_id=paper_id,
            task_name="task",
            task_type=task_type,
            description="desc",
            status=TaskCandidateStatus.NEW,
        )

        def maybe_broken(field_name, cls, name, filename):
            if field_name == missing_source_field:
                return _dataset(cls, name, filename, source="", download_link=None)
            return _dataset(cls, name, filename)

        candidate.training_inputs.append(
            maybe_broken(
                "training_inputs", TrainInputDatasetCandidate, "train_x", "train_x.csv"
            )
        )
        candidate.training_targets.append(
            maybe_broken(
                "training_targets",
                TrainTargetDatasetCandidate,
                "train_y",
                "train_y.csv",
            )
        )
        candidate.test_inputs.append(
            maybe_broken(
                "test_inputs", TestInputDatasetCandidate, "test_x", "test_x.csv"
            )
        )
        candidate.test_targets.append(
            maybe_broken(
                "test_targets", TestTargetDatasetCandidate, "test_y", "test_y.csv"
            )
        )
        if include_test_outputs:
            candidate.test_outputs.append(
                maybe_broken(
                    "test_outputs", TestOutputDatasetCandidate, "pred_y", "pred_y.csv"
                )
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
        data_availability_assessment="datasets publicly available",
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
        data_availability_assessment="datasets proprietary and not reconstructable",
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
        data_availability_assessment="dataset proprietary but reconstructable",
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


def test_missing_source_and_download_link_short_circuits_llm_call(db, scope, tmp_path):
    task_id = make_task_candidate(db, scope.id, missing_source_field="training_inputs")
    critic = make_critic(db)

    critic.critique_scope(tmp_path, scope)

    critic.llm.with_structured_output.assert_not_called()
    assert _get_status(db, task_id) == TaskCandidateStatus.REJECTED


def test_critique_task_records_score_and_confidence(db, scope, tmp_path):
    make_task_candidate(db, scope.id)
    critic = make_critic(db)
    critic.llm.with_structured_output.return_value.invoke.return_value = TaskCritique(
        consistency_assessment="fine",
        completeness_assessment="fine",
        data_availability_assessment="fine",
        issues="",
        label=JudgeLabel.ACCEPTED,
        score=8.0,
        confidence=0.9,
    )

    with Session(db) as session:
        candidate = session.execute(select(TaskCandidate)).scalar_one()
        spec = build_task_specification(candidate)

    result = critic.critique_task(scope, spec)

    assert result.score == 8.0
    assert result.confidence == 0.9


def test_critique_task_score_and_confidence_are_none_on_structural_reject(
    db, scope, tmp_path
):
    make_task_candidate(db, scope.id, include_test_outputs=False)
    critic = make_critic(db)

    with Session(db) as session:
        candidate = session.execute(select(TaskCandidate)).scalar_one()
        spec = build_task_specification(candidate)

    result = critic.critique_task(scope, spec)

    assert result.score is None
    assert result.confidence is None
