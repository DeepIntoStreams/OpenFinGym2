import pytest

from open_fin_gym.pipeline.db.tables import TaskType
from open_fin_gym.pipeline.steps.task_critic.utils import structural_issues
from open_fin_gym.pipeline.steps.task_extraction.prompts import (
    AssessmentMetric,
    Dataset,
)
from open_fin_gym.pipeline.steps.task_extraction.utils import TaskSpecification


def make_dataset(name: str, filename: str) -> Dataset:
    return Dataset(
        name=name,
        filename=filename,
        description="desc",
        source="src",
        relevant_urls=[],
        download_link=None,
    )


def make_spec(**overrides) -> TaskSpecification:
    defaults = dict(
        id=1,
        task_name="task",
        task_type=TaskType.FORECASTING,
        task_description="desc",
        training_inputs=[make_dataset("train_x", "train_x.csv")],
        training_targets=[make_dataset("train_y", "train_y.csv")],
        test_inputs=[make_dataset("test_x", "test_x.csv")],
        test_targets=[make_dataset("test_y", "test_y.csv")],
        test_outputs=[make_dataset("pred_y", "pred_y.csv")],
        metrics=[
            AssessmentMetric(
                name="mse",
                description="mean squared error",
                input_datasets=["pred_y.csv", "test_y.csv"],
            )
        ],
    )
    defaults.update(overrides)
    return TaskSpecification(**defaults)


def test_clean_spec_has_no_issues():
    assert structural_issues(make_spec()) == []


@pytest.mark.parametrize(
    "field",
    [
        "training_inputs",
        "training_targets",
        "test_inputs",
        "test_targets",
        "test_outputs",
    ],
)
def test_flags_empty_dataset_group(field):
    spec = make_spec(
        **{field: []},
        metrics=[
            AssessmentMetric(
                name="mse",
                description="mean squared error",
                input_datasets=[],
            )
        ],
    )
    issues = structural_issues(spec)
    assert issues == [f"{field} is empty"]


def test_flags_empty_metrics():
    issues = structural_issues(make_spec(metrics=[]))
    assert issues == ["assessment_metrics is empty"]


def test_flags_duplicate_filenames_across_roles():
    spec = make_spec(
        test_inputs=[make_dataset("test_x", "shared.csv")],
        test_targets=[make_dataset("test_y", "shared.csv")],
        metrics=[
            AssessmentMetric(
                name="mse",
                description="mean squared error",
                input_datasets=["pred_y.csv", "shared.csv"],
            )
        ],
    )
    issues = structural_issues(spec)
    assert len(issues) == 1
    assert "shared.csv" in issues[0]
    assert "duplicate" in issues[0].lower()


def test_flags_metric_referencing_unknown_dataset():
    spec = make_spec(
        metrics=[
            AssessmentMetric(
                name="mse",
                description="mean squared error",
                input_datasets=["does_not_exist.csv"],
            )
        ]
    )
    issues = structural_issues(spec)
    assert len(issues) == 1
    assert "mse" in issues[0]
    assert "does_not_exist.csv" in issues[0]


def test_generation_spec_with_empty_targets_and_inputs_has_no_issues():
    spec = make_spec(
        task_type=TaskType.GENERATION,
        training_targets=[],
        test_inputs=[],
    )
    assert structural_issues(spec) == []


def test_generation_spec_missing_required_group_is_flagged():
    spec = make_spec(task_type=TaskType.GENERATION, test_outputs=[])
    assert structural_issues(spec) == ["test_outputs is empty"]
