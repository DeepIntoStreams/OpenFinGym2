from typing import Type

from pydantic import BaseModel, Field

from open_fin_gym.pipeline.db.tables import BaseDatasetCandidate  # New
from open_fin_gym.pipeline.db.tables import (
    MetricCandidate,
    TaskCandidate,
    TestInputDatasetCandidate,
    TestOutputDatasetCandidate,
    TestTargetDatasetCandidate,
    TrainInputDatasetCandidate,
    TrainTargetDatasetCandidate,
)

from .prompts import AssessmentMetric, Dataset

type DatasetType = (
    TrainInputDatasetCandidate
    | TrainTargetDatasetCandidate
    | TestInputDatasetCandidate
    | TestOutputDatasetCandidate
    | TestTargetDatasetCandidate
)


def unpack_datasets(
    dataset_type: Type[DatasetType],
    task_candidate: TaskCandidate,
    datasets: list[Dataset],
) -> list[DatasetType]:

    return [
        dataset_type(
            name=x.name.lower().strip().replace(" ", "_"),
            filename=x.filename,
            description=x.description,
            source=x.source,
            relevant_urls=x.relevant_urls,
            download_link=x.download_link,
            task_candidate=task_candidate,
        )
        for x in datasets
    ]


def unpack_metrics(
    task_candidate: TaskCandidate, metrics: list[AssessmentMetric]
) -> list[MetricCandidate]:
    return [
        MetricCandidate(**x.model_dump(), task_candidate=task_candidate)
        for x in metrics
    ]


def convert_dataset(d: BaseDatasetCandidate) -> Dataset:
    return Dataset(
        name=d.name,
        filename=d.filename,
        description=d.description,
        source=d.source,
        relevant_urls=d.relevant_urls,
        download_link=d.download_link,
    )


def convert_metric(d: MetricCandidate) -> AssessmentMetric:
    return AssessmentMetric(
        name=d.name,
        description=d.description,
        input_datasets=d.input_datasets,
    )


class TaskSpecification(BaseModel):
    id: int = Field(description="Task id")
    task_name: str = Field(description="Name id assigned to the task")
    task_description: str = Field(
        description="Description of the ML task including the data and how it is assessed"
    )
    training_inputs: list[Dataset] = Field(
        description="List of training input datasets"
    )
    training_targets: list[Dataset] = Field(
        description="List of training ground-truth/target datasets"
    )
    test_inputs: list[Dataset] = Field(description="List of test input datasets")
    test_targets: list[Dataset] = Field(
        description="List of ground truth datasets used to assess the users output against"
    )
    test_outputs: list[Dataset] = Field(description="List of expected user outputs")
    metrics: list[AssessmentMetric] = Field(description="List of assessment metrics")


def build_task_specification(task: TaskCandidate) -> TaskSpecification:
    return TaskSpecification(
        id=task.task_id,
        task_name=task.task_name,
        task_description=task.description,
        training_inputs=[convert_dataset(x) for x in task.training_inputs],
        training_targets=[convert_dataset(x) for x in task.training_targets],
        test_inputs=[convert_dataset(x) for x in task.test_inputs],
        test_targets=[convert_dataset(x) for x in task.test_targets],
        test_outputs=[convert_dataset(x) for x in task.test_outputs],
        metrics=[convert_metric(x) for x in task.assessment_metrics],
    )
