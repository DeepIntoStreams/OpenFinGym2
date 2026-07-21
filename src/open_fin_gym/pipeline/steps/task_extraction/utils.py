from typing import Type

from open_fin_gym.pipeline.db.tables import (
    MetricCandidate,
    TaskCandidate,
    TestInputDatasetCandidate,
    TestOutputDatasetCandidate,
    TestTargetDatasetCandidate,
    TrainInputDatasetCandidate,
    TrainTargetDatasetCandidate,
)

from .prompts import Dataset

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


def unpack_metrics(task_candidate: TaskCandidate) -> list[MetricCandidate]:
    return [
        MetricCandidate(**x.model_dump(), task_candidate=task_candidate)
        for x in task_candidate.assessment_metrics
    ]
