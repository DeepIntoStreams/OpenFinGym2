from open_fin_gym.pipeline.db.tables import (
    GroundTruthDatasetCandidate,
    MetricCandidate,
    TaskCandidate,
    TestInputDatasetCandidate,
    TestOutputDatasetCandidate,
    TrainDatasetCandidate,
)


def unpack_datasets(
    task_candidate: TaskCandidate,
) -> list[
    TrainDatasetCandidate
    | TestInputDatasetCandidate
    | TestOutputDatasetCandidate
    | GroundTruthDatasetCandidate
]:
    training_data = [
        TrainDatasetCandidate(**x.model_dump(), task_candidate=task_candidate)
        for x in task_candidate.training_data
    ]
    test_input = [
        TestInputDatasetCandidate(**x.model_dump(), task_candidate=task_candidate)
        for x in task_candidate.test_input
    ]
    test_output = [
        TestOutputDatasetCandidate(**x.model_dump(), task_candidate=task_candidate)
        for x in task_candidate.test_output
    ]
    ground_truth_data = [
        GroundTruthDatasetCandidate(**x.model_dump(), task_candidate=task_candidate)
        for x in task_candidate.ground_truth_data
    ]

    return training_data + test_input + test_output + ground_truth_data


def unpack_metrics(task_candidate: TaskCandidate) -> list[MetricCandidate]:
    return [
        MetricCandidate(**x.model_dump(), task_candidate=task_candidate)
        for x in task_candidate.assessment_metrics
    ]
