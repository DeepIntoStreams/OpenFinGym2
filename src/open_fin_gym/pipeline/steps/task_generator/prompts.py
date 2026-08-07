from pydantic import BaseModel, Field

from open_fin_gym.pipeline.db.tables import (
    BaseDatasetCandidate,
    MetricCandidate,
    TaskType,
)
from open_fin_gym.pipeline.steps.task_extraction.prompts import (
    AssessmentMetric,
    Dataset,
)


class TaskSpecification(BaseModel):
    id: int = Field(description="Task id")
    task_name: str = Field(description="Name id assigned to the task")
    task_type: TaskType = Field(description="Whether outputs are predicted or sampled")
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


class DatasetRetrieval(BaseModel):
    training_script: str
    testing_script: str
    requirements: list[str]


class Assessment(BaseModel):
    assessment_script: str
    requirements: list[str]


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


def join_datasets(datasets: list[Dataset]) -> str:
    return "\n\t".join([f"- {x.model_dump()}" for x in datasets])


def join_metrics(metrics: list[AssessmentMetric]) -> str:
    return "\n\t".join([f"- {x.model_dump()}" for x in metrics])


def build_dataset_download_prompt(task_spec: TaskSpecification) -> str:

    training_inputs = join_datasets(task_spec.training_inputs)
    training_targets = join_datasets(task_spec.training_targets)
    test_inputs = join_datasets(task_spec.test_inputs)
    test_targets = join_datasets(task_spec.test_targets)

    return f"""
You are given the specification of a machine learning task. You should write
two Python scripts to retrieve and save the datasets used as part of the
assessment.

1. training_script: A script that retrieves and saves the training_inputs, training_targets, and test_inputs datasets
2. testing_script: A script that retrieves and saves the test_targets dataset(s)

You should also produce a list of the Python requirements required by the scripts

Notes:

- The datasets should be written to the files assigned in the specification
- The files should be written to the same folder as the script
- If any random sampling is used, the process should be seeded
- A dataset category may be empty, in which case skip it rather than writing a placeholder file

Specification:

Task name: {task_spec.task_name}
Task description: {task_spec.task_description}
Training input datasets:
    {training_inputs}
Training target / ground-truth datasets:
    {training_targets}
Test input datasets:
    {test_inputs}
Test target datasets:
    {test_targets}
"""


def build_metric_prompt(task_spec: TaskSpecification) -> str:
    test_outputs = join_datasets(task_spec.test_outputs)
    test_targets = join_datasets(task_spec.test_targets)
    metrics = join_metrics(task_spec.metrics)
    comparison = (
        "The user samples their output rather than predicting it row by row, so the "
        "metrics compare the two datasets as distributions and must not assume the rows "
        "are paired."
        if task_spec.task_type == TaskType.GENERATION
        else "Each row of the user output corresponds to the same row of the test target."
    )
    return f"""
You are given the specification of a machine learning task. You should write
a Python script that assess the test_output produced by the user against the
target test dataset. The script should retrieve the users output, and the test
target data, and apply the set of assessment metrics. It should then save the
results as a json file containing a dictionary of individual metric results. The
results should be written to `/logs/verifier/reward.json`.

{comparison}

You should also produce a list of Python requirements required by the assessment script.

Specification:

Task name: {task_spec.task_name}
Task description: {task_spec.task_description}
Test outputs (produced by the user):
    {test_outputs}
Test targets/ground-truth:
    {test_targets}
Assessment metrics:
    {metrics}
"""


def build_description_summary_prompt(description: str) -> str:
    return f"""
Write a short one paragraph summary of the following machine learning task:

{description}
"""


def build_difficulty_explanation_prompt(description: str) -> str:
    return f"""
Write a short one paragraph assessment of the difficulty of the following
machine learning task:

{description}
"""
