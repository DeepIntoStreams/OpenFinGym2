from pydantic import BaseModel

from open_fin_gym.pipeline.steps.task_extraction.prompts import Dataset
from open_fin_gym.pipeline.steps.task_extraction.utils import TaskSpecification


class DatasetRetrieval(BaseModel):
    training_script: str
    testing_script: str
    requirements: list[str]


class Assessment(BaseModel):
    assessment_script: str
    requirements: list[str]


def join_datasets(datasets: list[Dataset]) -> str:
    return "\n\t".join([f"- {x.model_dump()}" for x in datasets])


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
    metrics = join_datasets(task_spec.metrics)
    return f"""
You are given the specification of a machine learning task. You should write
a Python script that assess the test_output produced by the user against the
target test dataset. The script should retrieve the users output, and the test
target data, and apply the set of assessment metrics. It should then save the
results as a json file containing a dictionary of individual metric results. The
results should be written to `/logs/verifier/reward.json`.

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
