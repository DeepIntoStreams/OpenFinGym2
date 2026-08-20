from pydantic import BaseModel

from open_fin_gym.pipeline.db.tables import TaskType
from open_fin_gym.pipeline.steps.task_extraction.prompts import (
    AssessmentMetric,
    Dataset,
)
from open_fin_gym.pipeline.steps.task_extraction.utils import TaskSpecification

# Whether row i of the user output corresponds to row i of the test target, which
# decides whether the verifier can compare them row by row
ROW_CORRESPONDENCE_BY_TASK_TYPE = {
    TaskType.FORECASTING: (
        "Each row of the user output matches the same row of the test target."
    ),
    TaskType.GENERATION: (
        "The user samples their output, so compare the two datasets as "
        "distributions and do not assume the rows are paired."
    ),
}

# Prompts are built before the generator's error handling, so an unhandled task type
# would abort a run that has already spent LLM calls. Fail on import instead.
if set(ROW_CORRESPONDENCE_BY_TASK_TYPE) != set(TaskType):
    raise ValueError(
        f"No row correspondence defined for task type(s) "
        f"{sorted(set(TaskType) - set(ROW_CORRESPONDENCE_BY_TASK_TYPE))}"
    )


class DatasetRetrieval(BaseModel):
    training_script: str
    testing_script: str
    requirements: list[str]


class Assessment(BaseModel):
    assessment_script: str
    requirements: list[str]


def join_specs(specs: list[Dataset] | list[AssessmentMetric]) -> str:
    return "\n\t".join([f"- {x.model_dump()}" for x in specs])


def build_dataset_download_prompt(task_spec: TaskSpecification) -> str:

    training_inputs = join_specs(task_spec.training_inputs)
    training_targets = join_specs(task_spec.training_targets)
    test_inputs = join_specs(task_spec.test_inputs)
    test_targets = join_specs(task_spec.test_targets)

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
    test_outputs = join_specs(task_spec.test_outputs)
    test_targets = join_specs(task_spec.test_targets)
    metrics = join_specs(task_spec.metrics)
    row_correspondence = ROW_CORRESPONDENCE_BY_TASK_TYPE[task_spec.task_type]
    return f"""
You are given the specification of a machine learning task. You should write
a Python script that assess the test_output produced by the user against the
target test dataset. The script should retrieve the users output, and the test
target data, and apply the set of assessment metrics. It should then save the
results as a json file containing a dictionary of individual metric results. The
results should be written to `/logs/verifier/reward.json`.

{row_correspondence}

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
