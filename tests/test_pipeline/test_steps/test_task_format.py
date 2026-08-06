import pytest
from jinja2 import Environment, FileSystemLoader

from open_fin_gym.pipeline.db.tables import TaskType
from open_fin_gym.pipeline.steps.task_extraction.prompts import Dataset
from open_fin_gym.pipeline.steps.task_generator.prompts import (
    TaskSpecification,
    build_metric_prompt,
)


def dataset(name: str) -> Dataset:
    return Dataset(
        name=name,
        filename=f"{name}.csv",
        description=f"The {name} dataset",
        source="paper",
        relevant_urls=[],
        download_link=None,
    )


@pytest.fixture
def instructions_template():
    return Environment(loader=FileSystemLoader("templates")).get_template(
        "instructions.md.j2"
    )


def render(instructions_template, test_inputs: list[Dataset]) -> str:
    return instructions_template.render(
        task_description="A task",
        training_inputs=[dataset("train")],
        training_targets=[],
        test_inputs=test_inputs,
        test_outputs=[dataset("output")],
        metrics=[],
    )


def test_instructions_condition_on_test_inputs(instructions_template) -> None:
    instructions = render(instructions_template, [dataset("x_test")])

    assert "### Test Inputs" in instructions
    assert "conditioned on the test-inputs" in instructions


def test_instructions_omit_absent_test_inputs(instructions_template) -> None:
    # A generation task samples its output, so has nothing to condition on
    instructions = render(instructions_template, [])

    assert "### Test Inputs" not in instructions
    assert "conditioned on the test-inputs" not in instructions
    assert "/logs/artifacts/" in instructions


def spec(task_type: TaskType) -> TaskSpecification:
    return TaskSpecification(
        id=1,
        task_name="task",
        task_type=task_type,
        task_description="A task",
        training_inputs=[dataset("train")],
        training_targets=[],
        test_inputs=[],
        test_targets=[dataset("target")],
        test_outputs=[dataset("output")],
        metrics=[],
    )


@pytest.mark.parametrize(
    "task_type,expected",
    [
        (TaskType.GENERATION, "as distributions"),
        (TaskType.FORECASTING, "corresponds to the same row"),
    ],
)
def test_metric_prompt_matches_task_type(task_type: TaskType, expected: str) -> None:
    assert expected in build_metric_prompt(spec(task_type))
