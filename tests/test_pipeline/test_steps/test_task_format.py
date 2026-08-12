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
        description=name,
        source="paper",
        relevant_urls=[],
        download_link=None,
    )


@pytest.mark.parametrize("test_inputs", [[dataset("x_test")], []])
def test_instructions_condition_only_on_real_test_inputs(
    test_inputs: list[Dataset],
) -> None:
    # A generation task samples its output, so it has nothing to condition on
    template = Environment(loader=FileSystemLoader("templates")).get_template(
        "instructions.md.j2"
    )
    instructions = template.render(
        task_description="A task",
        training_inputs=[dataset("train")],
        training_targets=[],
        test_inputs=test_inputs,
        test_outputs=[dataset("output")],
        metrics=[],
    )

    assert ("### Test Inputs" in instructions) is bool(test_inputs)
    assert ("conditioned on the test-inputs" in instructions) is bool(test_inputs)
    assert "/logs/artifacts/" in instructions


@pytest.mark.parametrize(
    "task_type,expected",
    [
        (TaskType.GENERATION, "as distributions"),
        (TaskType.FORECASTING, "matches the same row"),
    ],
)
def test_metric_prompt_matches_task_type(task_type: TaskType, expected: str) -> None:
    spec = TaskSpecification(
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

    assert expected in build_metric_prompt(spec)
