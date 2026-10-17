from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from omegaconf import DictConfig, OmegaConf
from sqlalchemy import Engine

from open_fin_gym.pipeline.steps.task_generator.code_checks import run_pylint
from open_fin_gym.pipeline.steps.task_generator.config import (
    TaskGenerationConfig,
)
from open_fin_gym.pipeline.steps.task_generator.pipeline import TaskGenerator


@pytest.mark.parametrize(
    "code, expected", [("print('Hello world')", True), ("x 10", False)]
)
def test_pylint_checks(code: str, expected: bool) -> None:

    with NamedTemporaryFile() as tmpfile:
        with open(tmpfile.name, "w") as f:
            f.write(code)

        path = Path(tmpfile.name)
        result, messages = run_pylint(path)

    assert expected == result


@pytest.fixture
def llm_config() -> DictConfig:
    return OmegaConf.create(
        {
            "_target_": "langchain_core.language_models.fake_chat_models.FakeChatModel",
            "responses": [],
        }
    )


def test_tools_code_check_tools_pylint(db: Engine, llm_config: DictConfig) -> None:

    config = TaskGenerationConfig(llm_config, templates_path="./templates")
    generator = TaskGenerator(db, config)

    run_pylint_checks, _, _ = generator.get_code_check_tools(["foo.txt"], ["bar.txt"])

    good_code_string = """def foo():\n\treturn 10"""
    result = run_pylint_checks(good_code_string)

    assert result[0]

    bad_code_string = "x 10"
    result = run_pylint_checks(bad_code_string)

    assert not result[0]


@pytest.mark.parametrize(
    "script, expected",
    [
        ("with open('foo.txt', 'w') as f:\n\tf.write('Hello World')", True),
        ("with open('bar.txt', 'w') as f:\n\tf.write('Hello World')", False),
        ("x 10", False),
    ],
)
def test_tools_code_check_tools_train_data(
    db: Engine, llm_config: DictConfig, script: str, expected: bool
) -> None:

    config = TaskGenerationConfig(llm_config, templates_path="./templates")
    generator = TaskGenerator(db, config)

    _, check_training_data_script, _ = generator.get_code_check_tools(
        ["foo.txt"], ["bar.txt"]
    )

    result = check_training_data_script(script, [])
    assert result[0] == expected
