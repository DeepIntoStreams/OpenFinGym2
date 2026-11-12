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
from open_fin_gym.pipeline.steps.task_generator.prompts import (
    CodeFormattingError,
    CodeTestError,
    get_dataset_script_model,
)


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


def test_pydantic_pylint(db: Engine, llm_config: DictConfig) -> None:

    config = TaskGenerationConfig(llm_config, templates_path="./templates")
    generator = TaskGenerator(db, config)
    model = get_dataset_script_model(
        ["foo.txt"],
        ["bar.txt"],
        generator.train_docker_template,
        generator.test_docker_template,
        generator.test_sh_template,
    )

    good_train_code_string = """with open('foo.txt', 'w') as f:\n\tf.write('OK')"""
    good_test_code_string = """with open('bar.txt', 'w') as f:\n\tf.write('OK')"""

    model(
        training_script=good_train_code_string,
        testing_script=good_test_code_string,
        requirements_file="",
    )

    with pytest.raises(CodeFormattingError):
        bad_train_code_string = """with open('foo.txt', 'w'):\nf.write('OK')"""
        model(
            training_script=bad_train_code_string,
            testing_script=good_test_code_string,
            requirements_file="",
        )


@pytest.mark.parametrize(
    "script, expected",
    [
        ("with open('foo.txt', 'w') as f:\n\tf.write('Hello World')", True),
        ("with open('bar.txt', 'w') as f:\n\tf.write('Hello World')", False),
    ],
)
def test_tools_code_check_tools_train_data(
    db: Engine, llm_config: DictConfig, script: str, expected: bool
) -> None:

    config = TaskGenerationConfig(llm_config, templates_path="./templates")
    generator = TaskGenerator(db, config)
    model = get_dataset_script_model(
        ["foo.txt"],
        ["bar.txt"],
        generator.train_docker_template,
        generator.test_docker_template,
        generator.test_sh_template,
    )

    good_test_code_string = """with open('bar.txt', 'w') as f:\n\tf.write('OK')"""

    if expected:
        model(
            training_script=script,
            testing_script=good_test_code_string,
            requirements_file="",
        )
    else:
        with pytest.raises(CodeTestError):
            model(
                training_script=script,
                testing_script=good_test_code_string,
                requirements_file="",
            )
