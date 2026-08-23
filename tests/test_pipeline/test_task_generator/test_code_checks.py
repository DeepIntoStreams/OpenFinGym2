from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from open_fin_gym.pipeline.steps.task_generator.code_checks import run_pylint


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
