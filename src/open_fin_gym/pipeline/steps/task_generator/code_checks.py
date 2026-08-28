import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from docker.errors import BuildError
from jinja2 import Template
from pylint.lint import Run
from pylint.reporters.json_reporter import JSON2Reporter

import docker


def run_pylint(script_path: Path) -> tuple[bool, str]:
    """
    Run pylint and report any code errors or fatal issues

    Args:
        script_path: Path to Python script/module

    Returns:
    Tuple containing flag indicating failure, and list of any error messages
    """
    pylint_output = StringIO()
    reporter = JSON2Reporter(pylint_output)
    Run([str(script_path), "--disable=imports"], reporter=reporter, exit=False)
    output = pylint_output.getvalue()
    results = json.loads(output)
    counts = results["statistics"]["messageTypeCount"]

    if counts["fatal"] == 0 and counts["error"] == 0:
        return True, []

    error_messages = [
        m["message"] for m in results["messages"] if m["type"] in {"fatal", "error"}
    ]
    error_messages = "\n".join([f"- {x}" for x in error_messages])

    return False, error_messages


def test_train_download_script(
    docker_template: Template,
    download_script: str,
    requirements: list[str],
    expected_files: list[str],
) -> tuple[bool, str]:
    """
    Build train docker image and check expected data files are created

    Args:
        docker_template: Train docker image template
        download_script: Train data download Python script
        requirements: List of python requirements
        expected_files: list of expected data files

    Returns:
    Tuple containing success flag, and reasoning string
    """

    client = docker.from_env()

    with TemporaryDirectory() as folder:
        folder_path = Path(folder)
        script_path = folder_path / "data.py"

        with open(script_path, "w") as f:
            f.write(download_script)

        docker_file = docker_template.render(requirements=requirements)

        with open(folder_path / "Dockerfile", "w") as f:
            f.write(docker_file)

        try:
            image, logs = client.images.build(path=str(folder_path))
        except BuildError as be:
            return False, be.msg

    try:
        output = client.containers.run(image, command="ls -a", remove=True)
    except Exception as e:
        return False, str(e)

    files = set(output.decode("utf-8").split("\n"))

    for f in expected_files:
        if f not in files:
            return False, f"Expected train data file {f} not found in test docker image"

    return True, "Train data retrieval script successful"


def test_test_download_script(
    docker_template: Template,
    test_template: Template,
    data_download_script: str,
    verification_script: str,
    requirements: list[str],
    expected_files: list[str],
) -> tuple[bool, str]:
    """
    Build test docker image and check expected data files are created

    Args:
        docker_template: Test docker image template
        test_template: tesh.sh bash script template
        data_download_script: Test data download Python script
        verification_script: Python results verification script
        requirements: List of python requirements
        expected_files: list of expected data files

    Returns:
    Tuple containing success flag, and reasoning string
    """

    client = docker.from_env()

    with TemporaryDirectory() as folder:
        folder_path = Path(folder)
        data_script_path = folder_path / "data.py"
        verifier_script_path = folder_path / "verifier.py"

        with open(data_script_path, "w") as f:
            f.write(data_download_script)

        with open(verifier_script_path, "w") as f:
            f.write(verification_script)

        docker_file = docker_template.render(requirements=requirements)

        with open(folder_path / "Dockerfile", "w") as f:
            f.write(docker_file)

        test_sh_file = test_template.render(requirements=requirements)

        with open(folder_path / "test.sh", "w") as f:
            f.write(test_sh_file)

        try:
            image, logs = client.images.build(path=str(folder_path))
        except BuildError as be:
            return False, be.msg

    try:
        output = client.containers.run(image, command="ls -a", remove=True)
    except Exception as e:
        return False, str(e)

    files = set(output.decode("utf-8").split("\n"))

    for f in expected_files:
        if f not in files:
            return False, f"Expected test data file {f} not found in test docker image"

    return True, "Test data retrieval script successful"
