from pathlib import Path
from tempfile import TemporaryDirectory

import docker
from docker.errors import BuildError
from jinja2 import Template
from pylint import run_pylint


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

        with open(folder_path / "data.py", "w") as f:
            f.write(download_script)

        run_pylint(argv=[str(folder_path / "data.py")])

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
    data_download_script: str,
    verification_script: str,
    requirements: list[str],
    expected_files: list[str],
) -> tuple[bool, str]:
    """
    Build test docker image and check expected data files are created

    Args:
        docker_template: Test docker image template
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

        with open(folder_path / "data.py", "w") as f:
            f.write(data_download_script)

        with open(folder_path / "verifier.py", "w") as f:
            f.write(verification_script)

        run_pylint(argv=[str(folder_path / "data.py")])

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
            return False, f"Expected test data file {f} not found in test docker image"

    return True, "Test data retrieval script successful"
