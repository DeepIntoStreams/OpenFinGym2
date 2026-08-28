import logging
from pathlib import Path

from hydra.utils import instantiate
from jinja2 import Environment, FileSystemLoader
from langchain_core.language_models import BaseChatModel
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from open_fin_gym.pipeline.config import Scope
from open_fin_gym.pipeline.db.tables import (
    Task,
    TaskCandidate,
    TaskCandidateStatus,
    TaskStatus,
)
from open_fin_gym.pipeline.db.utils import set_task_candidate_status
from open_fin_gym.pipeline.steps.task_extraction.utils import (
    build_task_specification,
)

from .code_checks import (
    run_pylint,
    test_test_download_script,
    test_train_download_script,
)
from .config import TaskGenerationConfig
from .prompts import (
    Assessment,
    DatasetRetrieval,
    build_dataset_download_prompt,
    build_description_summary_prompt,
    build_difficulty_explanation_prompt,
    build_metric_prompt,
)

logger = logging.getLogger(__name__)


class TaskGenerator:
    def __init__(self, db: Engine, cfg: TaskGenerationConfig) -> None:
        self.db = db
        self.llm: BaseChatModel = instantiate(cfg.llm)
        self.template_env = Environment(loader=FileSystemLoader(cfg.templates_path))

        def strformat(v, fmt):
            return fmt % v

        self.template_env.filters["strformat"] = strformat

        self.instructions_template = self.template_env.get_template(
            "instructions.md.j2"
        )
        self.train_docker_template = self.template_env.get_template(
            "train.Dockerfile.j2"
        )
        self.test_docker_template = self.template_env.get_template("test.Dockerfile.j2")
        self.test_sh_template = self.template_env.get_template("test.sh.j2")
        assert isinstance(self.llm, BaseChatModel)

    def run(self, output_path: Path, scopes: list[Scope]) -> None:
        """
        Run task generation across provided scopes

        Args:
            output_path: Hydra run output path
            scopes: List of paper scopes
        """
        output_path = output_path / "task_generation/"
        output_path.mkdir(exist_ok=True)

        for scope in scopes:
            self.extract_scope(scope, output_path)

    def extract_scope(self, scope: Scope, output_path: Path) -> None:

        scripts_path = output_path / "scripts/"
        scripts_path.mkdir(exist_ok=True)

        with Session(self.db) as session:
            stmt = select(TaskCandidate).where(
                TaskCandidate.scope_id == scope.id,
                TaskCandidate.status == TaskCandidateStatus.APPROVED,
            )
            tasks: list[TaskCandidate] = session.execute(stmt).scalars().all()
            task_specs = [build_task_specification(task) for task in tasks]

        logger.info(f"Generating {len(task_specs)} tasks for scope {scope.name}")

        for task_spec in task_specs:
            dataset_prompt = build_dataset_download_prompt(task_spec)
            metric_prompt = build_metric_prompt(task_spec, scope.task_params)
            description_prompt = build_description_summary_prompt(
                task_spec.task_description
            )
            difficulty_prompt = build_difficulty_explanation_prompt(
                task_spec.task_description
            )

            try:
                dataset_scripts: DatasetRetrieval = self.llm.with_structured_output(
                    DatasetRetrieval
                ).invoke(dataset_prompt)
                assessment_script: Assessment = self.llm.with_structured_output(
                    Assessment
                ).invoke(metric_prompt)
                short_description = self.llm.invoke(description_prompt).content
                difficulty_explanation = self.llm.invoke(difficulty_prompt).content

                requirements = list(
                    set(dataset_scripts.requirements).union(
                        set(assessment_script.requirements)
                    )
                )
                instructions = self.instructions_template.render(
                    task_description=task_spec.task_description,
                    training_inputs=task_spec.training_inputs,
                    training_targets=task_spec.training_targets,
                    test_inputs=task_spec.test_inputs,
                    test_outputs=task_spec.test_outputs,
                    metrics=task_spec.metrics,
                )

                task_path = scripts_path / f"{task_spec.task_name}"
                task_path.mkdir(exist_ok=True)

                # Write files and run PyLint over generated Python scripts
                with open(task_path / "train.py", "w") as f:
                    f.write(dataset_scripts.training_script)

                train_pass, pylint_messages = run_pylint(task_path / "train.py")

                if not train_pass:
                    raise RuntimeError(
                        f"PyLint detected the following errors in the train data script \n\n{pylint_messages}"
                    )

                with open(task_path / "test.py", "w") as f:
                    f.write(dataset_scripts.testing_script)

                test_pass, pylint_messages = run_pylint(task_path / "test.py")

                if not test_pass:
                    raise RuntimeError(
                        f"PyLint detected the following errors in the test data script \n\n{pylint_messages}"
                    )

                with open(task_path / "verifier.py", "w") as f:
                    f.write(assessment_script.assessment_script)

                verifier_pass, pylint_messages = run_pylint(task_path / "verifier.py")

                if not verifier_pass:
                    raise RuntimeError(
                        f"PyLint detected the following errors in the verifier script \n\n{pylint_messages}"
                    )

                with open(task_path / "requirements.txt", "w") as f:
                    f.write("\n".join(requirements))

                with open(task_path / "instruction.md", "w") as f:
                    f.write(instructions)

                # Test Dockerfiles build and data is retrieved
                train_build_success, train_build_message = test_train_download_script(
                    self.train_docker_template,
                    dataset_scripts.training_script,
                    requirements,
                    [x.filename for x in task_spec.training_inputs]
                    + [x.filename for x in task_spec.training_targets]
                    + [x.filename for x in task_spec.test_inputs],
                )
                if not train_build_success:
                    raise RuntimeError(
                        f"Train DockerFile build failed - {train_build_message}"
                    )

                test_build_success, test_build_message = test_test_download_script(
                    self.test_docker_template,
                    self.test_sh_template,
                    dataset_scripts.testing_script,
                    assessment_script.assessment_script,
                    requirements,
                    [x.filename for x in task_spec.test_targets],
                )
                if not test_build_success:
                    raise RuntimeError(
                        f"Test DockerFile build failed - {test_build_message}"
                    )

                new_task = Task(
                    task_candidate_id=task_spec.id,
                    name=task_spec.task_name,
                    status=TaskStatus.NEW,
                    train_script=dataset_scripts.training_script,
                    test_script=dataset_scripts.testing_script,
                    assessment_script=assessment_script.assessment_script,
                    requirements=requirements,
                    instructions=instructions,
                    short_description=short_description,
                    difficulty_explanation=difficulty_explanation,
                )

                with Session(self.db) as session:
                    session.add(new_task)
                    session.commit()

                set_task_candidate_status(
                    self.db, task_spec.id, TaskCandidateStatus.PROCESSED
                )

            except Exception as e:
                set_task_candidate_status(
                    self.db, task_spec.id, TaskCandidateStatus.FAILED
                )
                logger.error(
                    f"Task generation failed for task {task_spec.task_name} - {e}"
                )
