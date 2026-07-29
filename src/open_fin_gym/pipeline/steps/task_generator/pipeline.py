import logging
from pathlib import Path

from hydra.utils import instantiate
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

from .config import TaskGenerationConfig
from .prompts import (
    Assessment,
    DatasetRetrieval,
    TaskSpecification,
    build_dataset_download_prompt,
    build_description_summary_prompt,
    build_difficulty_explanation_prompt,
    build_metric_prompt,
    build_task_markdown,
    convert_dataset,
    convert_metric,
)

logger = logging.getLogger(__name__)


class TaskGenerator:
    def __init__(self, db: Engine, cfg: TaskGenerationConfig) -> None:
        self.db = db
        self.llm: BaseChatModel = instantiate(cfg.llm)
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
                TaskCandidate.status == TaskCandidateStatus.NEW,
            )
            tasks: list[TaskCandidate] = session.execute(stmt).scalars().all()
            task_specs = [
                TaskSpecification(
                    id=task.task_id,
                    task_name=task.task_name,
                    task_description=task.description,
                    training_inputs=[convert_dataset(x) for x in task.training_inputs],
                    training_targets=[
                        convert_dataset(x) for x in task.training_targets
                    ],
                    test_inputs=[convert_dataset(x) for x in task.test_inputs],
                    test_targets=[convert_dataset(x) for x in task.test_targets],
                    test_outputs=[convert_dataset(x) for x in task.test_outputs],
                    metrics=[convert_metric(x) for x in task.assessment_metrics],
                )
                for task in tasks
            ]

        logger.info(f"Generating {len(task_specs)} tasks for scope {scope.name}")

        for task_spec in task_specs:
            dataset_prompt = build_dataset_download_prompt(task_spec)
            metric_prompt = build_metric_prompt(task_spec)
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
                instructions = build_task_markdown(task_spec)

                task_path = scripts_path / f"{task_spec.task_name}"
                task_path.mkdir(exist_ok=True)

                with open(task_path / "train.py", "w") as f:
                    f.write(dataset_scripts.training_script)

                with open(task_path / "test.py", "w") as f:
                    f.write(dataset_scripts.testing_script)

                with open(task_path / "grader.py", "w") as f:
                    f.write(assessment_script.assessment_script)

                with open(task_path / "requirements.txt", "w") as f:
                    f.write("\n".join(requirements))

                with open(task_path / "instruction.md", "w") as f:
                    f.write(instructions)

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
