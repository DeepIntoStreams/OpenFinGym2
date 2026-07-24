import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from open_fin_gym.pipeline.config import Scope
from open_fin_gym.pipeline.db.tables import Task, TaskStatus
from open_fin_gym.pipeline.db.utils import set_task_status

from .config import TaskExportConfig

logger = logging.getLogger(__name__)


class TaskExporter:
    def __init__(self, db: Engine, cfg: TaskExportConfig) -> None:
        self.db = db
        self.export_path = Path(cfg.export_path)
        self.export_path.mkdir(exist_ok=True)
        self.template_env = Environment(loader=FileSystemLoader(cfg.templates_path))
        self.task_meta_template = self.template_env.get_template("task.toml.j2")
        self.train_docker_template = self.template_env.get_template(
            "train.Dockerfile.j2"
        )
        self.test_docker_template = self.template_env.get_template("test.Dockerfile.j2")
        self.test_script_template = self.template_env.get_template("test.sh.j2")
        self.task_meta = cfg.task_config

    def run(self, output_path: Path, scopes: list[Scope]) -> None:
        """
        Run task export across provided scopes

        Args:
            output_path: Hydra run output path
            scopes: List of paper scopes
        """
        output_path = output_path / "task_generation/"
        output_path.mkdir(exist_ok=True)

        for scope in scopes:
            self.extract_scope(scope, output_path)

    def extract_scope(self, scope: Scope, output_path: Path) -> None:

        with Session(self.db) as session:
            stmt = select(Task).where(Task.status == TaskStatus.NEW)
            tasks: list[Task] = session.execute(stmt).scalars().all()

        logger.info(f"Exporting {len(tasks)} tasks for scope {scope.name}")

        for task in tasks:
            try:
                task_id = task.name.strip().lower().replace(" ", "_")
                task_dir = self.export_path / task_id
                task_dir.mkdir()

                task_config = self.task_meta_template.render(
                    org_name=self.task_meta.org_name,
                    task_name=task_id,
                    description="",
                    keywords=[],
                    difficulty_explanation="",
                )

                with open(task_dir / "task.toml", "w") as f:
                    f.write(task_config)

                with open(task_dir / "instruction.md", "w") as f:
                    f.write(task.instructions)

                env_dir = task_dir / "environment"
                env_dir.mkdir()

                uv_req = " ".join([f"--with {x}" for x in task.requirements])
                train_docker = self.train_docker_template.render(requirements=uv_req)

                with open(env_dir / "Dockerfile", "w") as f:
                    f.write(train_docker)

                with open(env_dir / "data.py", "w") as f:
                    f.write(task.train_script)

                test_dir = task_dir / "tests"
                test_dir.mkdir()

                test_docker = self.test_docker_template.render(requirements=uv_req)

                with open(test_dir / "Dockerfile", "w") as f:
                    f.write(test_docker)

                with open(test_dir / "data.py", "w") as f:
                    f.write(task.test_script)

                with open(test_dir / "verifier.py", "w") as f:
                    f.write(task.assessment_script)

                test_script = self.test_script_template.render(requirements=uv_req)

                with open(test_dir / "test.sh", "w") as f:
                    f.write(test_script)

                logger.info(f"Exported task {task.name} to {task_dir}")
                set_task_status(self.db, task.task_id, TaskStatus.EXPORTED)

            except Exception as e:
                logger.error(f"Export of task {task.name} failed - {e}")
                set_task_status(self.db, task.task_id, TaskStatus.EXPORT_FAILED)
