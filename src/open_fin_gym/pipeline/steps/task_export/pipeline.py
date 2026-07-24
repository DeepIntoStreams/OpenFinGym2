import logging
from pathlib import Path

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
                # Output this structure
                # my-task/
                # ├── task.toml
                # ├── instruction.md
                # ├── environment/
                # │   └── Dockerfile
                # └── tests/
                #     ├── Dockerfile
                #     ├── test.sh
                #     └── grader.py

                task_id = task.name.strip().lower().replace(" ", "_")
                task_dir = self.export_path / task_id
                task_dir.mkdir()

                # TODO: Write task.toml file

                with open(task_dir / "instruction.md", "w") as f:
                    f.write(task.instructions)

                env_dir = task_dir / "environment"
                env_dir.mkdir()

                # TODO: Write DockerFile to this dir

                with open(env_dir / "data.py", "w") as f:
                    f.write(task.train_script)

                test_dir = task_dir / "tests"
                test_dir.mkdir()

                # TODO: Write verifier DockerFile

                with open(test_dir / "data.py", "w") as f:
                    f.write(task.test_script)

                with open(test_dir / "assessor.py", "w") as f:
                    f.write(task.assessment_script)

                logger.info(f"Exported task {task.name} to {task_dir}")
                set_task_status(self.db, task.task_id, TaskStatus.EXPORTED)

            except Exception as e:
                logger.error(f"Export of task {task.name} failed - {e}")
                set_task_status(self.db, task.task_id, TaskStatus.EXPORT_FAILED)
