from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from open_fin_gym.pipeline.config import Scope
from open_fin_gym.pipeline.db.tables import (
    Task,
    TaskCandidate,
    TaskCandidateStatus,
    TaskStatus,
    TaskType,
)
from open_fin_gym.pipeline.steps.task_export.config import (
    TaskConfig,
    TaskExportConfig,
)
from open_fin_gym.pipeline.steps.task_export.pipeline import TaskExporter


def make_scope(scope_id: str) -> Scope:
    return Scope(
        id=scope_id, name=scope_id.upper(), task_type="forecasting", description=""
    )


def make_candidate(db: Engine, scope_id: str, name: str = "candidate") -> int:
    """Insert a TaskCandidate, returning its task_id (the key Task points at)."""
    with Session(db) as session:
        candidate = TaskCandidate(
            scope_id=scope_id,
            paper_id="p1",
            task_name=name,
            task_type=TaskType.FORECASTING,
            plan="plan",
            instructions="instructions",
            status=TaskCandidateStatus.PROCESSED,
        )
        session.add(candidate)
        session.commit()
        return candidate.task_id


def make_task(
    db: Engine,
    scope_id: str,
    name: str,
    status: TaskStatus = TaskStatus.NEW,
) -> int:
    """Insert a Task (via its TaskCandidate, as task_generator does)."""
    candidate_id = make_candidate(db, scope_id, name)
    with Session(db) as session:
        task = Task(
            task_candidate_id=candidate_id,
            name=name,
            status=status,
            train_script="print('train')",
            test_script="print('test')",
            assessment_script="print('verify')",
            requirements=["pandas"],
            instructions="instructions",
            short_description="short description",
            difficulty_explanation="difficulty",
        )
        session.add(task)
        session.commit()
        return task.task_id


def make_exporter(db: Engine, tmp_path: Path) -> TaskExporter:
    cfg = TaskExportConfig(
        export_path=str(tmp_path / "export"),
        templates_path="templates",
        task_config=TaskConfig(org_name="org"),
    )
    return TaskExporter(db, cfg)


def statuses(db: Engine) -> dict[str, TaskStatus]:
    with Session(db) as session:
        tasks = session.execute(select(Task)).scalars().all()
        return {t.name: t.status for t in tasks}


def test_extract_scope_only_exports_tasks_from_that_scope(
    db: Engine, tmp_path: Path
) -> None:
    make_task(db, "a", "task a")
    make_task(db, "b", "task b")

    make_exporter(db, tmp_path).extract_scope(make_scope("a"), tmp_path)

    assert statuses(db) == {
        "task a": TaskStatus.EXPORTED,
        "task b": TaskStatus.NEW,
    }


def test_extract_scope_does_not_write_other_scopes_files(
    db: Engine, tmp_path: Path
) -> None:
    make_task(db, "a", "task a")
    make_task(db, "b", "task b")

    make_exporter(db, tmp_path).extract_scope(make_scope("a"), tmp_path)

    assert (tmp_path / "export" / "task_a" / "task.toml").exists()
    assert not (tmp_path / "export" / "task_b").exists()


def test_scope_join_uses_candidate_id_not_task_id(db: Engine, tmp_path: Path) -> None:
    # Pad the candidate table so candidate ids and task ids no longer coincide.
    # A join on Task.task_id == TaskCandidate.task_id would pick the wrong rows.
    for i in range(3):
        make_candidate(db, "b", f"padding {i}")
    make_task(db, "a", "task a")
    make_task(db, "b", "task b")

    make_exporter(db, tmp_path).extract_scope(make_scope("a"), tmp_path)

    assert statuses(db) == {
        "task a": TaskStatus.EXPORTED,
        "task b": TaskStatus.NEW,
    }


def test_scope_without_tasks_exports_nothing(db: Engine, tmp_path: Path) -> None:
    make_task(db, "b", "task b")

    make_exporter(db, tmp_path).extract_scope(make_scope("a"), tmp_path)

    assert statuses(db) == {"task b": TaskStatus.NEW}
    assert list((tmp_path / "export").iterdir()) == []


def test_extract_scope_skips_tasks_that_are_not_new(db: Engine, tmp_path: Path) -> None:
    make_task(db, "a", "done", TaskStatus.EXPORTED)
    make_task(db, "a", "broken", TaskStatus.EXPORT_FAILED)

    make_exporter(db, tmp_path).extract_scope(make_scope("a"), tmp_path)

    assert statuses(db) == {
        "done": TaskStatus.EXPORTED,
        "broken": TaskStatus.EXPORT_FAILED,
    }
    assert list((tmp_path / "export").iterdir()) == []


def test_run_exports_every_scope_once(db: Engine, tmp_path: Path) -> None:
    make_task(db, "a", "task a")
    make_task(db, "b", "task b")

    make_exporter(db, tmp_path).run(tmp_path, [make_scope("a"), make_scope("b")])

    assert statuses(db) == {
        "task a": TaskStatus.EXPORTED,
        "task b": TaskStatus.EXPORTED,
    }
    assert (tmp_path / "export" / "task_a" / "task.toml").exists()
    assert (tmp_path / "export" / "task_b" / "task.toml").exists()


@pytest.mark.parametrize("scope_id", ["a", "b"])
def test_run_with_one_scope_leaves_the_other_scopes_tasks(
    db: Engine, tmp_path: Path, scope_id: str
) -> None:
    # The single-paper pipeline calls run() with just the requested scope
    make_task(db, "a", "task a")
    make_task(db, "b", "task b")

    make_exporter(db, tmp_path).run(tmp_path, [make_scope(scope_id)])

    expected = {
        "task a": TaskStatus.EXPORTED if scope_id == "a" else TaskStatus.NEW,
        "task b": TaskStatus.EXPORTED if scope_id == "b" else TaskStatus.NEW,
    }
    assert statuses(db) == expected