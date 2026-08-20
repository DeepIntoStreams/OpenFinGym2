import json
import logging
from pathlib import Path

from hydra.utils import instantiate
from langchain_core.language_models import BaseChatModel
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from open_fin_gym.pipeline.config import Scope
from open_fin_gym.pipeline.db.tables import (
    JudgeLabel,
    TaskCandidate,
    TaskCandidateStatus,
)
from open_fin_gym.pipeline.db.utils import set_task_candidate_status
from open_fin_gym.pipeline.steps.task_extraction.utils import (
    TaskSpecification,
    build_task_specification,
)

from .config import TaskCriticConfig
from .prompts import TaskCritique, build_critique_prompt
from .types import CritiqueResult
from .utils import structural_issues

logger = logging.getLogger(__name__)


class TaskCritic:
    def __init__(self, db: Engine, cfg: TaskCriticConfig) -> None:
        self.db = db
        self.cfg = cfg
        self.llm: BaseChatModel = instantiate(cfg.llm)
        assert isinstance(self.llm, BaseChatModel)

    def run(self, output_path: Path, scopes: list[Scope]) -> None:
        output_path = output_path / "task_critic/"
        output_path.mkdir(exist_ok=True)
        for scope in scopes:
            self.critique_scope(output_path, scope)

    def critique_scope(self, output_path: Path, scope: Scope) -> None:
        with Session(self.db) as session:
            stmt = select(TaskCandidate).where(
                TaskCandidate.scope_id == scope.id,
                TaskCandidate.status == TaskCandidateStatus.NEW,
            )
            candidates: list[TaskCandidate] = session.execute(stmt).scalars().all()
            specs = [build_task_specification(task) for task in candidates]

        logger.info(f"Critiquing {len(specs)} task candidates for scope {scope.id}")

        results = [self.critique_task(scope, spec) for spec in specs]

        with open(output_path / f"{scope.id}.json", "w") as f:
            json.dump([x.model_dump(mode="json") for x in results], f, indent=4)

    def critique_task(self, scope: Scope, spec: TaskSpecification) -> CritiqueResult:
        issues = structural_issues(spec)
        if issues:
            self.set_status(spec.id, TaskCandidateStatus.REJECTED)
            return CritiqueResult(
                task_id=spec.id,
                scope_id=scope.id,
                approved=False,
                score=None,
                confidence=None,
                reasoning=f"Rejected on structural checks: {'; '.join(issues)}",
            )

        prompt = build_critique_prompt(scope, spec)
        llm = self.llm.with_structured_output(TaskCritique)

        try:
            response: TaskCritique = llm.invoke(prompt)
            approved = (
                response.label == JudgeLabel.ACCEPTED
                and response.score >= self.cfg.threshold_default
            )
            reasoning = (
                f"{response.issues}\n\n"
                f"Consistency: {response.consistency_assessment}\n"
                f"Completeness: {response.completeness_assessment}\n"
                f"Data Availability: {response.data_availability_assessment}"
            )
            score = response.score
            confidence = response.confidence
        except Exception as e:
            logger.error(
                f"Task critique failed for task {spec.id} scope {scope.id}: {e}"
            )
            approved = False
            reasoning = f"LLM error during critique: {e}"
            score = None
            confidence = None

        status = (
            TaskCandidateStatus.APPROVED if approved else TaskCandidateStatus.REJECTED
        )
        self.set_status(spec.id, status)
        return CritiqueResult(
            task_id=spec.id,
            scope_id=scope.id,
            approved=approved,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
        )

    def set_status(self, task_id: int, status: TaskCandidateStatus) -> None:
        set_task_candidate_status(self.db, task_id, status)
