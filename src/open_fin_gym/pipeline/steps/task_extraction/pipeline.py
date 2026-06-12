from typing import Optional

from hydra.utils import instantiate
from langchain_core.language_models import BaseChatModel
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from open_fin_gym.pipeline.db.tables import (
    Chunk,
    DatasetCandidate,
    MetricCandidate,
    Paper,
    TaskCandidate,
)
from open_fin_gym.pipeline.db.utils import set_paper_status
from open_fin_gym.pipeline.steps.judge.utils import filter_chunks
from open_fin_gym.pipeline.steps.scrape_papers.types import PaperStatus, Scope

from .prompts import PaperTaskSummary, build_paper_summary_prompt
from .types import TaskExtractionConfig


class TaskExtractor:
    def __init__(self, db: Engine, cfg: TaskExtractionConfig) -> None:
        self.db: Engine = db
        self.llm: BaseChatModel = instantiate(cfg.llm)
        assert isinstance(self.llm, BaseChatModel)

    def run(self, scopes: list[Scope]) -> None:
        for scope in scopes:
            self.extract_scope(scope)

    def extract_scope(self, scope: Scope) -> None:

        with Session(self.db) as session:
            stmt = select(Paper).where(
                Paper.scope_id == scope.id, Paper.status == PaperStatus.ACCEPTED
            )
            papers: list[Paper] = session.execute(stmt).scalars().all()

        task_candidates = []
        dataset_candidates = []
        metric_candidates = []

        for paper in papers:
            with Session(self.db) as session:
                stmt = select(Chunk).where(Chunk.paper_id == paper.paper_id)
                chunks: list[Chunk] = session.execute(stmt).scalars().all()

            chunks = filter_chunks(chunks)
            extract = "\n\n".join([x.text for x in chunks])
            prompt = build_paper_summary_prompt(scope, paper, extract)
            llm = self.llm.with_structured_output(PaperTaskSummary)

            try:
                task_summary: Optional[PaperTaskSummary] = llm.invoke(prompt)
            except Exception:
                task_summary: Optional[PaperTaskSummary] = None

            set_paper_status(self.db, paper.paper_id, PaperStatus.COMPLETE)

            if task_summary:
                task_candidate = TaskCandidate(
                    scope_id=scope.id,
                    paper_id=paper.paper_id,
                    task_name=task_summary.task_name,
                    ml_task_summary=task_summary.ml_task_summary,
                    experiments=task_summary.experiments,
                    links=task_summary.links,
                    task_family=task_summary.task_family,
                )
                dataset_candidates.extend(
                    [
                        DatasetCandidate(
                            name=x.name,
                            description=x.description,
                            dataset_type=x.dataset_type,
                            task_candidate=task_candidate,
                        )
                        for x in task_summary.datasets
                    ]
                )
                metric_candidates.extend(
                    [
                        MetricCandidate(
                            name=x.name,
                            description=x.description,
                            task_candidate=task_candidate,
                        )
                        for x in task_summary.metrics
                    ]
                )

        with Session(self.db) as session:
            session.add_all(task_candidates)
            session.add_all(dataset_candidates)
            session.add_all(metric_candidates)
            session.commit()
