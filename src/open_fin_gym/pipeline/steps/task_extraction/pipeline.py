import json
import logging
from pathlib import Path
from typing import Optional

from hydra.utils import instantiate
from langchain_core.language_models import BaseChatModel
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from open_fin_gym.pipeline.config import Scope
from open_fin_gym.pipeline.db.tables import (
    Chunk,
    Paper,
    TaskCandidate,
    TaskCandidateStatus,
    TestInputDatasetCandidate,
    TestOutputDatasetCandidate,
    TestTargetDatasetCandidate,
    TrainInputDatasetCandidate,
    TrainTargetDatasetCandidate,
)
from open_fin_gym.pipeline.db.utils import set_paper_status
from open_fin_gym.pipeline.steps.judge.utils import filter_chunks
from open_fin_gym.pipeline.steps.scrape_papers.types import PaperStatus

from .config import TaskExtractionConfig
from .prompts import PaperTaskSpecification, build_paper_summary_prompt
from .utils import unpack_datasets, unpack_metrics

logger = logging.getLogger(__name__)


class TaskExtractor:
    def __init__(self, db: Engine, cfg: TaskExtractionConfig) -> None:
        """
        Extract candidate task metadata from accepted paper

        Args:
            db: SQLAlchemy db engine
            cfg: Task extraction configuration
        """
        self.db: Engine = db
        self.llm: BaseChatModel = instantiate(cfg.llm)
        assert isinstance(self.llm, BaseChatModel)

    def run(self, output_path: Path, scopes: list[Scope]) -> None:
        """
        Run task extraction across provided scopes

        Args:
            output_path: Hydra run output path
            scopes: List of paper scopes
        """
        output_path = output_path / "task_extraction/"
        output_path.mkdir(exist_ok=True)

        for scope in scopes:
            self.extract_scope(scope, output_path)

    def extract_scope(self, scope: Scope, output_path: Path) -> None:
        """
        Extract task from accepted papers for a given scope

        Args:
            scope: Task scope
            output_path: Hydra run output path
        """

        with Session(self.db) as session:
            stmt = select(Paper).where(
                Paper.scope_id == scope.id, Paper.status == PaperStatus.ACCEPTED
            )
            papers: list[Paper] = session.execute(stmt).scalars().all()

        logger.info(f"Extracting tasks from {len(papers)} papers for scope {scope.id}")

        task_summaries = []
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
            llm = self.llm.with_structured_output(PaperTaskSpecification)

            try:
                task_summary: Optional[PaperTaskSpecification] = llm.invoke(prompt)
                task_summaries.append(task_summary)
                set_paper_status(self.db, paper, PaperStatus.TASK_EXTRACTED)
            except Exception as e:
                logger.error(
                    f"Task extraction failed from paper {paper.paper_id} from scope {scope.id} due to {e}"
                )
                task_summary: Optional[PaperTaskSpecification] = None
                set_paper_status(self.db, paper, PaperStatus.TASK_EXTRACTION_FAILED)

            if task_summary:
                task_candidate = TaskCandidate(
                    scope_id=scope.id,
                    paper_id=paper.paper_id,
                    task_name=task_summary.task_name,
                    task_type=task_summary.task_type,
                    description=task_summary.task_description,
                    status=TaskCandidateStatus.NEW,
                )

                training_input_data = unpack_datasets(
                    TrainInputDatasetCandidate,
                    task_candidate,
                    task_summary.training_inputs,
                )
                task_candidate.training_inputs.extend(training_input_data)
                dataset_candidates.extend(training_input_data)

                training_target_data = unpack_datasets(
                    TrainTargetDatasetCandidate,
                    task_candidate,
                    task_summary.training_targets,
                )
                task_candidate.training_targets.extend(training_target_data)
                dataset_candidates.extend(training_target_data)

                test_input_datasets = unpack_datasets(
                    TestInputDatasetCandidate, task_candidate, task_summary.test_inputs
                )
                task_candidate.test_inputs.extend(test_input_datasets)
                dataset_candidates.extend(test_input_datasets)

                test_output_datasets = unpack_datasets(
                    TestOutputDatasetCandidate,
                    task_candidate,
                    task_summary.test_outputs,
                )
                task_candidate.test_outputs.extend(test_output_datasets)
                dataset_candidates.extend(test_output_datasets)

                test_target_datasets = unpack_datasets(
                    TestTargetDatasetCandidate,
                    task_candidate,
                    task_summary.test_targets,
                )
                task_candidate.test_targets.extend(test_target_datasets)
                dataset_candidates.extend(test_target_datasets)

                metrics = unpack_metrics(
                    task_candidate, task_summary.assessment_metrics
                )
                task_candidate.assessment_metrics.extend(metrics)
                metric_candidates.extend(metrics)

                task_candidates.append(task_candidate)

        with Session(self.db) as session:
            session.add_all(task_candidates)
            session.add_all(dataset_candidates)
            session.add_all(metric_candidates)
            session.commit()

        with open(output_path / f"{scope.id}.json", "w") as f:
            json.dump(
                [x.model_dump(mode="json") for x in task_summaries],
                f,
                indent=4,
            )
