import logging
from pathlib import Path

from hydra.utils import instantiate
from langchain_core.language_models import BaseChatModel
from sqlalchemy import Engine, select, update
from sqlalchemy.orm import Session

from open_fin_gym.pipeline.db.tables import Chunk, Paper
from open_fin_gym.pipeline.steps.scrape_papers.types import (
    JudgeLabel,
    PaperStatus,
    Scope,
)

from .prompts import (
    build_prefilter_prompt,
    build_sift_judge_prompt,
    prefilter_output_schema,
    sift_output_schema,
)
from .types import JudgeConfig
from .utils import rank_papers_for_llm

logger = logging.getLogger(__name__)


class Judge:
    def __init__(
        self,
        db: Engine,
        cfg: JudgeConfig,
    ) -> None:
        """
        Judge papers for acceptance to the task creation stage

        Args:
            db: SqlAlchemy database engine
            cfg: Judge configuration
        """
        self.db = db
        self.cfg = cfg
        self.llm = instantiate(cfg.llm)
        assert isinstance(self.llm, BaseChatModel)

    def run(
        self,
        output_path: Path,
        scopes: list[Scope],
    ) -> None:
        """
        Run the judgement process across the provided scopes and update db records

        Args:
            output_path: Hydra run output path
            scopes: List of task scopes
        """
        for scope in scopes:
            self.judge_scope(output_path, scope)

    def judge_scope(
        self,
        output_path: Path,
        scope: Scope,
    ) -> None:
        """
        Run judgement across papers for a given scope and update db records

        Args:
            output_path: Hydra run output path
            scope: Task scope
        """
        with Session(self.db) as session:
            stmt = select(Paper).where(
                Paper.scope_id == scope.id, Paper.status == PaperStatus.EXTRACTED
            )
            scope_papers = session.execute(stmt).scalars().all()

        candidates = []

        for paper in scope_papers:
            if self.cfg.prefilter_enabled:
                should_judge, prefilter_score, _ = self.llm_prefilter(scope, paper)
            else:
                should_judge = True
                prefilter_score = 0.0

            paper.prefilter_passed = should_judge
            paper.prefilter_score = prefilter_score

            if not paper.prefilter_passed:
                self.set_paper_status(paper.paper_id, PaperStatus.REJECTED)
            else:
                candidates.append(paper)

        sorted_candidates = rank_papers_for_llm(
            candidates,
            citation_boost=self.cfg.ranking_citation_boost,
            recency_weight=self.cfg.ranking_recency_weight,
            recency_half_life_days=self.cfg.ranking_recency_half_life_days,
        )

        accepted = sorted_candidates[: self.cfg.sift_budget]
        rejected = sorted_candidates[-self.cfg.sift_budget :]

        for paper in rejected:
            paper.status = PaperStatus.REJECTED

        for paper in accepted:
            with Session(self.db) as session:
                stmt = select(Chunk).where(Chunk.paper_id == paper.paper_id)
                chunks = session.execute(stmt).scalars().all()

            excerpt = "/n/n".join([x.text for x in chunks])
            prompt = build_sift_judge_prompt(scope, paper, excerpt=excerpt)
            llm = self.llm.with_structured_output(sift_output_schema())

            try:
                response = llm.invoke(prompt)
            except Exception as e:
                response = {
                    "label": "rejected",
                    "score_0_10": 0.0,
                    "confidence_0_1": 0.0,
                    "reasons": f"LLM unavailable for this paper {e}",
                    "evidence": {"experiments": "", "datasets": "", "metrics": ""},
                }

            accepted = (
                response["label"] == JudgeLabel.ACCEPTED
                and response["score_0_10"] >= self.cfg.threshold_default
            )
            status = PaperStatus.ACCEPTED if accepted else PaperStatus.REJECTED
            self.set_paper_status(paper.paper_id, status)

    def llm_prefilter(self, scope: Scope, paper: Paper) -> tuple[bool, float, dict]:
        """
        Pre-filter papers using an LLM judge

        Args:
            scope: Task scope
            paper: Paper record

        Returns:
            Judgement outcome
        """
        llm = self.llm.with_structured_output(prefilter_output_schema())
        prompt = build_prefilter_prompt(scope, paper)

        try:
            result = llm.invoke(prompt)
        except Exception as e:
            result = (
                {
                    "label": "rejected",
                    "relevance_score_0_10": 0.0,
                    "confidence_0_1": 0.0,
                    "reasons": f"prefilter fallback:  {e}",
                },
            )

        score = float(result.get("relevance_score_0_10", 0.0))
        conf = float(result.get("confidence_0_1", 0.0))
        keep = str(result.get("label", "rejected")).strip().lower() == "accepted"
        min_score = self.cfg.prefilter_llm_score_min
        min_conf = self.cfg.prefilter_llm_confidence_min
        passed = keep and score >= min_score and conf >= min_conf
        return passed, score, result

    def set_paper_status(self, paper_id: str, status: PaperStatus) -> None:
        """
        Update paper status in the DB

        Args:
            paper_id: Id of the paper
            status: Status value
        """
        with Session(self.db) as session:
            stmt = (
                update(Paper)
                .values({"status": status})
                .where(Paper.paper_id == paper_id)
            )
            session.execute(stmt)
            session.commit()
