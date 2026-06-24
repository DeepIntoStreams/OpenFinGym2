import json
import logging
from pathlib import Path

from hydra.utils import instantiate
from langchain_core.language_models import BaseChatModel
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from open_fin_gym.pipeline.config import Scope
from open_fin_gym.pipeline.db.utils import set_paper_status
from open_fin_gym.pipeline.db.tables import (
    Chunk,
    JudgeLabel,
    Paper,
    PaperStatus,
    RejectionReason,
)

from .config import JudgeConfig
from .prompts import (
    Evidence,
    PrefilterDecision,
    SiftJudgement,
    build_prefilter_prompt,
    build_sift_judge_prompt,
)
from .types import JudgeResult
from .utils import filter_chunks, rank_papers_for_llm

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
        output_path = output_path / "judgments/"
        output_path.mkdir(exist_ok=True)

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

        judgements = []
        candidates = []

        logger.info(f"Judging {len(scope_papers)} papers for scope {scope.id}")

        for paper in scope_papers:
            if self.cfg.prefilter_enabled:
                should_judge, prefilter_score, result = self.llm_prefilter(scope, paper)
            else:
                should_judge = True
                prefilter_score = 0.0
                result = None

            if not should_judge:
                self.set_paper_status(
                    paper,
                    PaperStatus.REJECTED,
                    prefilter_passed=False,
                    prefilter_score=prefilter_score,
                    rejection_reason=RejectionReason.JudgeRejected,
                )
                judgements.append(
                    JudgeResult(
                        paper_id=paper.paper_id,
                        scope_id=scope.id,
                        accepted=False,
                        rejection_reason=RejectionReason.JudgeRejected,
                        reasoning=result.reasons,
                    )
                )
            else:
                self.set_paper_status(
                    paper,
                    PaperStatus.EXTRACTED,
                    prefilter_passed=True,
                    prefilter_score=prefilter_score,
                )
                candidates.append(paper)

        sorted_candidates = rank_papers_for_llm(
            candidates,
            citation_boost=self.cfg.ranking_citation_boost,
            recency_weight=self.cfg.ranking_recency_weight,
            recency_half_life_days=self.cfg.ranking_recency_half_life_days,
        )

        accepted = sorted_candidates[: self.cfg.sift_budget]
        rejected = sorted_candidates[len(accepted) :]

        for paper in rejected:
            self.set_paper_status(
                paper,
                PaperStatus.REJECTED,
                rejection_reason=RejectionReason.JudgeCutoff,
            )
            judgements.append(
                JudgeResult(
                    paper_id=paper.paper_id,
                    scope_id=scope.id,
                    accepted=False,
                    rejection_reason=RejectionReason.JudgeCutoff,
                    reasoning="Did not make judge paper cutoff",
                )
            )

        for paper in accepted:
            with Session(self.db) as session:
                stmt = select(Chunk).where(Chunk.paper_id == paper.paper_id)
                chunks: list[Chunk] = session.execute(stmt).scalars().all()

            chunks = filter_chunks(chunks)
            excerpt = "/n/n".join([x.text for x in chunks])
            prompt = build_sift_judge_prompt(scope, paper, excerpt=excerpt)
            llm = self.llm.with_structured_output(SiftJudgement)

            try:
                response: SiftJudgement = llm.invoke(prompt)
                rejection_reason = None
            except Exception as e:
                logger.error(
                    f"LLM judgement failed for paper {paper.paper_id} from scope {paper.scope_id}: {e}"
                )
                response: SiftJudgement = SiftJudgement(
                    label=JudgeLabel.REJECTED,
                    score=0,
                    confidence=0.0,
                    reasons=f"LLM error for this paper {e}",
                    evidence=Evidence(experiments="", datasets="", metrics=""),
                )
                rejection_reason = RejectionReason.LLMError

            accepted = (
                response.label == JudgeLabel.ACCEPTED
                and response.score >= self.cfg.threshold_default
            )
            status = PaperStatus.ACCEPTED if accepted else PaperStatus.REJECTED
            self.set_paper_status(paper, status, rejection_reason=rejection_reason)
            judgements.append(
                JudgeResult(
                    paper_id=paper.paper_id,
                    scope_id=scope.id,
                    accepted=accepted,
                    rejection_reason=rejection_reason,
                    reasoning=response.reasons,
                )
            )

        # Dump judgements out to JSON for debugging
        with open(output_path / f"{scope.id}.json", "w") as f:
            json.dump(
                [x.model_dump(mode="json") for x in judgements],
                f,
                indent=4,
            )

    def llm_prefilter(
        self, scope: Scope, paper: Paper
    ) -> tuple[bool, float, PrefilterDecision]:
        """
        Pre-filter papers using an LLM judge

        Args:
            scope: Task scope
            paper: Paper record

        Returns:
            Judgement outcome
        """
        llm = self.llm.with_structured_output(PrefilterDecision)
        prompt = build_prefilter_prompt(scope, paper)

        try:
            result: PrefilterDecision = llm.invoke(prompt)
        except Exception as e:
            logger.error(
                f"LLM prefilter failed for paper {paper.paper_id} from scope {paper.scope_id}: {e}"
            )
            result: PrefilterDecision = PrefilterDecision(
                label=JudgeLabel.REJECTED,
                relevance_score=0.0,
                confidence=0.0,
                reasons=f"prefilter llm error: {e}",
            )

        score = result.relevance_score
        conf = result.confidence
        keep = result.label == JudgeLabel.ACCEPTED
        min_score = self.cfg.prefilter_llm_score_min
        min_conf = self.cfg.prefilter_llm_confidence_min
        passed = keep and score >= min_score and conf >= min_conf
        return passed, score, result

    def set_paper_status(self, paper: Paper, status: PaperStatus, **kwargs) -> None:
        """
        Update paper status in the DB

        Args:
            paper: Current paper instance
            status: Status value
        """
        set_paper_status(self.db, paper, status, **kwargs)
