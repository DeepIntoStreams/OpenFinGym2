from datetime import datetime
from pathlib import Path

from sqlalchemy import Engine
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from open_fin_gym.pipeline.db.tables import Paper

from .collection.collector import PaperCollector, logger
from .types import PaperRecord, Scope, ScrapingConfig


class PaperScrapingPipeline:
    def __init__(self, db: Engine, cfg: ScrapingConfig) -> None:
        """
        Pipeline for scraping and enriching paper-records

        Args:
            db: SQLAlchemy SQLite DB engine
            cfg: Scraping configuration
        """
        self.db = db
        self.collector = PaperCollector(
            cfg.arxiv,
            cfg.crossref,
            cfg.semantic_scholar,
        )

    def run(
        self,
        output_path: Path,
        scopes: list[Scope],
        since_date: datetime,
        until_date: datetime,
        per_scope_limit: int,
    ) -> None:
        """
        Run the scraping process and merge results to the DB

        Args:
            output_path: Hydra run output path
            scopes: List of search scopes
            since_date: Paper search from datetime
            until_date: Paper search to datetime
            per_scope_limit: Max number of papers to return per scope
        """
        output_path = output_path / "scraped_papers/"
        output_path.mkdir(exist_ok=True)

        for scope in scopes:
            logger.info(
                f"Scraping papers for scope {scope.id} from {since_date} to {until_date}"
            )
            papers = self.collector.collect_scope(
                scope,
                since_date,
                until_date,
                per_scope_limit,
            )
            self.insert_new_papers(papers)
            self.collector.dump_papers(output_path, scope, papers)

    def insert_new_papers(self, papers: dict[str, PaperRecord]) -> None:
        """
        Insert papers into db, skipping those already present

        Args:
            papers: Dictionary of scraped paper data
        """
        with Session(self.db) as session:
            inserts = [x.model_dump() for x in papers.values()]
            stmt = insert(Paper).values(inserts)
            # Skip papers that already have entries in the DB for this scope
            stmt = stmt.on_conflict_do_nothing(index_elements=["paper_id", "scope_id"])
            session.execute(stmt)
            session.commit()
