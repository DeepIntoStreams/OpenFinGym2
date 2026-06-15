import json
import logging
from datetime import datetime
from pathlib import Path

from open_fin_gym.pipeline.steps.scrape_papers.types import (
    ArxivConfig,
    CrossrefConfig,
    PaperRecord,
    Scope,
    SemanticScholarConfig,
)

from .arxiv import ArxivClient
from .crossref import CrossrefClient
from .semantic_scholar import SemanticScholarClient

logger = logging.getLogger(__name__)


class PaperCollector:
    def __init__(
        self,
        arxiv_config: ArxivConfig,
        crossref_config: CrossrefConfig,
        s2_config: SemanticScholarConfig,
    ):
        """
        Paper record scraping and enrichment

        Args:
            arxiv_config: Arxiv scraping config
            crossref_config: Crossref doi enrichment config
            s2_config: Semantic-scholar metadate enrichment config
        """
        self.arxiv = ArxivClient(arxiv_config)
        self.crossref = CrossrefClient(crossref_config)
        self.s2 = SemanticScholarClient(s2_config)

    def collect_scope(
        self,
        scope: Scope,
        since_date: datetime | None,
        until_date: datetime,
        per_scope_limit: int,
    ) -> dict[str, PaperRecord]:
        """
        Scrape and enrich papers for a given scope

        Args:
            scope: Scope configuration
            since_date: Optional date to search from
            until_date: Optional date to search to
            per_scope_limit: Max number of papers to return for this scope

        Returns:
            Dictionary of paper records indexed by their id
        """
        papers = self.arxiv.scrape_scope(
            scope,
            since_date=since_date,
            until_date=until_date,
            max_papers=per_scope_limit,
        )
        papers = self.crossref.enrich_doi(papers)
        papers = self.s2.enrich(papers)

        return papers

    def dump_papers(
        self,
        output_path: Path,
        scope: Scope,
        papers: dict[str, PaperRecord],
    ) -> None:
        """
        Dump scraped data to a JSON file

        Args:
            output_path: Output folder
            scope: Scope associated with the scraped papers
            papers: Nested dictionary of scopes and papers scraped from each
        """
        output_file = output_path / f"{scope.id}.json"

        logger.info(f"Logging scraped papers to {output_file}")

        with open(output_file, "w") as f:
            json.dump(
                [x.model_dump(mode="json") for x in papers.values()],
                f,
                indent=4,
            )
