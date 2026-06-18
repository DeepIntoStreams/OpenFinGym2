import logging
import time
from typing import Optional

from semanticscholar import Paper, SemanticScholar

from open_fin_gym.pipeline.steps.scrape_papers.types import PaperRecord

from .config import SemanticScholarConfig

logger = logging.getLogger(__name__)

_TITLE_MATCH_MIN_LEN = 12


class SemanticScholarClient:
    def __init__(self, cfg: SemanticScholarConfig) -> None:
        """
        Enrich papers via Semantic Scholar

        Args:
            cfg: Enrichment configuration
        """
        self.client = SemanticScholar(
            timeout=cfg.timeout_sec,
            api_url=cfg.api_url,
            retry=cfg.retry,
        )
        self.enabled = cfg.enabled
        self.batch_size = max(1, min(cfg.batch_size, 500))
        self.request_delay_secs = cfg.request_delay_secs

    def enrich(self, papers: dict[str, PaperRecord]) -> dict[str, PaperRecord]:
        """
        Enrich paper-records by matching papers by their ArXiv id

        Args:
            papers: Dictionary of paper-records

        Returns:
            Updated dictionary of paper-records
        """
        if not self.enabled or not papers:
            return papers

        paper_ids = list(papers.keys())

        for i in range(0, len(papers), self.batch_size):
            ids = paper_ids[i : i + self.batch_size]

            results = self.client.get_papers(
                ids,
                fields=[
                    "citationCount",
                    "influentialCitationCount",
                    "paperId",
                    "venue",
                    "journal",
                    "publicationTypes",
                    "externalIds",
                ],
            )

            for result in results:
                arxiv_id = result.externalIds["ArXiv"]
                paper = papers[f"ArXiv:{arxiv_id}"]
                paper.citation_count = result.citationCount
                paper.influential_citation_count = result.influentialCitationCount
                paper.semantic_scholar_id = result.paperId
                paper.venue = result.venue
                paper.journal_name = result.journal.name if result.journal else None
                paper.publication_types = (
                    result.publicationTypes if result.publicationTypes else []
                )
                paper.peer_reviewed = (
                    _infer_peer_reviewed(paper.publication_types, paper.journal_name)
                    or paper.peer_reviewed
                )

            time.sleep(self.request_delay_secs)

        return papers

    def title_match(self, paper: PaperRecord) -> Optional[Paper]:
        """
        Look up a paper by its title

        Args:
            paper: Paper-record

        Returns:
            Paper result if match found, else None
        """
        title = (paper.title or "").strip()

        if len(title) < _TITLE_MATCH_MIN_LEN:
            return None
        try:
            return self.client.search_paper(query=title, match_title=True)
        except Exception:
            return None


def _infer_peer_reviewed(
    publication_types: list[str], journal_name: str | None
) -> bool:
    """
    Infer if a paper is peer-reviewed

    Args:
        publication_types: List of publication types associated with a paper
        journal_name: Journal name associated with a paper

    Returns:
        Flag indicated if a paper has been reviewed
    """
    if journal_name:
        return True
    normalized = {t.lower().replace(" ", "") for t in publication_types if t}
    return any(tag in normalized for tag in {"journalarticle", "reviewarticle"})
