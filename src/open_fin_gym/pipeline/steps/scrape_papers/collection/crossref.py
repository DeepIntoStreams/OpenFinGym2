import logging
import re
import time
from difflib import SequenceMatcher

from habanero import Crossref

from open_fin_gym.pipeline.steps.scrape_papers.types import PaperRecord

from .config import CrossrefConfig

logger = logging.getLogger(__name__)

_SPACE = re.compile(r"\s+")


class CrossrefClient:
    def __init__(self, cfg: CrossrefConfig) -> None:
        """
        Crossref enrichment for DOI and lightweight venue metadata

        Args:
            cfg: Enrichment configuration
        """
        self.client = Crossref(
            base_url=cfg.base_url, mailto=cfg.mailto, timeout=cfg.timeout_sec
        )
        self.enabled = cfg.enabled
        self.max_results = cfg.max_results
        self.request_delay_secs = cfg.request_delay_secs
        self.min_title_similarity = cfg.min_title_similarity

    def enrich_doi(self, papers: dict[str, PaperRecord]) -> dict[str, PaperRecord]:
        """
        Fill missing DOI and backfill basic venue/quality metadata

        Args:
            papers: Dictionary of paper-records

        Returns:
            Dictionary with enriched paper records
        """
        if not self.enabled:
            return papers

        for paper in papers.values():
            if not _needs_crossref_lookup(paper):
                continue

            meta = self._lookup_metadata(paper.title, paper.authors[0])

            if meta:
                if not paper.doi:
                    paper.doi = meta.get("doi")
                if not paper.journal_name:
                    paper.journal_name = meta.get("journal_name")
                publication_type = meta.get("publication_type")
                if publication_type and publication_type not in paper.publication_types:
                    paper.publication_types.append(publication_type)
                if paper.citation_count is None:
                    paper.citation_count = meta.get("citation_count")
                if not paper.peer_reviewed:
                    paper.peer_reviewed = _infer_peer_reviewed(
                        publication_type, paper.journal_name
                    )

            time.sleep(self.request_delay_secs)

        return papers

    def _lookup_metadata(
        self, title: str, author: str
    ) -> dict[str, str | int | None] | None:
        """
        Search for papers with matching title and author, and return close matches

        Args:
            title: Paper title
            author: Lead author

        Returns:
            Metadata if a sufficient match is found, else None
        """
        results = self.client.works(
            select=[
                "DOI",
                "title",
                "container-title",
                "type",
                "is-referenced-by-count",
            ],
            limit=self.max_results,
            query_title=title,
            query_author=author,
        )

        items = results["message"]["items"]

        best_item: dict | None = None
        best_similarity = 0.0
        normalized_target = _normalize_title(title)

        for item in items:
            doi = (item.get("DOI") or "").strip().lower()
            item_titles = item.get("title") or []
            if not doi or not item_titles:
                continue
            candidate_title = _normalize_title(str(item_titles[0]))
            similarity = SequenceMatcher(
                None, normalized_target, candidate_title
            ).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_item = item

        if best_item and best_similarity >= self.min_title_similarity:
            journal_list = best_item.get("container-title") or []
            journal_name = str(journal_list[0]).strip() if journal_list else ""
            publication_type = str(best_item.get("type") or "").strip()
            citation_count = best_item.get("is-referenced-by-count")

            return {
                "doi": str(best_item.get("DOI") or "").strip().lower() or None,
                "journal_name": journal_name or None,
                "publication_type": publication_type or None,
                "citation_count": citation_count
                if isinstance(citation_count, int)
                else None,
            }

        return None


def _normalize_title(value: str) -> str:
    """
    Normalise title text before fuzzy matching

    Args:
        value: Raw title

    Returns:
        Normalised title string
    """
    return _SPACE.sub(" ", value.lower()).strip()


def _infer_peer_reviewed(
    publication_type: str | None, journal_name: str | None
) -> bool:
    """
    Infer if a paper is peer-reviewed

    Args:
        publication_type: Publication type associated with a paper record
        journal_name: Journal name associated with a paper record

    Returns:
        Flag indicating whether a paper is inferred to be peer-reviewed
    """
    if journal_name:
        return True
    if not publication_type:
        return False
    return publication_type.replace("-", "").lower() in {
        "journalarticle",
        "reviewarticle",
    }


def _needs_crossref_lookup(paper: PaperRecord) -> bool:
    """
    Flag if a paper-record is a candidate for enrichment using CrossRef

    Args:
        paper: Paper-record

    Returns:
        Boolean flag indicating if a paper can be enriched
    """
    if not paper.title.strip():
        return False
    if not paper.doi:
        return True
    if not paper.journal_name:
        return True
    if paper.citation_count is None:
        return True
    if not paper.publication_types:
        return True
    if not paper.peer_reviewed:
        return True
    return False
