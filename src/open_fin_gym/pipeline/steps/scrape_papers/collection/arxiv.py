import logging
import math
from datetime import datetime, timezone
from itertools import islice

import arxiv

from open_fin_gym.pipeline.config import Scope
from open_fin_gym.pipeline.db.tables import SourceName
from open_fin_gym.pipeline.steps.scrape_papers.types import PaperRecord

from .config import ArxivConfig

logger = logging.getLogger(__name__)


class ArxivClient:
    def __init__(self, cfg: ArxivConfig) -> None:
        """
        Arxiv scraping client

        Args:
            cfg: Scraper configuration
        """
        self.client = arxiv.Client(
            page_size=cfg.page_size,
            delay_seconds=cfg.request_interval_sec,
            num_retries=cfg.num_retries,
        )
        self.sort_criteria = arxiv.SortCriterion[cfg.sort_by]

    def scrape_scope(
        self,
        scope: Scope,
        since_date: datetime | None,
        until_date: datetime | None,
        max_papers: int,
    ) -> dict[str, PaperRecord]:
        """
        Scrape paper records for a given scope

        Args:
            scope: Scope configuration
            since_date: Optional date to search from
            until_date: Optional date to search to
            max_papers: Max number of papers to return across the scope's queries

        Returns:
            Dictionary of paper records indexed by their id
        """
        papers: dict[str, PaperRecord] = dict()

        categories = scope.categories
        date_range = _to_arxiv_date_range(since_date, until_date)

        # Round up so a scope with more queries than budget still returns papers
        max_per_query = math.ceil(max_papers / len(scope.queries))

        for query in scope.queries:
            # Queries carry their own field prefix, e.g. all: or ti:
            search_parts = [query]
            if scope.categories:
                cat_filter = " OR ".join(f"cat:{c}" for c in categories)
                search_parts.append(f"({cat_filter})")
            if date_range:
                search_parts.append(f"submittedDate:[{date_range}]")

            search = arxiv.Search(
                query=" AND ".join(search_parts),
                max_results=max_per_query,
                sort_by=self.sort_criteria,
                sort_order=arxiv.SortOrder.Descending,
            )
            results = self.client.results(search)

            for result in results:
                paper_id = _format_arxiv_id(result.entry_id)
                journal_ref = result.journal_ref
                papers[paper_id] = PaperRecord(
                    paper_id=_format_arxiv_id(result.entry_id),
                    scope_id=scope.id,
                    arxiv_url=result.entry_id,
                    title=result.title,
                    abstract=result.summary,
                    source=SourceName.ARXIV,
                    authors=[x.name for x in result.authors],
                    categories=result.categories,
                    published_at=result.published,
                    updated_at=result.updated,
                    pdf_url=result.pdf_url,
                    doi=result.doi,
                    primary_category=result.primary_category,
                    journal_name=journal_ref,
                    publication_types=["journal_reference"] if journal_ref else [],
                    peer_reviewed=bool(journal_ref),
                )

        # Queries overlap, so the budget is applied to the deduplicated result
        return dict(islice(papers.items(), max_papers))


def _to_arxiv_date_range(start: datetime | None, end: datetime | None) -> str | None:
    """
    Build arXiv submittedDate range clause

    Args:
        start: Start datetime, defaults to 1/1/1991 if None
        end: End datetime, defaults to now if None

    Returns:
        Query string, or None if both start and end are None
    """
    if not start and not end:
        return None
    if start is None:
        start = datetime(1991, 1, 1, tzinfo=timezone.utc)
    if end is None:
        end = datetime.now(tz=timezone.utc)
    return f"{start.strftime('%Y%m%d')}0000 TO {end.strftime('%Y%m%d')}2359"


def _format_arxiv_id(arxiv_url: str) -> str:
    """
    Convert an arxiv url into an appropriate id

    Args:
        arxiv_url: Arxiv url, e.g. http://arxiv.org/abs/2602.00086v3

    Returns:
        Id of the form 'ArXiv:2602.00086'
    """
    return f"ArXiv:{arxiv_url.split('/')[-1].split('v')[0]}"
