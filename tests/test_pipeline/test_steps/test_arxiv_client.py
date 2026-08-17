from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from open_fin_gym.pipeline.config import Scope
from open_fin_gym.pipeline.steps.scrape_papers.collection.arxiv import (
    ArxivClient,
)
from open_fin_gym.pipeline.steps.scrape_papers.collection.config import (
    ArxivConfig,
)


def fake_result(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        entry_id=f"http://arxiv.org/abs/2601.{index:05d}v1",
        title=f"Paper {index}",
        summary="abstract",
        authors=[],
        categories=["q-fin.CP"],
        published=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pdf_url=f"https://arxiv.org/pdf/2601.{index:05d}",
        doi=None,
        primary_category="q-fin.CP",
        journal_ref=None,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> ArxivClient:
    """Client whose queries return distinct papers, one per requested result."""
    client = ArxivClient(ArxivConfig())
    counter = iter(range(1000))

    def results(search):
        return [fake_result(next(counter)) for _ in range(search.max_results)]

    monkeypatch.setattr(client.client, "results", results)
    return client


def scope(n_queries: int) -> Scope:
    return Scope(
        id="scope",
        name="Scope",
        task_type="forecasting",
        description="",
        queries=[f'all:("term {i}")' for i in range(n_queries)],
    )


@pytest.mark.parametrize(
    "max_papers,n_queries",
    [(5, 3), (12, 3), (10, 4), (1, 1), (7, 7)],
)
def test_scope_budget_is_respected(
    client: ArxivClient, max_papers: int, n_queries: int
) -> None:
    papers = client.scrape_scope(scope(n_queries), None, None, max_papers)

    assert len(papers) == max_papers


def test_more_queries_than_budget_still_scrapes(client: ArxivClient) -> None:
    # Flooring the per-query budget previously returned nothing at all here
    papers = client.scrape_scope(scope(4), None, None, 2)

    assert len(papers) == 2
