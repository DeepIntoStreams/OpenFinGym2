from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from open_fin_gym.pipeline.db.tables import Paper
from open_fin_gym.pipeline.steps.scrape_papers.pipeline import (
    PaperScrapingPipeline,
)
from open_fin_gym.pipeline.steps.scrape_papers.types import (
    ArxivConfig,
    CrossrefConfig,
    PaperRecord,
    ScrapingConfig,
    SemanticScholarConfig,
)


def test_paper_insert(db: Engine) -> None:
    pipeline = PaperScrapingPipeline(
        db,
        ScrapingConfig(
            root_dir="",
            scopes=[],
            arxiv=ArxivConfig(),
            semantic_scholar=SemanticScholarConfig(),
            crossref=CrossrefConfig(),
            max_papers_per_scope=0,
            max_accepts_per_scope=0,
            since="",
            until="",
        ),
    )

    paper_a = PaperRecord(
        paper_id="foo",
        scope_id="bar",
        title="foo-bar",
        abstract="abstract",
    )

    pipeline.insert_new_papers({"foo": paper_a})

    with Session(db) as session:
        papers_a = session.execute(select(Paper)).scalars().all()

    assert len(papers_a) == 1
    assert papers_a[0].paper_id == "foo"

    # Insert new paper
    paper_b = PaperRecord(
        paper_id="baz",
        scope_id="bar",
        title="baz-bar",
        abstract="abstract",
    )

    pipeline.insert_new_papers({"baz": paper_b})

    with Session(db) as session:
        papers_b = session.execute(select(Paper)).scalars().all()

    assert len(papers_b) == 2
    assert papers_b[0].paper_id == "foo"
    assert papers_b[1].paper_id == "baz"

    # Insert existing
    paper_c = PaperRecord(
        paper_id="foo",
        scope_id="bar",
        title="foo-bar-new",
        abstract="abstract",
    )

    pipeline.insert_new_papers({"foo": paper_c})

    with Session(db) as session:
        papers_c = session.execute(select(Paper)).scalars().all()

    assert len(papers_c) == 2
    assert papers_c[0].paper_id == "foo"
    assert papers_c[1].paper_id == "baz"
    assert papers_c[0].title == "foo-bar"
