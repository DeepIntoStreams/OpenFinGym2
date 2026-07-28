import pytest
from sqlalchemy import Engine

from open_fin_gym.pipeline.db.tables import Paper, SourceFormat, SourceName
from open_fin_gym.pipeline.steps.retrieval.config import RetrievalConfig
from open_fin_gym.pipeline.steps.retrieval.latexml import (
    arxiv_html_to_markdown,
)
from open_fin_gym.pipeline.steps.retrieval.pipeline import PaperRetrieval

# Mirrors how LaTeXML renders a paper: maths carries its LaTeX in `alttext`, and a
# tabular is emitted either as a `table` or, inside an inline context, as a `span`.
ARXIV_HTML = b"""
<html><body><article>
<h1 class="ltx_title ltx_title_document">A Paper</h1>
<h6 class="ltx_title ltx_title_abstract">Abstract</h6>
<p class="ltx_p">We predict <math alttext="r_{t+1}">rt+1</math> returns.</p>
<script>ignored()</script>
<h2 class="ltx_title ltx_title_section">1 Data</h2>
<p class="ltx_p">Available at <a href="https://example.com/data.csv">here</a>.</p>
<h5 class="ltx_title ltx_title_paragraph">Run-in title.</h5>
<figure class="ltx_table">
<figcaption>Table 1: Results.</figcaption>
<table class="ltx_tabular">
<tr class="ltx_tr"><th class="ltx_td ltx_th">Model</th><th class="ltx_td ltx_th">RMSE</th></tr>
<tr class="ltx_tr"><td class="ltx_td">LSTM</td><td class="ltx_td">0.041</td></tr>
</table>
</figure>
<h3 class="ltx_title ltx_title_subsection">1.1 Splits</h3>
<p class="ltx_p"><span class="ltx_tabular">
<span class="ltx_tr"><span class="ltx_td">Train</span><span class="ltx_td">Test</span></span>
<span class="ltx_tr"><span class="ltx_td">800</span><span class="ltx_td">200</span></span>
</span></p>
</article></body></html>
"""


@pytest.fixture
def dummy_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(
        source_preference=[SourceFormat.HTML, SourceFormat.PDF],
        arxiv_html_url="https://arxiv.org/html/{id}",
        request_interval_sec=0.0,
        timeout_sec=1.0,
        user_agent="test",
    )


def test_html_headers_drive_chunking(
    db: Engine, dummy_retrieval_config: RetrievalConfig
) -> None:
    pipeline = PaperRetrieval(db, dummy_retrieval_config)
    paper = Paper(paper_id="ArXiv:1", scope_id="scope")

    markdown = arxiv_html_to_markdown(ARXIV_HTML)
    chunks = pipeline.chunk(paper, markdown, SourceFormat.HTML)

    assert [x.chunk_index for x in chunks] == list(range(len(chunks)))
    assert all(x.source_format == SourceFormat.HTML for x in chunks)

    headers = [x.header for x in chunks]
    assert "a paper -- abstract" in headers
    assert "a paper -- 1 data" in headers
    assert "a paper -- 1 data -- 1.1 splits" in headers
    # Run-in paragraph titles are body text and must not open a chunk
    assert not any("run-in" in x for x in headers)


def test_html_preserves_maths_tables_and_links() -> None:
    markdown = arxiv_html_to_markdown(ARXIV_HTML)

    assert "$r_{t+1}$" in markdown
    assert "ignored()" not in markdown

    # Both tabular renderings survive as markdown tables, each emitted exactly once
    assert "| Model | RMSE |" in markdown
    assert markdown.count("| LSTM | 0.041 |") == 1
    assert markdown.count("| 800 | 200 |") == 1


@pytest.mark.parametrize(
    "source_format,source,expected",
    [
        (SourceFormat.HTML, SourceName.ARXIV, "https://arxiv.org/html/2601.00738v1"),
        (SourceFormat.PDF, SourceName.ARXIV, "https://arxiv.org/pdf/2601.00738"),
        (SourceFormat.HTML, SourceName.MANUAL, None),
    ],
)
def test_source_url(
    db: Engine,
    dummy_retrieval_config: RetrievalConfig,
    source_format: SourceFormat,
    source: SourceName,
    expected: str | None,
) -> None:
    pipeline = PaperRetrieval(db, dummy_retrieval_config)
    paper = Paper(
        paper_id="ArXiv:2601.00738",
        scope_id="scope",
        source=source,
        arxiv_url="http://arxiv.org/abs/2601.00738v1",
        pdf_url="https://arxiv.org/pdf/2601.00738",
    )

    assert pipeline.source_url(paper, source_format) == expected
