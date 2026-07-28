import logging
import time
from pathlib import Path

import pymupdf
import pymupdf4llm
import requests
from langchain_text_splitters import MarkdownHeaderTextSplitter
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from open_fin_gym.pipeline.db.tables import (
    Chunk,
    Paper,
    PaperStatus,
    RejectionReason,
    SourceFormat,
    SourceName,
)
from open_fin_gym.pipeline.db.utils import set_paper_status

from .config import RetrievalConfig
from .latexml import arxiv_html_to_markdown

logger = logging.getLogger(__name__)


class PaperRetrieval:
    def __init__(self, db: Engine, cfg: RetrievalConfig) -> None:
        """
        Paper retrieval and chunking stage

        Args:
            db: SqlAlchemy db engine
            cfg: Retrieval configuration
        """
        self.db = db
        self.cfg = cfg
        self.headers_to_split = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.splitter = MarkdownHeaderTextSplitter(
            self.headers_to_split, strip_headers=False
        )
        # One session keeps the connection to the source host alive across papers
        self.session = requests.Session()
        self.session.headers["User-Agent"] = cfg.user_agent
        self.next_request_at = 0.0

    def download_and_chunk_papers(self, output_path: Path) -> None:
        """
        Download newly scraped papers, chunk them and persist chunks in DB

        Args:
            output_path: Hydra run output path
        """
        with Session(self.db) as session:
            stmt = select(Paper).where(Paper.status == PaperStatus.SCRAPED)
            papers: list[Paper] = session.execute(stmt).scalars().all()
            stmt = select(Chunk.paper_id).distinct()
            chunked: set[str] = set(session.execute(stmt).scalars().all())

        logger.info(f"Extracting chunks from {len(papers)} papers")

        for paper in papers:
            if paper.paper_id in chunked:
                # Paper has already been chunked, potentially by a different scope
                set_paper_status(self.db, paper, PaperStatus.EXTRACTED)
                continue

            if not paper.pdf_url:
                # If the paper has no PDF link then reject here
                set_paper_status(
                    self.db,
                    paper,
                    PaperStatus.REJECTED,
                    rejection_reason=RejectionReason.NoPaperURL,
                )
                continue

            try:
                markdown, source_format = self.retrieve_markdown(paper)
                with Session(self.db) as session:
                    session.add_all(self.chunk(paper, markdown, source_format))
                    session.commit()
            except Exception as e:
                logger.error(
                    f"Paper {paper.paper_id} retrieval and chunking failed: {e}"
                )
                set_paper_status(
                    self.db,
                    paper,
                    PaperStatus.ERRORED,
                    rejection_reason=RejectionReason.RetrievalError,
                )
                continue

            chunked.add(paper.paper_id)
            set_paper_status(self.db, paper, PaperStatus.EXTRACTED)
            logger.debug(f"Paper {paper.paper_id} chunked from {source_format.value}")

    def retrieve_markdown(self, paper: Paper) -> tuple[str, SourceFormat]:
        """
        Retrieve a paper as markdown, trying each configured format in preference order

        Args:
            paper: Paper record

        Returns:
            Paper markdown and the format it was derived from

        Raises:
            RuntimeError: If the paper is unavailable in every configured format
        """
        for source_format in map(SourceFormat, self.cfg.source_preference):
            url = self.source_url(paper, source_format)
            if not url:
                continue

            source = self.get(url)
            if source is None:
                continue

            if source_format == SourceFormat.HTML:
                return arxiv_html_to_markdown(source), source_format

            return pdf_to_markdown(source), source_format

        raise RuntimeError(f"No source available for {paper.paper_id}")

    def source_url(self, paper: Paper, source_format: SourceFormat) -> str | None:
        """
        Build the download URL for a paper in a given format

        Args:
            paper: Paper record
            source_format: Format to download

        Returns:
            Download URL, or None if the paper cannot be served in this format
        """
        if source_format == SourceFormat.PDF:
            return paper.pdf_url

        # arXiv renders its LaTeX submissions to HTML, addressed by versioned id
        if paper.source != SourceName.ARXIV or not paper.arxiv_url:
            return None

        return self.cfg.arxiv_html_url.format(id=paper.arxiv_url.rsplit("/", 1)[-1])

    def get(self, url: str) -> bytes | None:
        """
        Fetch a URL, rate limited across the stage

        The interval runs from the previous request rather than from its completion,
        so time spent parsing the last response counts towards it instead of adding
        to it.

        Args:
            url: URL to fetch

        Returns:
            Response body, or None if the source was unavailable
        """
        time.sleep(max(0.0, self.next_request_at - time.monotonic()))
        self.next_request_at = time.monotonic() + self.cfg.request_interval_sec

        try:
            response = self.session.get(url, timeout=self.cfg.timeout_sec)
            response.raise_for_status()
        except requests.RequestException as e:
            # Papers with no HTML rendering 404 here and fall to the next format
            logger.debug(f"Could not retrieve {url}: {e}")
            return None

        return response.content

    def chunk(
        self, paper: Paper, markdown: str, source_format: SourceFormat
    ) -> list[Chunk]:
        """
        Split paper markdown into chunks on its section headers

        Args:
            paper: Paper record
            markdown: Paper markdown
            source_format: Format the markdown was derived from

        Returns:
            List of chunk records
        """
        sections = [x for x in self.splitter.split_text(markdown) if x.metadata]
        return [
            Chunk(
                paper_id=paper.paper_id,
                chunk_index=i,
                header=get_header(x.metadata),
                text=x.page_content,
                source_format=source_format,
            )
            for i, x in enumerate(sections)
        ]


def pdf_to_markdown(source: bytes) -> str:
    """
    Convert a PDF into markdown

    Args:
        source: Raw PDF bytes

    Returns:
        Markdown document with ATX headers
    """
    with pymupdf.open(stream=source, filetype="pdf") as document:
        return pymupdf4llm.to_markdown(document, header=False, footer=False)


def get_header(sections: dict[str, str]) -> str:
    """
    Get chunk section header tree string from metadata

    Args:
        sections: Dictionary of section IDs and their titles

    Returns:
        Joined section header tree
    """
    return " -- ".join([x.replace("*", "").lower().strip() for x in sections.values()])
