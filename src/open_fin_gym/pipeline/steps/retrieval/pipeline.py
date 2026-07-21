import logging
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from open_fin_gym.pipeline.db.tables import (
    Chunk,
    Paper,
    PaperStatus,
    RejectionReason,
)
from open_fin_gym.pipeline.db.utils import set_paper_status

logger = logging.getLogger(__name__)


class PaperRetrieval:
    def __init__(self, db: Engine) -> None:
        """
        Paper retrieval and chunking stage

        Args:
            db: SqlAlchemy db engine
        """
        self.db = db
        self.headers_to_split = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.splitter = MarkdownHeaderTextSplitter(
            self.headers_to_split, strip_headers=False
        )

    def download_and_chunk_papers(self, output_path: Path) -> None:
        """
        Download PDF for newly scraped papers, chunk them and persist chunks in DB

        Args:
            output_path: Hydra run output path
        """
        with Session(self.db) as session:
            stmt = select(Paper).where(Paper.status == PaperStatus.SCRAPED)
            papers: list[Paper] = session.execute(stmt).scalars().all()

        logger.info(f"Extracting chunks from {len(papers)} papers")

        for paper in papers:
            with Session(self.db) as session:
                paper_exists = (
                    session.query(Chunk).filter_by(paper_id=paper.paper_id).first()
                    is not None
                )

            if paper_exists:
                # Paper with the same id has already been inserted (potentially by a different scope)
                status = PaperStatus.EXTRACTED
                rejection_reason = None
                chunks = []

            elif not paper.pdf_url:
                # If the paper has no PDF link then reject here
                status = PaperStatus.REJECTED
                rejection_reason = RejectionReason.NoPaperURL
                chunks = []

            else:
                pdf_file = tempfile.NamedTemporaryFile()

                try:
                    _, response = urlretrieve(paper.pdf_url, pdf_file.name)
                    md = pymupdf4llm.to_markdown(pdf_file, header=False, footer=False)
                    chunks = self.splitter.split_text(md)
                    status = PaperStatus.EXTRACTED
                    rejection_reason = None
                    chunks = [
                        Chunk(
                            paper_id=paper.paper_id,
                            chunk_index=i,
                            header=get_header(x.metadata),
                            text=x.page_content,
                        )
                        for i, x in enumerate(chunks)
                        if x.metadata
                    ]
                except Exception as e:
                    logger.error(
                        f"Paper {paper.paper_id} PDF retrieval and chunking from {paper.pdf_url} failed: {e}"
                    )
                    status = PaperStatus.ERRORED
                    rejection_reason = RejectionReason.RetrievalError
                    chunks = []

            set_paper_status(self.db, paper, status, rejection_reason=rejection_reason)

            with Session(self.db) as session:
                session.add_all(chunks)
                session.execute(stmt)
                session.commit()


def get_header(sections: dict[str, str]) -> str:
    """
    Get chunk section header tree string from metadata

    Args:
        sections: Dictionary of section IDs and their titles

    Returns:
        Joined section header tree
    """
    return " -- ".join([x.replace("*", "").lower().strip() for x in sections.values()])
