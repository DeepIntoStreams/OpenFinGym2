import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from sqlalchemy import Engine, insert, select, update
from sqlalchemy.orm import Session

from open_fin_gym.pipeline.db.tables import Chunk, Paper
from open_fin_gym.pipeline.steps.scrape_papers.types import PaperStatus


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
            papers = session.execute(stmt).scalars().all()

        for paper in papers:
            if not paper.pdf_url:
                # If the paper has no pdf link then reject here
                status = PaperStatus.REJECTED
                chunks = []
            else:
                pdf_file = tempfile.NamedTemporaryFile()
                _, response = urlretrieve(paper.pdf_url, pdf_file.name)
                md = pymupdf4llm.to_markdown(pdf_file)
                chunks = self.splitter.split_text(md)
                status = PaperStatus.EXTRACTED
                chunks = [
                    dict(
                        paper_id=paper.paper_id,
                        chunk_index=i,
                        header=get_header(x.metadata),
                        text=x.page_content,
                    )
                    for i, x in enumerate(chunks)
                    if x.metadata
                ]

            with Session(self.db) as session:
                stmt = insert(Chunk).values(chunks)
                session.execute(stmt)
                stmt = update(Paper).values({"status": status})
                stmt = stmt.where(Paper.paper_id == paper.paper_id)
                session.execute(stmt)
                session.commit()


def get_header(sections: dict[str, str]) -> str:
    """
    Get chunk section header from metadata

    Args:
        sections: Dictionary of section IDs and their titles

    Returns:
        The actual heading name of the chunk
    """
    return sorted([(k, v) for k, v in sections.items()], key=lambda x: x[0])[-1][1]
