from typing import Any, Optional

import pymupdf
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter

from open_fin_gym.pipeline.db.tables import Chunk, SourceFormat

DEFAULT_HEADERS_TO_SPLIT = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
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


def pdf_file_to_markdown(path: str) -> str:
    with pymupdf.open(filename=path, filetype="pdf") as document:
        return pymupdf4llm.to_markdown(document, header=False, footer=False)


def chunk_markdown(
    paper_id: str,
    markdown: str,
    source_format: SourceFormat,
    splitter: Optional[Any] = None,
) -> list[Chunk]:
    if splitter is None:
        splitter = MarkdownHeaderTextSplitter(
            DEFAULT_HEADERS_TO_SPLIT, strip_headers=False
        )

    sections = [x for x in splitter.split_text(markdown) if x.metadata]
    return [
        Chunk(
            paper_id=paper_id,
            chunk_index=i,
            header=get_header(x.metadata),
            text=x.page_content,
            source_format=source_format,
        )
        for i, x in enumerate(sections)
    ]


def get_header(sections: dict[str, str]) -> str:
    """
    Get chunk section header tree string from metadata

    Args:
        sections: Dictionary of section IDs and their titles

    Returns:
        Joined section header tree
    """
    return " -- ".join([x.replace("*", "").lower().strip() for x in sections.values()])
