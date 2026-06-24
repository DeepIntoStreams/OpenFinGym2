from dataclasses import dataclass
from typing import Optional

from open_fin_gym.pipeline.steps.scrape_papers.collection.config import (
    ArxivConfig,
    CrossrefConfig,
    SemanticScholarConfig,
)


@dataclass
class ScrapingConfig:
    arxiv: ArxivConfig
    semantic_scholar: SemanticScholarConfig
    crossref: CrossrefConfig
    max_papers_per_scope: int
    max_accepts_per_scope: int
    since: str
    until: str
    max_accepts: Optional[int] = None
