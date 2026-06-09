from dataclasses import dataclass

from .steps.scrape_papers.types import ScrapingConfig


@dataclass
class PipelineConfig:
    db_engine: str
    scraping: ScrapingConfig
