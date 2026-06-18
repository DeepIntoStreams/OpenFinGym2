from dataclasses import dataclass

from .steps.judge.types import JudgeConfig
from .steps.scrape_papers.types import ScrapingConfig


@dataclass
class PipelineConfig:
    db_engine: str
    scraping: ScrapingConfig
    judge: JudgeConfig
