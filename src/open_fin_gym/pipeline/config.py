from dataclasses import dataclass

from .steps.judge.types import JudgeConfig
from .steps.scrape_papers.types import ScrapingConfig
from .steps.task_extraction.types import TaskExtractionConfig


@dataclass
class PipelineConfig:
    db_engine: str
    scraping: ScrapingConfig
    judge: JudgeConfig
    task_extractor: TaskExtractionConfig
