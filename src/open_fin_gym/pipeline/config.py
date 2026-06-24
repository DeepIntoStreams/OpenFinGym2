from dataclasses import dataclass, field

from .steps.judge.config import JudgeConfig
from .steps.scrape_papers.config import ScrapingConfig


@dataclass
class Scope:
    id: str
    name: str
    description: str
    enabled: bool = True
    queries: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    db_engine: str
    scraping: ScrapingConfig
    judge: JudgeConfig
    scopes: list[Scope]
