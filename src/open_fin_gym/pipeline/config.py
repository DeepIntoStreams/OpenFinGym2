from dataclasses import dataclass, field

from open_fin_gym.pipeline.db.tables import TaskType

from .steps.judge.config import JudgeConfig
from .steps.scrape_papers.config import ScrapingConfig
from .steps.task_extraction.config import TaskExtractionConfig


@dataclass
class Scope:
    id: str
    name: str
    task_type: TaskType
    description: str
    enabled: bool = True
    queries: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.task_type = TaskType(self.task_type)


def scope_context(scope: Scope) -> str:
    query_block = "\n".join(f"- {q}" for q in scope.queries[:10]) or "- (none)"
    categories = ", ".join(scope.categories) or "(none)"
    return (
        f"Scope name: {scope.name}\n"
        f"Scope description: {scope.description}\n"
        f"Scope categories: {categories}\n"
        f"Scope queries:\n{query_block}"
    )


@dataclass
class PipelineConfig:
    db_engine: str
    scraping: ScrapingConfig
    judge: JudgeConfig
    task_extractor: TaskExtractionConfig
    scopes: list[Scope]
