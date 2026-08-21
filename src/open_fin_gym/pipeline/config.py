from dataclasses import dataclass, field

from open_fin_gym.pipeline.db.tables import TaskType

from .steps.judge.config import JudgeConfig
from .steps.retrieval.config import RetrievalConfig
from .steps.scrape_papers.config import ScrapingConfig
from .steps.task_critic.config import TaskCriticConfig
from .steps.task_export.config import TaskExportConfig
from .steps.task_extraction.config import TaskExtractionConfig
from .steps.task_generator.config import TaskGenerationConfig


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
        f"Scope task type: {scope.task_type}\n"
        f"Scope description: {scope.description}\n"
        f"Scope categories: {categories}\n"
        f"Scope queries:\n{query_block}"
    )


@dataclass
class PipelineConfig:
    db_engine: str
    scraping: ScrapingConfig
    retrieval: RetrievalConfig
    judge: JudgeConfig
    task_extractor: TaskExtractionConfig
    task_critic: TaskCriticConfig
    task_generator: TaskGenerationConfig
    task_exporter: TaskExportConfig
    scopes: list[Scope]
