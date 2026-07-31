from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass
class TaskGenerationConfig:
    llm: DictConfig
    templates_path: str
