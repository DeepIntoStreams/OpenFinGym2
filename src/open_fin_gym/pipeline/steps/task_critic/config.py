from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass
class TaskCriticConfig:
    threshold_default: float
    llm: DictConfig
