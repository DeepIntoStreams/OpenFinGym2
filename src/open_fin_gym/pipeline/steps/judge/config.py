from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass
class JudgeConfig:
    sift_budget: int
    prefilter_enabled: bool
    prefilter_llm_score_min: float
    prefilter_llm_confidence_min: float
    ranking_citation_boost: float
    ranking_recency_weight: float
    ranking_recency_half_life_days: int
    threshold_default: float
    llm: DictConfig
