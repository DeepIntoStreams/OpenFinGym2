from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig
from pydantic import BaseModel

from open_fin_gym.pipeline.db.tables import RejectionReason


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


class JudgeResult(BaseModel):
    paper_id: str
    scope_id: str
    accepted: bool
    rejection_reason: Optional[RejectionReason]
    reasoning: str
